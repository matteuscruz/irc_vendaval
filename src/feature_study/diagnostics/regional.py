"""Treinar o "Corrigido ML" (campeões do cluster 3 por trimestre + EQM) em qualquer conjunto de estações e avaliar
por cluster. Usado para RS/SC (clusters 1–4): um modelo por cluster, um regional e o regional deixando um
cluster de fora (LOCO).

O modelo de cada trimestre é o campeão do cluster 3 (`summary/top_models.json`, rank 1), ajustado pelo MESMO
pipeline do estudo (`worker.fit_arm`, LazyPredict com o regressor fixo, seed 42). O mapeamento de quantis
(`quantile_map`, EQM) é ajustado na VALIDAÇÃO do próprio conjunto de treino. Teste (jan/abr/jul/out) e 2025 nunca
entram no ajuste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.core.worker import fit_arm, select_regressors
from src.feature_study.data.groups_source import REFERENCE_COLUMN
from src.feature_study.diagnostics import inference as inf
from src.feature_study.diagnostics import quantile_map as qm
from src.pipelines.common import TARGET_VAR

SEASONS = ("DJF", "MAM", "JJA", "SON")
AVALIA = ("test", "novo")          # meses de teste de 2000–2024 e 2025


def fit_set(df: pd.DataFrame, columns, champion: dict[str, str], seed: int = 42) -> dict:
    """Ajusta, por trimestre, o campeão em `df[_split == train]` e o EQM em `df[_split == val]`."""
    columns = list(columns)
    out = {}
    for s in SEASONS:
        d = df[df.season == s]
        tr = d[d._split == "train"].reset_index(drop=True)
        va = d[d._split == "val"].reset_index(drop=True)
        if tr.empty or va.empty:
            raise ValueError(f"{s}: treino ({len(tr)}) ou validação ({len(va)}) vazios")
        pipes = {}
        fit_arm(tr, va, va, columns, select_regressors([champion[s]]), seed=seed,
                resid_models={champion[s]}, fitted_out=pipes)
        f = {"model": champion[s], "pipe": pipes[champion[s]], "train": tr[columns],
             "n_train": len(tr), "n_val": len(va)}
        f["mapa"] = qm.fit(inf.predict_grid(va, {s: f}, columns), va[TARGET_VAR])
        out[s] = f
    return out


DOY_COLS = ("doy_sin", "doy_cos")
ANUAL = "ALL"                      # "trimestre" do modelo único do ano todo
MODELO_ANUAL = "CatBoostRegressor"  # o único entre os 5 melhores nos quatro trimestres do estudo


def with_doy(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta seno e cosseno do dia do ano (2π·DOY/365,2425, como no spec): é como o modelo único sabe a época."""
    a = 2 * np.pi * df["time"].dt.dayofyear.to_numpy() / 365.2425
    return df.assign(doy_sin=np.sin(a), doy_cos=np.cos(a))


def fit_single(df: pd.DataFrame, columns, model: str = MODELO_ANUAL, seed: int = 42) -> dict:
    """UM modelo para todos os trimestres (`df[_split == train]`) e UM EQM ajustado na validação inteira.

    Devolve no formato de `fit_set` com a chave única `ANUAL`, para `predict(df.assign(season=ANUAL), …)`."""
    columns = list(columns)
    tr = df[df._split == "train"].reset_index(drop=True)
    va = df[df._split == "val"].reset_index(drop=True)
    pipes = {}
    fit_arm(tr, va, va, columns, select_regressors([model]), seed=seed, resid_models={model}, fitted_out=pipes)
    f = {"model": model, "pipe": pipes[model], "train": tr[columns], "n_train": len(tr), "n_val": len(va)}
    f["mapa"] = qm.fit(inf.predict_grid(va.assign(season=ANUAL), {ANUAL: f}, columns), va[TARGET_VAR])
    return {ANUAL: f}


def predict(df: pd.DataFrame, fitted: dict, columns, chunk: int = 250_000) -> tuple[np.ndarray, np.ndarray]:
    """(modelo sem EQM, Corrigido ML = modelo + EQM) para cada linha de `df`.

    Em blocos de `chunk` linhas: a grade de RS/SC tem ~5 milhões de linhas e prever tudo de uma vez com o
    ExtraTrees regional estoura a memória da máquina."""
    columns = list(columns)
    bruto = np.full(len(df), np.nan, dtype="float32")
    for i in range(0, len(df), chunk):
        bloco = df.iloc[i:i + chunk]
        bruto[i:i + len(bloco)] = inf.predict_grid(bloco, fitted, columns)
    mapeado = np.full(len(df), np.nan, dtype="float32")
    for s, f in fitted.items():
        m = (df["season"] == s).to_numpy()
        mapeado[m] = qm.apply(bruto[m], f["mapa"])
    return bruto, mapeado


def _metricas(d: pd.DataFrame, col: str) -> dict:
    y, p = d[TARGET_VAR].to_numpy(float), d[col].to_numpy(float)
    ok = np.isfinite(p)
    d, y, p = d[ok], y[ok], p[ok]
    out = {"n": len(d), "estacoes": d.estacao.nunique(), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
           "vies": float(np.mean(p - y)), "corr": float(np.corrcoef(p, y)[0, 1]) if len(d) > 2 else np.nan}
    for nome, q in (("P95", 0.95), ("P99", 0.99)):
        lim = d.groupby("estacao")[TARGET_VAR].transform(lambda x: x.quantile(q)).to_numpy()
        ev, al = y >= lim, p >= lim
        out[f"POD_{nome}"] = float((ev & al).sum() / max(ev.sum(), 1))
        out[f"FAR_{nome}"] = float((~ev & al).sum() / max(al.sum(), 1))
    e = pd.DataFrame({"est": d.estacao.to_numpy(), "y": y, "p": p}).groupby("est").mean()
    out["inclinacao_media"] = float(np.polyfit(e.y, e.p, 1)[0]) if len(e) >= 3 else np.nan
    return out


def evaluate(df: pd.DataFrame, preds: dict[str, np.ndarray], *, rotulo: dict) -> pd.DataFrame:
    """Métricas por cluster avaliado × trimestre (e "Todos") × versão, só em `_split ∈ AVALIA`.

    `preds` = {versão: previsão alinhada com `df`}; o ERA5 entra sempre. `rotulo` vai para todas as linhas
    (ex.: {"desenho": "regional", "treino": "1+2+3+4"})."""
    d = df.assign(**{k: v for k, v in preds.items()}, ERA5=df[REFERENCE_COLUMN].to_numpy(float))
    d = d[d._split.isin(AVALIA)]
    linhas = []
    for (c, rec), g in d.groupby(["cluster_id", np.where(d._split == "novo", "2025", "teste 2000–2024")]):
        for esc, gg in [("Todos", g), *g.groupby("season")]:
            for versao in ("ERA5", *preds):
                linhas.append({**rotulo, "cluster": int(c), "recorte": rec, "trimestre": esc, "versao": versao,
                               **_metricas(gg, versao)})
    return pd.DataFrame(linhas)


RS_SC = (1, 2, 3, 4)
GLOBAL = (1, 2, 3, 4, 5, 12)


def designs(clusters=GLOBAL, regional=RS_SC) -> list[dict]:
    """Os desenhos de treino, iguais para o ML e a LSTM.

    B  local: cada cluster treinado só com as próprias estações, avaliado em todos (matriz treino × teste);
    C  regional: os clusters de `regional` (RS/SC) juntos, avaliado em todos (no 12, é região não vista);
    G  global: todos os `clusters` juntos, avaliado em todos;
    L  LOCO global: todos menos um, avaliado só no que ficou de fora (região nunca vista);
    GA global anual: todos os `clusters` e TODOS os trimestres num modelo só (com o dia do ano como coluna) e um
       EQM só — contra o G, que tem um modelo e um EQM por trimestre.
    """
    out = [{"chave": f"B{c}", "desenho": "B: local", "treino": str(c), "train": (c,), "avalia": tuple(clusters)}
           for c in clusters]
    out.append({"chave": "C", "desenho": "C: regional RS/SC", "treino": "+".join(map(str, regional)),
                "train": tuple(regional), "avalia": tuple(clusters)})
    out.append({"chave": "G", "desenho": "G: global", "treino": "+".join(map(str, clusters)),
                "train": tuple(clusters), "avalia": tuple(clusters)})
    for c in clusters:
        outros = tuple(x for x in clusters if x != c)
        out.append({"chave": f"L{c}", "desenho": "L: global sem o cluster avaliado", "treino": "+".join(map(str, outros)),
                    "train": outros, "avalia": (c,)})
    out.append({"chave": "GA", "desenho": "GA: global anual (1 modelo)", "treino": "+".join(map(str, clusters)),
                "train": tuple(clusters), "avalia": tuple(clusters), "anual": True})
    return out
