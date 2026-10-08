"""Todos os modelos de ML do estudo juntos: os tabulares (árvores, lineares, MLP) e a LSTM.

Cada família tem as suas unidades (`units/<tag>` para os tabulares, uma por seed; `units/<tag>_lstm` para a
LSTM) e o seu `aggregate`. Aqui elas entram na MESMA tabela, por modelo, para responder "a LSTM é melhor ou pior
que os demais, e o que acrescentar features faz com cada um".

Duas leituras, ambas sobre os resíduos e métricas já gravados (nada é retreinado):

  - erro de teste por modelo × arm × trimestre, com a rajada ERA5 bruta como referência nas mesmas linhas;
  - o efeito `full` contra `base` POR MODELO, com o mesmo bootstrap pareado em blocos (ano, mês) do estudo.

Cada modelo só existe nos trimestres em que foi eleito (os 5 melhores de cada trimestre), então a comparação
entre modelos é feita POR TRIMESTRE e a média entre trimestres vale só onde o modelo existe: `n_trimestres` diz
quantos. O LSTM existe nos quatro. A LSTM costuma ter uma seed e os tabulares cinco (a média das seeds é por
modelo e arm): a diferença de seeds fica registrada em `n_seeds`.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.feature_study.core.analysis import compute_effects, load_arms
from src.feature_study.core.config import MODEL_FAMILY
from src.feature_study.core.worker_lstm import LSTM_NAME
from src.feature_study.data.groups_source import REFERENCE_COLUMN
from src.pipelines.common import TARGET_VAR

FAMILIA_LSTM = "LSTM"


def familia(modelo: str) -> str:
    return FAMILIA_LSTM if modelo == LSTM_NAME else MODEL_FAMILY.get(modelo, "outros")


def load_metrics(study_dir, tags, arms=None) -> pd.DataFrame:
    """Métricas de TESTE de todas as unidades dos `tags`, uma linha por (tag, modelo, arm, trimestre)."""
    partes = []
    for t in tags:
        for p in glob.glob(str(Path(study_dir) / "units" / t / "metrics__*__*.parquet")):
            m = pd.read_parquet(p)
            m = m[m["split"] == "test"]
            if arms is not None:
                m = m[m["arm"].isin(arms)]
            partes.append(m[["arm", "season", "model_seed", "model", "n", "R2", "RMSE", "RMSE_P90", "Bias"]])
    if not partes:
        raise FileNotFoundError(f"nenhuma métrica de teste em {study_dir}/units/{{{','.join(tags)}}}")
    return pd.concat(partes, ignore_index=True)


def era5_by_season(data_dir) -> pd.Series:
    """RMSE da rajada ERA5 bruta contra o INMET, por trimestre, no teste."""
    t = pd.read_parquet(Path(data_dir) / "test.parquet", columns=["season", TARGET_VAR, REFERENCE_COLUMN])
    e = t[REFERENCE_COLUMN] - t[TARGET_VAR]
    return (e ** 2).groupby(t["season"]).mean().pow(0.5).rename("era5_rmse")


def per_season(metrics: pd.DataFrame, era5: pd.Series) -> pd.DataFrame:
    """RMSE por (modelo, arm, trimestre), média entre as seeds, com a família e o ERA5 ao lado."""
    g = (metrics.groupby(["model", "arm", "season"])
         .agg(RMSE=("RMSE", "mean"), RMSE_P90=("RMSE_P90", "mean"), R2=("R2", "mean"),
              n_seeds=("model_seed", "nunique"), n=("n", "first")).reset_index())
    g["familia"] = g["model"].map(familia)
    g["era5_rmse"] = g["season"].map(era5)
    g["ganho_sobre_era5"] = g["era5_rmse"] - g["RMSE"]
    return g


def summary(ps: pd.DataFrame) -> pd.DataFrame:
    """Média entre os trimestres em que o modelo existe, por (modelo, arm)."""
    s = (ps.groupby(["model", "familia", "arm"])
         .agg(RMSE=("RMSE", "mean"), RMSE_P90=("RMSE_P90", "mean"), ganho_sobre_era5=("ganho_sobre_era5", "mean"),
              n_trimestres=("season", "nunique"), n_seeds=("n_seeds", "max")).reset_index())
    return s.sort_values(["arm", "RMSE"]).reset_index(drop=True)


def effects_by_model(study_dir, tags_por_familia: dict[str, list[str]], arms=("base", "full"),
                     n_boot: int = 2000) -> pd.DataFrame:
    """Efeito `full` contra `base` POR MODELO (ganho de RMSE; positivo = o `full` é melhor), com IC do
    bootstrap pareado em blocos. Cada família usa as próprias unidades (`tags`)."""
    study_dir = Path(study_dir)
    data_dir = study_dir / "data"
    ams = [a for a in load_arms(data_dir) if a.name in arms]
    linhas = []
    for fam, tags in tags_por_familia.items():
        modelos = sorted(load_metrics(study_dir, tags, arms)["model"].unique())
        for m in modelos:
            eff = compute_effects(study_dir, data_dir, tags, ams, n_boot=n_boot, models=[m])
            r = eff[(eff["comparison"] == "full_vs_base") & (eff["metric"] == "rmse") & (eff["family"] == "all")]
            if r.empty:
                continue
            r = r.iloc[0]
            linhas.append({"model": m, "familia": familia(m), "efeito_full_vs_base": float(r["effect"]),
                           "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]), "sesoi": float(r["sesoi"]),
                           "veredito": r["verdict"], "n_tags": len(tags)})
    return pd.DataFrame(linhas).sort_values("efeito_full_vs_base").reset_index(drop=True)


def run(study_dir, tree_tags, lstm_tags, out_dir=None, n_boot: int = 2000, arms=("base", "full")) -> dict:
    """Gera as tabelas e as grava em `<estudo>/todos_os_modelos/`."""
    study_dir = Path(study_dir)
    out = Path(out_dir) if out_dir else study_dir / "todos_os_modelos"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([load_metrics(study_dir, tree_tags, arms), load_metrics(study_dir, lstm_tags, arms)],
                        ignore_index=True)
    ps = per_season(metrics, era5_by_season(study_dir / "data"))
    resumo = summary(ps)
    efeitos = effects_by_model(study_dir, {"tabulares": list(tree_tags), "LSTM": list(lstm_tags)}, arms, n_boot)
    ps.to_csv(out / "erro_por_modelo_arm_trimestre.csv", index=False)
    resumo.to_csv(out / "resumo_por_modelo_arm.csv", index=False)
    efeitos.to_csv(out / "efeito_full_vs_base_por_modelo.csv", index=False)
    return {"por_trimestre": ps, "resumo": resumo, "efeitos": efeitos, "out": out}


def best_tree_by_season(ps: pd.DataFrame, arm: str = "base") -> pd.DataFrame:
    """O melhor modelo NÃO-LSTM de cada trimestre num arm, para comparar com a LSTM."""
    d = ps[(ps["arm"] == arm) & (ps["familia"] != FAMILIA_LSTM)]
    idx = d.groupby("season")["RMSE"].idxmin()
    return d.loc[idx].set_index("season")[["model", "RMSE"]].rename(columns={"RMSE": "RMSE_melhor_nao_lstm"})
