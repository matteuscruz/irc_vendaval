"""Calibração e detecção de extremos de cinco previsores, por trimestre e no total.

Os previsores, todos avaliados nas MESMAS linhas do teste:

  ERA5                  a rajada bruta do ERA5 (`era5_gust_max`);
  ML, base              o melhor modelo tabular de cada trimestre (o eleito pelo estudo) com o grupo 1;
  ML, selecionadas      o mesmo modelo com as variáveis selecionadas (`sel__val12`);
  LSTM, base            a LSTM com o grupo 1;
  LSTM, selecionadas    a LSTM com as variáveis selecionadas.

A previsão de cada um vem do resíduo gravado pelo estudo (`resid = previsão − observado`), então nada é retreinado
aqui. As linhas são a interseção das do ML e das da LSTM (a LSTM descarta as ~15 sem janela completa).

As curvas são as da análise de robustez: calibração CONDICIONADA NA PREVISÃO (`reliability`) e POD/FAR de excedência
(`contingency`) em P90/P95/P99 do observado do próprio escopo. A calibração é condicionada na previsão, não no
observado: condicionar no observado alto dá viés negativo até para um previsor perfeito (regressão à média).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.core.worker_lstm import LSTM_NAME
from src.feature_study.data.groups_source import REFERENCE_COLUMN
from src.feature_study.diagnostics.robustness import contingency, reliability
from src.pipelines.common import TARGET_VAR

SEASONS = ("DJF", "MAM", "JJA", "SON")
ERA5 = "ERA5"
PREVISORES = (ERA5, "ML, base", "ML, selecionadas", "LSTM, base", "LSTM, selecionadas")
LIMIARES = (("P90", 0.90), ("P95", 0.95), ("P99", 0.99))
TODOS = "Todos os trimestres"


def champions(summary_dir) -> dict[str, str]:
    """O modelo de ML de cada trimestre: o rank 1 do `top_models.json` congelado pela triagem."""
    by = json.loads((Path(summary_dir) / "top_models.json").read_text())["by_season"]
    return {s: m[0] for s, m in by.items()}


def _resid(units_dir, tag, season, arm, model) -> pd.DataFrame:
    p = Path(units_dir) / tag / f"resid__{season}__{arm}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe — o arm `{arm}` ainda não foi ajustado para este conjunto")
    r = pd.read_parquet(p)
    if model not in r.columns:
        raise ValueError(f"{p.name} não tem o modelo {model!r} (tem {[c for c in r.columns if c != 'row_id']})")
    return r[["row_id", model]]


def load_predictors(ml_dir, lstm_dir, champion: dict[str, str], *, ml_tag: str = "full", lstm_tag: str = "full_lstm",
                    base_arm: str = "base", sel_arm: str = "sel__val12") -> pd.DataFrame:
    """Uma linha por (linha do teste) com o observado e a previsão dos cinco previsores. Levanta se as duas
    populações (ML e LSTM) tiverem alvos diferentes na mesma linha: seria misturar estudos."""
    ml_dir, lstm_dir = Path(ml_dir), Path(lstm_dir)
    t_ml = pd.read_parquet(ml_dir / "data/test.parquet", columns=["row_id", "season", TARGET_VAR, REFERENCE_COLUMN])
    t_ls = pd.read_parquet(lstm_dir / "data/test.parquet", columns=["row_id", TARGET_VAR])
    conf = t_ml.merge(t_ls, on="row_id", suffixes=("", "_l"))
    if not np.allclose(conf[TARGET_VAR], conf[f"{TARGET_VAR}_l"]):
        raise ValueError("o alvo difere entre os estudos do ML e da LSTM na mesma linha: populações diferentes")

    partes = []
    for s in SEASONS:
        base = t_ml[t_ml["season"] == s][["row_id", "season", TARGET_VAR, REFERENCE_COLUMN]]
        d = base.rename(columns={TARGET_VAR: "y", REFERENCE_COLUMN: ERA5})
        for nome, units, tag, arm, modelo in (
            ("ML, base", ml_dir / "units", ml_tag, base_arm, champion[s]),
            ("ML, selecionadas", ml_dir / "units", ml_tag, sel_arm, champion[s]),
            ("LSTM, base", lstm_dir / "units", lstm_tag, base_arm, LSTM_NAME),
            ("LSTM, selecionadas", lstm_dir / "units", lstm_tag, sel_arm, LSTM_NAME),
        ):
            r = _resid(units, tag, s, arm, modelo).rename(columns={modelo: nome})
            d = d.merge(r, on="row_id", how="inner")           # interseção das linhas de todos os previsores
        for nome in PREVISORES[1:]:
            d[nome] = d[nome] + d["y"]                          # resid = previsão − observado
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def scopes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """O conjunto todo e cada trimestre."""
    return {TODOS: df, **{s: df[df["season"] == s] for s in SEASONS if (df["season"] == s).any()}}


def curves(df: pd.DataFrame, previsores=PREVISORES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(calibração, excedência)` por escopo × previsor. Os limiares são os quantis do observado DO ESCOPO."""
    cal, exc = [], []
    for escopo, d in scopes(df).items():
        y = d["y"].to_numpy(float)
        limiares = {nome: float(np.quantile(y, q)) for nome, q in LIMIARES}
        for p in previsores:
            pred = d[p].to_numpy(float)
            r = reliability(y, pred)
            cal.append(r.assign(escopo=escopo, previsor=p))
            for nome, lim in limiares.items():
                exc.append({"escopo": escopo, "previsor": p, "limiar": nome, "valor_m/s": lim, "n": len(y),
                            **contingency(y, pred, lim)})
    return pd.concat(cal, ignore_index=True), pd.DataFrame(exc)
