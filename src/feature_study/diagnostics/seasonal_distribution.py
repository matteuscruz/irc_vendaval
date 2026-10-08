"""Distribuição da rajada diária por trimestre e por método, só com o que o estudo já gravou.

A figura compara a distribuição da rajada máxima diária (média entre as estações de cada dia) do INMET, do ERA5 e de
vários previsores. Cada previsor vem do resíduo gravado (`resid = previsão − observado`), então NADA é treinado aqui.

Os previsores de um conjunto de colunas são lidos de um arm do estudo: `base` (o grupo I), `add_grp__grupoN` (o grupo I
somado ao grupo N) e `sel__val12` (as variáveis selecionadas). "Grupo isolado" (só as colunas de um grupo) NÃO existe
gravado — só foi treinado dentro de um notebook —, então o que se mostra para os grupos II, III e IV é o grupo somado ao I.

Média entre estações: cada dia vale a média das estações com previsão naquele dia (a cobertura varia por dia).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.data.groups_source import REFERENCE_COLUMN
from src.pipelines.common import TARGET_VAR

SEASONS = ("DJF", "MAM", "JJA", "SON")
INMET, ERA5 = "INMET", "ERA5"


def _resid(units_dir, tag, season, arm, model) -> pd.DataFrame:
    p = Path(units_dir) / tag / f"resid__{season}__{arm}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe — o arm `{arm}` não foi ajustado neste conjunto")
    r = pd.read_parquet(p)
    if model not in r.columns:
        raise ValueError(f"{p.name} não tem o modelo {model!r}")
    return r[["row_id", model]]


def load_series(ml_dir, lstm_dir, champion: dict[str, str], ml_arms: dict[str, str], lstm_arms: dict[str, str], *,
                ml_tag: str = "full", lstm_tag: str = "full_lstm", lstm_model: str = "LSTM") -> pd.DataFrame:
    """Uma linha por (estação, dia) do teste: observado, ERA5 e a previsão de cada rótulo em `ml_arms` e `lstm_arms`
    (`{rótulo: arm}`). Só ficam as linhas que TODOS os previsores têm (a LSTM descarta as sem janela completa)."""
    ml_dir, lstm_dir = Path(ml_dir), Path(lstm_dir)
    t = pd.read_parquet(ml_dir / "data/test.parquet",
                        columns=["row_id", "season", "estacao", "time", TARGET_VAR, REFERENCE_COLUMN])
    partes = []
    for s in SEASONS:
        d = t[t["season"] == s].rename(columns={TARGET_VAR: INMET, REFERENCE_COLUMN: ERA5})
        for rot, arm in ml_arms.items():
            d = d.merge(_resid(ml_dir / "units", ml_tag, s, arm, champion[s]).rename(columns={champion[s]: rot}),
                        on="row_id", how="inner")
        for rot, arm in lstm_arms.items():
            d = d.merge(_resid(lstm_dir / "units", lstm_tag, s, arm, lstm_model).rename(columns={lstm_model: rot}),
                        on="row_id", how="inner")
        for rot in (*ml_arms, *lstm_arms):
            d[rot] = d[rot] + d[INMET]                       # resid = previsão − observado
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def daily_station_mean(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Rajada diária média entre as estações presentes em cada dia, por trimestre."""
    return df.groupby(["season", "time"], as_index=False)[list(columns)].mean()


def summary(daily: pd.DataFrame, columns) -> pd.DataFrame:
    """Média, mediana, P90 e máximo da série diária de cada método, por trimestre e no total."""
    linhas = []
    for escopo, d in (("Todos", daily), *((s, daily[daily.season == s]) for s in SEASONS)):
        for c in columns:
            x = d[c].dropna().to_numpy(float)
            linhas.append({"escopo": escopo, "metodo": c, "media": x.mean(), "mediana": float(np.median(x)),
                           "p90": float(np.quantile(x, 0.9)), "maximo": x.max(), "n_dias": len(x)})
    return pd.DataFrame(linhas)
