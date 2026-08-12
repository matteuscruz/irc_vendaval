"""Climatologia populacional (cluster-mean) para uso em folds de station-
holdout.

`src/data/climatology.py::get_climatology` é calculada POR ESTAÇÃO (média
histórica de `wind_mag_max` por dia-do-ano, só no período de treino) e
mesclada no DataFrame flat como a feature `era5_clim_wind` — ver
`src/pipelines/common.py::build_flat_dataframe`. Isso é seguro sob o split
temporal atual (todas as estações aparecem em treino e teste), mas sob um
station-holdout espacial, a climatologia da estação held-out ainda carregaria
informação derivada da própria série histórica dessa estação.

Este módulo recalcula `era5_clim_wind` como a média por dia-do-ano sobre as
DEMAIS estações do fold de treino (cluster-mean), e a aplica às linhas da(s)
estação(ões) held-out — análogo à convenção do projeto de interpolação
clássica, onde variogramas/parâmetros de dependência espacial são ajustados
populacionalmente (nunca com a própria estação-alvo).
"""
from __future__ import annotations

import pandas as pd

from src.pipelines.common import ERA5_GUST_PROXY

CLIM_COL = "era5_clim_wind"


def compute_population_climatology(
    df_train_fold: pd.DataFrame,
    value_col: str = ERA5_GUST_PROXY,
) -> pd.Series:
    """Média por dia-do-ano de `value_col`, agregada sobre TODAS as estações
    presentes em `df_train_fold` (cluster-mean). `df_train_fold` já deve estar
    restrito ao período de treino e às estações do fold de treino (held-out
    excluída). Retorna Series indexada por `dayofyear`."""
    if "dayofyear" not in df_train_fold.columns:
        raise ValueError("df_train_fold precisa da coluna 'dayofyear'")
    return df_train_fold.groupby("dayofyear")[value_col].mean()


def apply_population_climatology(
    df: pd.DataFrame,
    population_clim: pd.Series,
    clim_col: str = CLIM_COL,
) -> pd.DataFrame:
    """Substitui `clim_col` (climatologia por-estação, já mesclada no
    DataFrame) pela climatologia populacional, mapeada por `dayofyear`. Linhas
    cujo `dayofyear` não está coberto pela climatologia populacional (raro,
    ano bissexto/limites do período de treino) são descartadas."""
    df = df.copy()
    df[clim_col] = df["dayofyear"].map(population_clim)
    return df.dropna(subset=[clim_col])
