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

import numpy as np
import pandas as pd

from src.data.climatology import _harmonic_design
from src.pipelines.common import ERA5_GUST_PROXY

CLIM_COL = "era5_clim_wind"


def compute_population_climatology(
    df_train_fold: pd.DataFrame,
    value_col: str = ERA5_GUST_PROXY,
    n_harmonics: int = 3,
) -> pd.Series:
    """Climatologia populacional (cluster-mean) de `value_col`, indexada por
    `dayofyear` (1..366). `df_train_fold` já deve estar restrito ao período de
    treino e às estações do fold de treino (held-out excluída).

    O ajuste é por **série harmônica**, não por média direta do dia-do-ano,
    pelo mesmo motivo de `src/data/climatology.py::get_harmonic_climatology`:
    sob o split por blocos de mês, os meses de teste (Jan/Abr/Jul/Out) não têm
    NENHUMA amostra de treino, então a média por dia-do-ano ficaria indefinida
    em ~120 dos 126 dias-do-ano avaliados e `apply_population_climatology`
    descartaria quase todo o conjunto de avaliação do fold. A série harmônica é
    contínua no ano e fica definida nos 366 dias.
    """
    if "dayofyear" not in df_train_fold.columns:
        raise ValueError("df_train_fold precisa da coluna 'dayofyear'")

    obs = df_train_fold.groupby("dayofyear")[value_col].mean().dropna()
    doy_grid = np.arange(1, 367)
    if obs.empty:
        return pd.Series(np.nan, index=pd.Index(doy_grid, name="dayofyear"))

    # Ajusta quantos harmônicos a amostra suportar (cada um custa 2
    # coeficientes, mais o intercepto). Com poucos dias distintos o ajuste
    # degenera para a média constante do fold — que continua DEFINIDA nos 366
    # dias, que é a propriedade da qual o fold depende. Cair na média por
    # dia-do-ano aqui reintroduziria os buracos que motivaram a harmônica.
    n_harmonics = min(n_harmonics, max(0, (len(obs) - 1) // 2))
    if n_harmonics == 0:
        return pd.Series(
            float(obs.mean()), index=pd.Index(doy_grid, name="dayofyear"), name=value_col,
        )

    design = _harmonic_design(obs.index.to_numpy(), n_harmonics)
    coef, *_ = np.linalg.lstsq(design, obs.to_numpy(dtype=float), rcond=None)
    return pd.Series(
        _harmonic_design(doy_grid, n_harmonics) @ coef,
        index=pd.Index(doy_grid, name="dayofyear"),
        name=value_col,
    )


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
