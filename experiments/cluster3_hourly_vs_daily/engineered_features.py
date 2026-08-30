"""Features de engenharia equivalentes às que o braço diário já usa
(`src.pipelines.common.build_flat_dataframe` + `src/data/netcdf_loader.py`),
pra somar ao braço horário e isolar o efeito real da granularidade em vez de
confundir com a diferença de feature engineering entre os dois braços.

Sem isso, o braço diário carrega ~35 features (lags do alvo, sazonalidade
cíclica, climatologia, tendência de pressão, contexto regional `_basin`) que
o arquivo horário cru simplesmente não tinha — o gap de desempenho medido
antes não isolava só a resolução temporal.

Calculadas a partir da série DIÁRIA por estação (alvo INMET + proxy ERA5
`ws_h` diário) — não dependem da resolução horária em si, só reaproveitam a
mesma lógica causal (lag/rolling/climatologia/sazonalidade) do braço diário,
usando cada dia D só com informação de D e de dias anteriores.

Não modifica nenhum módulo de `src/` — só reimplementa a mesma lógica
(pequena o bastante pra não valer a pena importar/fatiar o código de
produção, que opera sobre `xr.Dataset`, não sobre a série por estação que
temos aqui).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_COLUMNS = [
    "day_sin", "day_cos", "month_sin", "month_cos", "era5_clim_wind",
    "lag1_era5_proxy", "lag2_era5_proxy", "lag3_era5_proxy", "lag7_era5_proxy",
    "rolling7d_era5_proxy", "rolling3d_era5_proxy", "era5_proxy_anom",
    "gust_factor",
    "lag1_gust_obs", "lag2_gust_obs", "lag3_gust_obs", "rolling7d_gust_obs",
]


def add_engineered_columns(
    df_daily: pd.DataFrame, train_start: str, train_end: str
) -> pd.DataFrame:
    """`df_daily`: index = data (diária, uma linha por dia), colunas
    ["target", "era5_proxy"] (âncora ERA5 do dia, ex.: `ws_h` diário-máx) e
    opcionalmente "era5_mean" (`ws_h` diário-média, usada no `gust_factor`).

    Retorna cópia com as colunas de `ENGINEERED_COLUMNS` — cálculo causal
    (lags/rolling usam `shift`, nunca olham o próprio dia D à frente; a
    climatologia é ajustada só no período de treino, mesma regra de
    `src.data.climatology.get_climatology`)."""
    df = df_daily.sort_index().copy()

    doy = df.index.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["dayofyear"] = doy
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    clim = df.loc[train_mask].groupby("dayofyear")["era5_proxy"].mean()
    df["era5_clim_wind"] = df["dayofyear"].map(clim)

    for lag in (1, 2, 3, 7):
        df[f"lag{lag}_era5_proxy"] = df["era5_proxy"].shift(lag)
    df["rolling7d_era5_proxy"] = df["era5_proxy"].shift(1).rolling(7, min_periods=3).mean()
    df["rolling3d_era5_proxy"] = df["era5_proxy"].shift(1).rolling(3, min_periods=2).mean()
    df["era5_proxy_anom"] = df["era5_proxy"] - df["rolling7d_era5_proxy"]

    if "era5_mean" in df.columns:
        df["gust_factor"] = df["era5_proxy"] / df["era5_mean"].clip(lower=0.1)
    else:
        df["gust_factor"] = np.nan

    for lag in (1, 2, 3):
        df[f"lag{lag}_gust_obs"] = df["target"].shift(lag)
    df["rolling7d_gust_obs"] = df["target"].shift(1).rolling(7, min_periods=3).mean()

    return df.drop(columns=["dayofyear"])
