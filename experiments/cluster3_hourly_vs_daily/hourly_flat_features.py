"""Constrói uma tabela flat (uma linha por estação/dia) a partir do dataset
ERA5 horário do Cluster 3, para o braço LazyPredict do experimento.

Diferente de `hourly_windower.py` (que produz tensores `(N, lookback,
n_features)` pro LSTM), o LazyPredict precisa de features escalares por
amostra — aqui cada uma das 12 variáveis horárias vira 4 colunas
(mean/max/min/std) agregadas sobre a janela causal `[D-7dias, D)` (mesma
regra de "só passado" do windower do LSTM: não inclui o próprio dia D).

Não modifica nenhum módulo de `src/` — só importa e reaproveita.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.cluster_assigner import assign_station_clusters
from src.pipeline.data.cluster_preprocessor import SEASONS
from src.pipelines.common import TARGET_VAR, TEST_SLICE, TRAIN_SLICE, VAL_SLICE

from experiments.cluster3_hourly_vs_daily.hourly_windower import (
    HOURLY_FEATURES,
    LOOKBACK_HOURS,
    TARGET_CLUSTER,
)
from experiments.cluster3_hourly_vs_daily.engineered_features import (
    ENGINEERED_COLUMNS,
    add_engineered_columns,
)

_MONTH_TO_SEASON: dict[int, str] = {
    m: s for s, months in SEASONS.items() for m in months
}

_AGGS = ("mean", "max", "min", "std")


def build_hourly_cluster3_flat_table(
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    hourly_filename: str = "test_cluster_3_hourly.nc",
    lookback_hours: int = LOOKBACK_HOURS,
    train_slice: tuple[str, str] = TRAIN_SLICE,
    val_slice: tuple[str, str] = VAL_SLICE,
    test_slice: tuple[str, str] = TEST_SLICE,
    max_days_per_station: int | None = None,
    stations_limit: int | None = None,
) -> pd.DataFrame:
    """Retorna um DataFrame com 1 linha por (estação, dia): 12×4=48 colunas
    agregadas + `gust_P50` (mediana do alvo por estação, só no treino) +
    `daily_wind_gust_max` (alvo), `era5_proxy`, `season`, `split`,
    `estacao`/`latitude`/`longitude`/`time`.

    `max_days_per_station`/`stations_limit`: só usados pelo modo `--smoke`.
    """
    raw_path = Path(raw_dir)
    ds_hourly = xr.open_dataset(raw_path / hourly_filename)
    ds_inmet = xr.open_dataset(raw_path / "INMET_Stratified.nc")

    stations = sorted(str(s) for s in ds_hourly["estacao"].values)

    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    cluster3_stations = set(
        station_clusters.loc[station_clusters["cluster_id"] == TARGET_CLUSTER, "estacao"]
    )
    if set(stations) != cluster3_stations:
        raise ValueError(
            f"Estações do arquivo horário ({sorted(stations)}) não batem mais "
            f"com o Cluster {TARGET_CLUSTER} atual ({sorted(cluster3_stations)})."
        )
    if stations_limit is not None:
        stations = stations[:stations_limit]

    train_sl = slice(*train_slice)
    val_sl = slice(*val_slice)
    test_sl = slice(*test_slice)

    target_da = ds_inmet[TARGET_VAR]
    ws_h_daily_max = ds_hourly["ws_h"].resample(time="1D").max()
    ws_h_daily_mean = ds_hourly["ws_h"].resample(time="1D").mean()

    rows: list[dict] = []
    for est in stations:
        df_feat = ds_hourly[HOURLY_FEATURES].sel(estacao=est).to_dataframe()[HOURLY_FEATURES]
        hourly_values = df_feat.values.astype("float64")
        hour_index = df_feat.index
        hour_start = hour_index[0]

        lat = float(ds_inmet["latitude"].sel(estacao=est).values)
        lon = float(ds_inmet["longitude"].sel(estacao=est).values)

        tgt = target_da.sel(estacao=est).to_series().rename("target")
        era5 = ws_h_daily_max.sel(estacao=est).to_series().rename("era5_proxy")
        era5_mean = ws_h_daily_mean.sel(estacao=est).to_series().rename("era5_mean")
        df_daily = pd.concat([tgt, era5, era5_mean], axis=1).dropna(subset=["target", "era5_proxy"])
        df_daily = add_engineered_columns(df_daily, train_slice[0], train_slice[1])

        for split_name, sl in (("train", train_sl), ("val", val_sl), ("test", test_sl)):
            days = df_daily.loc[sl.start:sl.stop].index
            if max_days_per_station is not None:
                days = days[:max_days_per_station]

            for day in days:
                pos = int((day - hour_start) / pd.Timedelta(hours=1))
                if pos < lookback_hours:
                    continue
                window = hourly_values[pos - lookback_hours: pos]
                if window.shape[0] != lookback_hours or not np.isfinite(window).all():
                    continue
                eng_row = df_daily.loc[day, ENGINEERED_COLUMNS]
                if not np.isfinite(eng_row.to_numpy(dtype="float64")).all():
                    continue  # sem histórico suficiente pros lags/climatologia ainda

                row = {
                    "estacao": est, "latitude": lat, "longitude": lon, "time": day,
                    "split": split_name, "season": _MONTH_TO_SEASON[day.month],
                    TARGET_VAR: df_daily.loc[day, "target"],
                    "era5_proxy": df_daily.loc[day, "era5_proxy"],
                    **eng_row.to_dict(),
                }
                for j, var in enumerate(HOURLY_FEATURES):
                    col = window[:, j]
                    row[f"{var}_mean"] = col.mean()
                    row[f"{var}_max"] = col.max()
                    row[f"{var}_min"] = col.min()
                    row[f"{var}_std"] = col.std()
                rows.append(row)

    df = pd.DataFrame(rows)

    # gust_P50 (mediana do alvo por estação, só no treino) — mesma lógica de
    # src/pipelines/cluster_lazy.py:432-437, com fallback pra mediana do
    # cluster nas estações/linhas fora do treino.
    train_median_by_station = df.loc[df["split"] == "train"].groupby("estacao")[TARGET_VAR].median()
    cluster_median = float(df.loc[df["split"] == "train", TARGET_VAR].median())
    df["gust_P50"] = df["estacao"].map(train_median_by_station).fillna(cluster_median)

    return df


def flat_feature_columns() -> list[str]:
    """Nomes das 48 colunas agregadas + `ENGINEERED_COLUMNS` + `gust_P50` —
    o `train_features` que entra no LazyPredict (mesmo papel de
    `BASE_FEATURES` pro braço diário, incluindo as mesmas features de
    lag/sazonalidade/climatologia pra não confundir efeito de granularidade
    com efeito de feature engineering)."""
    return (
        [f"{var}_{agg}" for var in HOURLY_FEATURES for agg in _AGGS]
        + ENGINEERED_COLUMNS
        + ["gust_P50"]
    )
