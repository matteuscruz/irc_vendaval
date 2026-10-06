"""Fonte HORÁRIA da LSTM: janela = as 24 horas do dia D → rajada máxima de D.

Promovido de `experiments/cluster3_hourly_vs_daily/hourly_windower.py` e
generalizado: estações = (arquivo horário) ∩ (INMET) ∩ (clusters); clusters
sem dado horário só geram aviso. Hoje o único arquivo horário é
`test_cluster_3_hourly.nc` (21 estações do cluster 3).

Features por hora = variáveis horárias do arquivo + contexto diário do dia D
(sazonalidade) repetido nas 24 horas + one-hot do cluster. O INMET entra só
como alvo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import RobustScaler

from src.data.cluster_assigner import assign_station_clusters
from src.pipeline.data.cluster_preprocessor import (
    SPLITS, ClusterDataBatch, WindowBuckets, cluster_onehot, station_coords,
)
from src.pipeline.data.splits import split_from_config
from src.pipeline.data.windowing import HOURS_PER_DAY, build_hourly_day_windows
from src.pipelines.common import TARGET_VAR

INMET_FILENAME = "INMET_Stratified.nc"
DEFAULT_HOURLY_FILE = "test_cluster_3_hourly.nc"
HOURLY_FEATURES = [
    "ws_h", "wd_h", "sin_dir_h", "cos_dir_h", "msl_h", "t2m_h",
    "rh_h", "td_dep_h", "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h",
]
SEASONAL_CONTEXT = ["day_sin", "day_cos", "month_sin", "month_cos"]
DEFAULT_DAILY_CONTEXT = list(SEASONAL_CONTEXT)


def daily_context_frame(days) -> pd.DataFrame:
    """Contexto diário indexado por `days`: mesma definição de sazonalidade
    de `build_flat_dataframe`."""
    cal = pd.DatetimeIndex(days)
    doy = cal.dayofyear
    return pd.DataFrame(
        {
            "day_sin": np.sin(2 * np.pi * doy / 365.25),
            "day_cos": np.cos(2 * np.pi * doy / 365.25),
            "month_sin": np.sin(2 * np.pi * cal.month / 12),
            "month_cos": np.cos(2 * np.pi * cal.month / 12),
        },
        index=cal,
    )


def build_hourly_batch(
    raw_dir: str,
    shp_dir: str,
    *,
    hourly_cfg: dict | None = None,
    split_cfg: dict | None = None,
    seed: int = 42,
    target_var: str = TARGET_VAR,
) -> ClusterDataBatch:
    hourly_cfg = dict(hourly_cfg or {})
    filename = hourly_cfg.get("file") or DEFAULT_HOURLY_FILE
    features = list(hourly_cfg.get("features") or HOURLY_FEATURES)
    ctx_cols = list(hourly_cfg.get("daily_context_features") or DEFAULT_DAILY_CONTEXT)
    offset = int(hourly_cfg.get("day_offset_hours", 0))
    unknown_ctx = [c for c in ctx_cols if c not in DEFAULT_DAILY_CONTEXT]
    if unknown_ctx:
        raise ValueError(
            f"daily_context_features desconhecidas: {unknown_ctx} (válidas: {DEFAULT_DAILY_CONTEXT})"
        )

    raw = Path(raw_dir)
    ds_hourly = xr.open_dataset(raw / filename)
    ds_inmet = xr.open_dataset(raw / INMET_FILENAME)
    missing = [f for f in features if f not in ds_hourly.data_vars]
    if missing:
        raise ValueError(f"{filename} não tem as variáveis horárias {missing}")

    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    cluster_of = station_clusters.set_index("estacao")["cluster_id"]
    coords = station_coords(ds_inmet)
    hourly_st = {str(s) for s in ds_hourly["estacao"].values}
    inmet_st = {str(s) for s in ds_inmet["estacao"].values}
    stations = sorted(hourly_st & inmet_st & set(map(str, cluster_of.index)))
    if not stations:
        raise ValueError(f"nenhuma estação em comum entre {filename}, INMET e os clusters")
    cluster_ids = sorted({int(cluster_of[s]) for s in stations})
    no_hourly = sorted(set(station_clusters["cluster_id"]) - set(cluster_ids))
    if no_hourly:
        print(f"[hourly] AVISO: clusters sem dado horário (não treinados): {no_hourly}")
    print(f"[hourly] {len(stations)} estações, clusters {cluster_ids}, offset={offset}h")

    hour_times = pd.DatetimeIndex(ds_hourly["time"].values)
    day_of_hour_all = (hour_times - pd.Timedelta(hours=offset)).floor("D")
    split = split_from_config(day_of_hour_all.unique(), split_cfg, seed)

    base_names = features + ctx_cols
    per_station = []
    for est in stations:
        df_h = ds_hourly[features].sel(estacao=est).to_dataframe()[features].sort_index()
        day_of_hour = (df_h.index - pd.Timedelta(hours=offset)).floor("D")
        cal = pd.date_range(day_of_hour.min(), day_of_hour.max(), freq="D")
        target = ds_inmet[target_var].sel(estacao=est).to_series()
        target = target[~target.index.duplicated()].reindex(cal)
        ctx = daily_context_frame(cal)[ctx_cols]
        mat = np.hstack([df_h.to_numpy(dtype=float), ctx.reindex(day_of_hour).to_numpy(dtype=float)])
        per_station.append((est, df_h.index, mat, split.label(day_of_hour), target))

    fit_x = np.concatenate([m[lab == "train"] for _, _, m, lab, _ in per_station], axis=0)
    fit_x = fit_x[np.isfinite(fit_x).all(axis=1)]  # sem imputação
    fit_y = pd.concat(
        [t[(split.label(t.index) == "train") & t.notna().to_numpy()] for *_, t in per_station]
    )
    if len(fit_x) == 0 or fit_y.empty:
        raise ValueError("nenhuma hora/dia de treino — confira split/date_range e o arquivo horário")
    scaler_x = RobustScaler().fit(fit_x)
    scaler_y = RobustScaler().fit(fit_y.to_frame())
    del fit_x

    feature_names = base_names + [f"cluster_{c}" for c in cluster_ids]
    buckets = WindowBuckets()
    for est, hours, mat, _, target in per_station:
        cid = int(cluster_of[est])
        scaled = scaler_x.transform(mat).astype("float32")
        onehot = np.broadcast_to(cluster_onehot(cluster_ids, cid), (len(scaled), len(cluster_ids)))
        windows, days = build_hourly_day_windows(
            np.hstack([scaled, onehot]), hours, day_offset_hours=offset,
        )
        y = target.reindex(days).to_numpy(dtype=float)
        lab = split.label(days)
        keep = np.isfinite(y) & np.isfinite(windows).all(axis=(1, 2))
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel() if len(y) else y
        lat, lon = coords.loc[est, "latitude"], coords.loc[est, "longitude"]
        for sp in SPLITS:
            idx = np.flatnonzero(keep & (lab == sp))
            if len(idx):
                buckets.add(
                    sp, windows[idx], y_scaled[idx], days[idx],
                    estacao=est, latitude=lat, longitude=lon, cluster_id=cid,
                )

    batch = ClusterDataBatch(
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_names=feature_names,
        cluster_ids=cluster_ids,
        station_clusters_df=station_clusters[station_clusters["estacao"].isin(stations)],
        resolution="hourly",
        window_length=HOURS_PER_DAY,
        split=split,
        interp_method="none",
        climatology={"method": "none"},
        hourly={
            "file": filename,
            "features": features,
            "daily_context_features": ctx_cols,
            "day_offset_hours": offset,
        },
    )
    buckets.fill(batch, HOURS_PER_DAY, len(feature_names))
    for sp in SPLITS:
        n = sum(len(v) for v in getattr(batch, f"x_{sp}").values())
        print(f"[hourly]   {sp}: {n} janelas")
    return batch
