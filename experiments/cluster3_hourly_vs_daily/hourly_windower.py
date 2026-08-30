"""Constrói um ClusterDataBatch (mesmo tipo usado pela pipeline de produção)
a partir do dataset ERA5 horário do Cluster 3, para o experimento isolado
diário-vs-horário.

Espelha a semântica de
`src.pipeline.data.cluster_preprocessor.ClusterPreprocessor._make_windows`,
só que com janela em horas (168h = 7 dias) em vez de em dias (lookback=7):
a janela de entrada de um dia D é sempre [D-lookback_hours, D) — não inclui
o próprio dia D, igual ao braço diário — e o alvo/âncora ERA5 são do próprio
dia D (mesma regra de "âncora do mesmo dia" que o braço diário já usa para
`wind_mag_max`).

Não modifica nenhum módulo de `src/` — só importa e reaproveita.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.data.cluster_assigner import assign_station_clusters
from src.data.static_features import StationStaticFeatures
from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS
from src.pipelines.common import TARGET_VAR, TEST_SLICE, TRAIN_SLICE, VAL_SLICE

from experiments.cluster3_hourly_vs_daily.engineered_features import (
    ENGINEERED_COLUMNS,
    add_engineered_columns,
)

HOURLY_FEATURES = [
    "ws_h", "wd_h", "sin_dir_h", "cos_dir_h", "msl_h", "t2m_h",
    "rh_h", "td_dep_h", "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h",
]

LOOKBACK_HOURS = 168  # 7 dias x 24h — mesmo horizonte temporal do lookback=7 diário
TARGET_CLUSTER = 3

_MONTH_TO_SEASON: dict[int, str] = {
    m: s for s, months in SEASONS.items() for m in months
}


def build_hourly_cluster3_batch(
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    hourly_filename: str = "test_cluster_3_hourly.nc",
    lookback_hours: int = LOOKBACK_HOURS,
    train_slice: tuple[str, str] = TRAIN_SLICE,
    val_slice: tuple[str, str] = VAL_SLICE,
    test_slice: tuple[str, str] = TEST_SLICE,
    max_days_per_station: int | None = None,
    stations_limit: int | None = None,
) -> ClusterDataBatch:
    """Monta o ClusterDataBatch do braço horário, restrito ao Cluster 3.

    Parâmetros
    ----------
    max_days_per_station, stations_limit
        Só usados pelo modo `--smoke` (validação local rápida) — em produção
        (Modal, "100% dos dados") ficam `None`, sem nenhum corte.
    """
    raw_path = Path(raw_dir)
    ds_hourly = xr.open_dataset(raw_path / hourly_filename)
    ds_inmet = xr.open_dataset(raw_path / "INMET_Stratified.nc")

    stations = sorted(str(s) for s in ds_hourly["estacao"].values)

    # Confirma que o Cluster 3 (spatial join atual) e o arquivo horário
    # cobrem exatamente as mesmas estações (já validado em
    # notebooks/cluster3_daily_vs_hourly_data_comparison.ipynb) — falha alto
    # e cedo se isso mudar no futuro em vez de silenciosamente ignorar.
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    cluster3_stations = set(
        station_clusters.loc[station_clusters["cluster_id"] == TARGET_CLUSTER, "estacao"]
    )
    if set(stations) != cluster3_stations:
        raise ValueError(
            f"Estações do arquivo horário ({sorted(stations)}) não batem mais "
            f"com o Cluster {TARGET_CLUSTER} atual ({sorted(cluster3_stations)}) "
            "— reveja o experimento antes de continuar."
        )
    if stations_limit is not None:
        stations = stations[:stations_limit]

    cluster_col = f"cluster_{TARGET_CLUSTER}"
    # ENGINEERED_COLUMNS (lags/sazonalidade/climatologia do braço diário) somadas
    # aqui pra não confundir efeito de granularidade com efeito de feature
    # engineering — sem isso o braço diário tinha ~35 features a mais que o
    # horário não tinha nada equivalente.
    feature_names = HOURLY_FEATURES + ENGINEERED_COLUMNS + [cluster_col]

    train_sl = slice(*train_slice)
    val_sl = slice(*val_slice)
    test_sl = slice(*test_slice)

    target_da = ds_inmet[TARGET_VAR]
    ws_h_daily_max = ds_hourly["ws_h"].resample(time="1D").max()
    ws_h_daily_mean = ds_hourly["ws_h"].resample(time="1D").mean()

    # ---- 1) fit imputer/scaler_x só nas HORAS de treino (fit-on-train-only,
    # mesma regra de ClusterPreprocessor.run) ----
    train_hours_parts = []
    for est in stations:
        df_est = ds_hourly[HOURLY_FEATURES].sel(estacao=est).to_dataframe()[HOURLY_FEATURES]
        train_hours_parts.append(df_est.loc[train_sl.start:train_sl.stop])
    train_hours_df = pd.concat(train_hours_parts, axis=0)

    imputer_x = SimpleImputer(strategy="mean", keep_empty_features=True)
    imputer_x.fit(train_hours_df[HOURLY_FEATURES])
    scaler_x = RobustScaler()
    scaler_x.fit(imputer_x.transform(train_hours_df[HOURLY_FEATURES]))

    # ---- 2) série diária por estação: alvo, âncora ERA5, ratio; scaler_y
    # ajustado só no ratio de treino ----
    daily_by_station: dict[str, pd.DataFrame] = {}
    train_ratios = []
    train_eng_parts = []
    for est in stations:
        tgt = target_da.sel(estacao=est).to_series().rename("target")
        era5 = ws_h_daily_max.sel(estacao=est).to_series().rename("era5_proxy")
        era5_mean = ws_h_daily_mean.sel(estacao=est).to_series().rename("era5_mean")
        df_daily = pd.concat([tgt, era5, era5_mean], axis=1).dropna(subset=["target", "era5_proxy"])
        df_daily["ratio"] = df_daily["target"] / df_daily["era5_proxy"].clip(lower=0.1)
        df_daily["season"] = df_daily.index.month.map(_MONTH_TO_SEASON)
        df_daily = add_engineered_columns(df_daily, train_slice[0], train_slice[1])
        daily_by_station[est] = df_daily
        train_ratios.append(df_daily.loc[train_sl.start:train_sl.stop, "ratio"])
        train_eng_parts.append(df_daily.loc[train_sl.start:train_sl.stop, ENGINEERED_COLUMNS])

    scaler_y = RobustScaler()
    scaler_y.fit(pd.concat(train_ratios).to_frame())

    imputer_eng = SimpleImputer(strategy="mean", keep_empty_features=True)
    imputer_eng.fit(pd.concat(train_eng_parts))
    scaler_eng = RobustScaler()
    scaler_eng.fit(imputer_eng.transform(pd.concat(train_eng_parts)))

    # ---- 3) static features (mesmo builder do braço diário) ----
    static_builder = StationStaticFeatures()
    df_static = static_builder.compute(
        ds_inmet, TARGET_VAR, train_sl, np.array(stations)
    )
    static_lookup = df_static.set_index("estacao")

    # ---- 4) janelas horárias por estação/dia/split ----
    buckets = {
        split: {s: {"x": [], "xs": [], "y": [], "era5": [], "est": [], "lat": [], "lon": [], "time": []}
                for s in SEASONS}
        for split in ("train", "val", "test")
    }

    for est in stations:
        df_feat = ds_hourly[HOURLY_FEATURES].sel(estacao=est).to_dataframe()[HOURLY_FEATURES]
        x_raw_scaled = scaler_x.transform(imputer_x.transform(df_feat.values)).astype("float32")
        hour_index = df_feat.index
        hour_start = hour_index[0]

        df_daily = daily_by_station[est]

        # Alinha as ENGINEERED_COLUMNS (por dia) a cada hora do dia correspondente
        # — cada uma das 24h de um dia D carrega o mesmo vetor "do dia" (lags,
        # sazonalidade, climatologia), igual ao braço diário onde essas features
        # são por-linha-por-dia, não por-hora.
        day_labels = hour_index.floor("D")
        eng_aligned = df_daily.reindex(day_labels)[ENGINEERED_COLUMNS].to_numpy(dtype="float64")
        eng_scaled = scaler_eng.transform(imputer_eng.transform(eng_aligned)).astype("float32")

        x_scaled = np.hstack([
            x_raw_scaled, eng_scaled,
            np.ones((len(x_raw_scaled), 1), dtype="float32"),  # coluna cluster_3
        ])

        lat = float(ds_inmet["latitude"].sel(estacao=est).values)
        lon = float(ds_inmet["longitude"].sel(estacao=est).values)
        static_vec = static_lookup.loc[est].values.astype("float32")

        for split_name, sl in (("train", train_sl), ("val", val_sl), ("test", test_sl)):
            days = df_daily.loc[sl.start:sl.stop].index
            if max_days_per_station is not None:
                days = days[:max_days_per_station]

            for day in days:
                pos = int((day - hour_start) / pd.Timedelta(hours=1))
                if pos < lookback_hours:
                    continue  # sem histórico suficiente (não ocorre no período oficial)
                window = x_scaled[pos - lookback_hours: pos]
                if window.shape[0] != lookback_hours or not np.isfinite(window).all():
                    continue

                row = df_daily.loc[day]
                y_scaled = scaler_y.transform([[row["ratio"]]])[0]
                if not np.isfinite(y_scaled).all():
                    continue

                season = row["season"]
                b = buckets[split_name][season]
                b["x"].append(window)
                b["xs"].append(static_vec)
                b["y"].append(y_scaled)
                b["era5"].append(row["era5_proxy"])
                b["est"].append(est)
                b["lat"].append(lat)
                b["lon"].append(lon)
                b["time"].append(day)

    batch = ClusterDataBatch(
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        imputer_x=imputer_x,
        feature_names=feature_names,
        static_feature_names=StationStaticFeatures.FEATURE_NAMES,
        cluster_ids=[TARGET_CLUSTER],
        station_clusters_df=station_clusters,
    )
    n_static = len(StationStaticFeatures.FEATURE_NAMES)

    def _stack(lst: list, shape_tail: tuple[int, ...]) -> np.ndarray:
        # Uma lista vazia via np.array(lst) dá shape (0,), não (0, *shape_tail)
        # — quebra o masking `arr[:, -1, col_idx]` de ClusterTrainer/
        # ClusterMetricsEvaluator quando um trimestre fica sem amostras
        # (comum em recortes pequenos, como o --smoke).
        if len(lst) == 0:
            return np.zeros((0, *shape_tail), dtype="float32")
        return np.array(lst)

    for split_name in ("train", "val", "test"):
        setattr(batch, f"x_{split_name}", {
            s: _stack(buckets[split_name][s]["x"], (lookback_hours, len(feature_names)))
            for s in SEASONS
        })
        setattr(batch, f"x_static_{split_name}", {
            s: _stack(buckets[split_name][s]["xs"], (n_static,)) for s in SEASONS
        })
        setattr(batch, f"y_{split_name}", {
            s: _stack(buckets[split_name][s]["y"], (1,)) for s in SEASONS
        })
        setattr(batch, f"era5_{split_name}", {s: np.array(buckets[split_name][s]["era5"]) for s in SEASONS})
        setattr(batch, f"meta_{split_name}", {
            s: {
                "estacao": np.array(buckets[split_name][s]["est"]),
                "latitude": np.array(buckets[split_name][s]["lat"]),
                "longitude": np.array(buckets[split_name][s]["lon"]),
                "time": np.array(buckets[split_name][s]["time"]),
            }
            for s in SEASONS
        })

    return batch
