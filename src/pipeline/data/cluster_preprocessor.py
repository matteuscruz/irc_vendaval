"""Construção dos dados da LSTM por cluster — fonte DIÁRIA.

`ClusterDataBatch` é o contrato comum entre as fontes (diária aqui, horária em
`hourly_builder.py`), o trainer e as métricas: janelas por
split × estação do ano, alvo = rajada máxima diária em m/s escalonada por
`scaler_y` (sem razão × ERA5), cluster como one-hot no último passo da janela.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.climatology import get_harmonic_climatology
from src.data.cluster_assigner import assign_station_clusters
from src.data.netcdf_loader import NetCDFLoader
from src.pipeline.data.splits import MonthBlockSplit, split_from_config
from src.pipeline.data.windowing import build_daily_windows

SEASONS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
_MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}
SPLITS = ("train", "val", "test")
META_KEYS = ("estacao", "latitude", "longitude", "time", "cluster_id")

from src.pipelines.common import (  # noqa: E402
    ERA5_GUST_PROXY, TARGET_VAR, build_flat_dataframe,
    assert_no_missing, resolve_feature_groups, select_complete_rows,
)


@dataclass
class ClusterDataBatch:
    # {season: (N, T, F)} — T = dias (diário) ou 24 h (horário); o último
    # passo é sempre o dia-alvo.
    x_train: dict[str, np.ndarray] = field(default_factory=dict)
    x_val: dict[str, np.ndarray] = field(default_factory=dict)
    x_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: (N, 1)} — rajada máxima diária (m/s) escalonada por scaler_y
    y_train: dict[str, np.ndarray] = field(default_factory=dict)
    y_val: dict[str, np.ndarray] = field(default_factory=dict)
    y_test: dict[str, np.ndarray] = field(default_factory=dict)
    # {season: {"estacao"/"latitude"/"longitude"/"time"/"cluster_id": (N,)}} —
    # identidade de cada janela, alinhada posicionalmente com x_*/y_*.
    meta_train: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta_val: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta_test: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scaler_x: RobustScaler = field(default_factory=RobustScaler)
    scaler_y: RobustScaler = field(default_factory=RobustScaler)
    feature_names: list[str] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)
    station_clusters_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Como o batch foi construído — vai para run_meta.json e dl_metadata.joblib
    resolution: str = "daily"
    window_length: int = 7
    split: MonthBlockSplit = field(default_factory=MonthBlockSplit)
    interp_method: str = "nearest"
    climatology: dict = field(default_factory=lambda: {"method": "harmonic", "n_harmonics": 3})
    hourly: dict | None = None


class WindowBuckets:
    """Acumula janelas por split × estação do ano e preenche o batch no fim
    (uma concatenação por bucket em vez de append janela a janela)."""

    def __init__(self) -> None:
        self._d = {
            sp: {s: {k: [] for k in ("x", "y", *META_KEYS)} for s in SEASONS}
            for sp in SPLITS
        }

    def add(self, split, x, y, times, *, estacao, latitude, longitude, cluster_id) -> None:
        times = pd.DatetimeIndex(times)
        seasons = np.asarray(pd.Series(times.month).map(_MONTH_TO_SEASON))
        for s in SEASONS:
            m = seasons == s
            n = int(m.sum())
            if not n:
                continue
            b = self._d[split][s]
            b["x"].append(np.asarray(x[m], dtype="float32"))
            b["y"].append(np.asarray(y[m], dtype="float32").reshape(-1, 1))
            b["time"].append(np.asarray(times[m].values))
            b["estacao"].append(np.full(n, estacao, dtype=object))
            b["latitude"].append(np.full(n, latitude, dtype=float))
            b["longitude"].append(np.full(n, longitude, dtype=float))
            b["cluster_id"].append(np.full(n, cluster_id))

    def fill(self, batch: ClusterDataBatch, n_steps: int, n_features: int) -> ClusterDataBatch:
        def cat(parts, empty):
            return np.concatenate(parts, axis=0) if parts else empty

        for sp in SPLITS:
            xs, ys, metas = {}, {}, {}
            for s in SEASONS:
                b = self._d[sp][s]
                xs[s] = cat(b["x"], np.zeros((0, n_steps, n_features), dtype="float32"))
                ys[s] = cat(b["y"], np.zeros((0, 1), dtype="float32"))
                metas[s] = {k: cat(b[k], np.array([])) for k in META_KEYS}
            setattr(batch, f"x_{sp}", xs)
            setattr(batch, f"y_{sp}", ys)
            setattr(batch, f"meta_{sp}", metas)
        return batch


def station_coords(ds_inmet) -> pd.DataFrame:
    """lat/lon por estação (índice = estacao) — mesmo recorte de
    assign_station_clusters, funciona com lat/lon por estação ou por tempo."""
    return (
        ds_inmet[["latitude", "longitude"]].to_dataframe()
        .groupby("estacao")[["latitude", "longitude"]].first()
    )


def cluster_onehot(cluster_ids: list, cluster_id) -> np.ndarray:
    return (np.asarray(cluster_ids) == cluster_id).astype("float32")


class ClusterPreprocessor:
    """Fonte diária da LSTM:

      1. `load_extended(interp_method)` — INMET + ERA5-Basin (+ new_features)
      2. clusters por spatial join
      3. split por blocos de mês (`split_from_config`)
      4. climatologia ERA5 harmônica ajustada só nos dias de treino
      5. `build_flat_dataframe(require_target=False)` → calendário diário
         contínuo por estação (dias ausentes viram rótulo "out")
      6. linhas completas (`select_complete_rows`: clusters 100% cobertos
         pelas features novas, sem NaN) — dia descartado vira "out" e purga
         as janelas que o tocam
      7. scaler_x/scaler_y ajustados em linhas de treino com alvo
      8. janelas [i-L+1 .. i] com purga (`build_daily_windows(labels=...)`)
    """

    def __init__(
        self,
        raw_dir: str,
        shp_dir: str,
        *,
        target_var: str = TARGET_VAR,
        lookback: int = 7,
        split_cfg: dict | None = None,
        seed: int = 42,
        feature_groups: str | None = None,
        interp_method: str = "nearest",
        n_harmonics: int = 3,
    ) -> None:
        self.raw_dir = raw_dir
        self.shp_dir = shp_dir
        self.target_var = target_var
        self.lookback = int(lookback)
        self.split_cfg = split_cfg
        self.seed = seed
        self.feature_groups = feature_groups
        self.interp_method = interp_method
        self.n_harmonics = n_harmonics

    def run(self) -> ClusterDataBatch:
        print(f"[preprocessor] Carregando NetCDF (interp={self.interp_method})...")
        ds_inmet, ds_era5 = NetCDFLoader(self.raw_dir).load_extended(
            interp_method=self.interp_method,
        )

        print("[preprocessor] Atribuindo clusters...")
        station_clusters = assign_station_clusters(ds_inmet, self.shp_dir)
        cluster_of = station_clusters.set_index("estacao")["cluster_id"]
        coords = station_coords(ds_inmet)

        times = pd.DatetimeIndex(ds_era5["time"].values)
        split = split_from_config(times, self.split_cfg, self.seed)
        print(
            f"[preprocessor] Split por blocos de mês: teste={list(split.test_months)}, "
            f"{len(split.val_units)} blocos (ano, mês) de validação."
        )

        print(f"[preprocessor] Climatologia ERA5 harmônica (n={self.n_harmonics}, dias de treino)...")
        clim = get_harmonic_climatology(
            ds_era5, ERA5_GUST_PROXY, times[split.label(times) == "train"], self.n_harmonics,
        ).reset_coords(drop=True)

        print("[preprocessor] Construindo DataFrame de features...")
        df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, clim, require_target=False)
        avail = resolve_feature_groups(self.feature_groups, df.columns)
        df = select_complete_rows(df, avail, label="preprocessor")
        cluster_ids = sorted(df["cluster_id"].unique().tolist())

        stations = self._calendarize(df, split, avail)

        fit = pd.concat(
            [g.loc[(g["_label"] == "train") & g[self.target_var].notna(), avail + [self.target_var]]
             for _, g in stations],
            axis=0,
        )
        if fit.empty:
            raise ValueError("nenhuma linha de treino com alvo — confira split/date_range e os dados")
        print(f"[preprocessor] Ajustando scalers em {len(fit)} linhas de treino...")
        assert_no_missing(fit[avail], "features de treino")
        scaler_x = RobustScaler().fit(fit[avail])
        scaler_y = RobustScaler().fit(fit[[self.target_var]])
        del fit

        feature_names = avail + [f"cluster_{c}" for c in cluster_ids]
        print(f"[preprocessor] Janelas diárias (L={self.lookback}, {len(feature_names)} features)...")
        buckets = WindowBuckets()
        for est, g in stations:
            cid = cluster_of.get(est, -1)
            # Dias sem linha completa ficam NaN aqui, mas são rótulo "out" e a
            # purga descarta toda janela que os toque.
            base = scaler_x.transform(g[avail]).astype("float32")
            onehot = np.broadcast_to(cluster_onehot(cluster_ids, cid), (len(g), len(cluster_ids)))
            mat = np.hstack([base, onehot])
            labels = g["_label"].to_numpy()

            windows, tpos, keep = build_daily_windows(mat, g.index, self.lookback, labels=labels)
            y = g[self.target_var].to_numpy(dtype=float)[tpos]
            keep &= np.isfinite(y) & np.isfinite(windows).all(axis=(1, 2))
            y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel() if len(y) else y
            target_labels = labels[tpos]
            lat, lon = coords.loc[est, "latitude"], coords.loc[est, "longitude"]
            for sp in SPLITS:
                idx = np.flatnonzero(keep & (target_labels == sp))
                if len(idx):
                    buckets.add(
                        sp, windows[idx], y_scaled[idx], g.index[tpos[idx]],
                        estacao=est, latitude=lat, longitude=lon, cluster_id=cid,
                    )

        batch = ClusterDataBatch(
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            feature_names=feature_names,
            cluster_ids=cluster_ids,
            station_clusters_df=station_clusters,
            resolution="daily",
            window_length=self.lookback,
            split=split,
            interp_method=self.interp_method,
            climatology={"method": "harmonic", "n_harmonics": self.n_harmonics},
            hourly=None,
        )
        buckets.fill(batch, self.lookback, len(feature_names))
        for sp in SPLITS:
            n = sum(len(v) for v in getattr(batch, f"x_{sp}").values())
            print(f"[preprocessor]   {sp}: {n} janelas")
        return batch

    def _calendarize(
        self, df: pd.DataFrame, split: MonthBlockSplit, avail: list[str],
    ) -> list[tuple[str, pd.DataFrame]]:
        """Por estação: calendário diário contínuo (índice = time), rótulo de
        split em `_label` (dia sem linha no merge → "out", purgado junto com
        toda janela que o toque)."""
        keep_cols = list(dict.fromkeys([self.target_var, *avail]))
        out = []
        for est, g in df.groupby("estacao", sort=True):
            g = g.set_index("time").sort_index()
            g = g[~g.index.duplicated(keep="first")][keep_cols]
            cal = pd.date_range(g.index[0], g.index[-1], freq="D", name="time")
            present = cal.isin(g.index)
            g = g.reindex(cal)
            labels = split.label(cal)
            labels[~present] = "out"
            g["_label"] = labels
            out.append((est, g))
        return out
