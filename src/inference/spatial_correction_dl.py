"""Inferência Espacial — Correção de Rajadas ERA5 via LSTM (schema v2).

Carrega os modelos Keras + `dl_metadata.joblib` salvos pela pipeline
cluster_lstm, reconstrói as janelas com a MESMA convenção do treino
(`src/pipeline/data/windowing.py`: janela [D-L+1 .. D]) e interpola as
predições para a grade ERA5. A saída do modelo já é a rajada em m/s — sem
razão × ERA5 nem troca de cabeça. Só modelos DIÁRIOS: não há ERA5 horário em
grade. Artefatos dual-head antigos são recusados (`LegacyLSTMArtifactError`).

Estruturalmente paralelo a `SpatialCorrector` (joblib): só `prepare()` e
`predict_stations()` são sobrescritos.

Uso:
    python -m src.inference.spatial_correction_dl \\
        --raw-dir dataset/raw --shp-dir dataset/shp \\
        --models-dir artifacts/modal/experiments/original/fitted_models \\
        --out-dir output_corrigido_dl --year 2024
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.climatology import get_harmonic_climatology
from src.data.cluster_assigner import assign_station_clusters
from src.inference.dl_metadata import load_dl_metadata
from src.inference.spatial_correction import SpatialCorrector
from src.pipeline.data.splits import MonthBlockSplit
from src.pipeline.data.target import inverse_target
from src.pipeline.data.windowing import build_daily_windows
from src.pipelines.common import ERA5_GUST_PROXY, TARGET_VAR, build_flat_dataframe

warnings.filterwarnings("ignore")

_OUT_COLS = [
    "time", "estacao", "latitude", "longitude", "cluster_id",
    TARGET_VAR, ERA5_GUST_PROXY, "rajada_corrigida",
]


class DLSpatialCorrector(SpatialCorrector):
    """Herda de SpatialCorrector; sobrescreve prepare() e predict_stations()
    pra lidar com Keras + janelas temporais."""

    def prepare(self) -> None:
        import tensorflow as tf

        self.metadata = load_dl_metadata(self.models_dir, require_resolution="daily")
        interp_method = self.metadata.get("interp_method", "nearest")
        print(
            f"[DLSpatialCorrector] Metadados v{self.metadata['schema_version']}: "
            f"lookback={self.metadata['lookback']}, interp={interp_method}"
        )

        keras_files = sorted(self.models_dir.glob("best_model_c*.keras"))
        if not keras_files:
            raise FileNotFoundError("Nenhum modelo .keras encontrado.")
        print(f"[DLSpatialCorrector] Carregando {len(keras_files)} modelo(s) Keras...")
        for kf in keras_files:
            cid = int(kf.stem.replace("best_model_c", ""))
            self.cluster_models[cid] = tf.keras.models.load_model(kf, compile=False)
            print(f"  ✓ Cluster {cid} carregado")

        print("\n[DLSpatialCorrector] Carregando dados (load_extended)...")
        ds_inmet, ds_era5 = self.loader.load_extended(interp_method=interp_method)

        self._station_lats = ds_inmet.latitude.values.astype(float)
        self._station_lons = ds_inmet.longitude.values.astype(float)
        self._station_ids = ds_inmet.estacao.values
        self._grid_lats, self._grid_lons = self._load_era5_basin_grid()

        print("[DLSpatialCorrector] Atribuindo clusters...")
        station_clusters = assign_station_clusters(ds_inmet, self.shp_dir)

        # Climatologia ERA5 (feature `era5_clim_wind`) reproduzida EXATAMENTE
        # como no treino: harmônica, ajustada nos dias rotulados "train" pelo
        # split salvo no metadata (antes a inferência usava TRAIN_SLICE e o
        # treino outro período — desalinhamento silencioso).
        split = MonthBlockSplit.from_dict(self.metadata["split"])
        times = pd.DatetimeIndex(ds_era5["time"].values)
        n_harmonics = int(self.metadata.get("climatology", {}).get("n_harmonics", 3))
        print("[DLSpatialCorrector] Climatologia ERA5 harmônica (dias de treino do split salvo)...")
        ds_clim = get_harmonic_climatology(
            ds_era5, ERA5_GUST_PROXY, times[split.label(times) == "train"], n_harmonics,
        ).reset_coords(drop=True)

        print("[DLSpatialCorrector] Construindo DataFrame flat...")
        self.df_all = build_flat_dataframe(
            ds_inmet, ds_era5, station_clusters, ds_clim, require_target=False,
        )
        print(f"  Shape: {self.df_all.shape}")

        ds_inmet.close()
        ds_era5.close()

    def predict_stations(self, time_slice: tuple[str, str]) -> pd.DataFrame:
        """Janelas diárias por estação sobre calendário contínuo e predição em
        lote — uma chamada Keras por CLUSTER."""
        meta = self.metadata
        lookback = int(meta["lookback"])
        base_features = list(meta["base_feature_names"])
        cluster_cols = list(meta["cluster_feature_names"])
        scaler_x, scaler_y = meta["scaler_x"], meta["scaler_y"]

        t0 = pd.Timestamp(time_slice[0])
        start = t0 - pd.Timedelta(days=lookback - 1)
        df_ext = self.df_all[
            (self.df_all["time"] >= start) & (self.df_all["time"] <= time_slice[1])
        ]
        if df_ext.empty:
            return pd.DataFrame(columns=_OUT_COLS)

        missing_base = [f for f in base_features if f not in df_ext.columns]
        if missing_base:
            raise ValueError(
                f"{len(missing_base)} features do modelo ausentes nos dados: {missing_base[:5]} "
                "— nada é imputado (confira dataset/raw/new_features)"
            )

        info_cols = ["latitude", "longitude", TARGET_VAR, ERA5_GUST_PROXY]
        windows_by_cluster: dict[int, list[np.ndarray]] = {}
        rows_by_cluster: dict[int, list[pd.DataFrame]] = {}
        for est, g in df_ext.groupby("estacao", sort=False):
            cid = int(g["cluster_id"].iloc[0])
            if cid not in self.cluster_models:
                continue
            g = g.set_index("time").sort_index()
            g = g[~g.index.duplicated(keep="first")]
            cal = pd.date_range(g.index[0], g.index[-1], freq="D")
            present = cal.isin(g.index)
            g = g.reindex(cal)

            base = scaler_x.transform(g[base_features])
            onehot = np.array([c == f"cluster_{cid}" for c in cluster_cols], dtype=float)
            mat = np.hstack([base, np.broadcast_to(onehot, (len(g), len(cluster_cols)))])
            windows, tpos, _ = build_daily_windows(
                mat.astype("float32"), cal, lookback, pad="edge",
            )
            # Sem imputação: janela com qualquer feature ausente não é predita.
            keep = present[tpos] & (cal[tpos] >= t0) & np.isfinite(windows).all(axis=(1, 2))
            if not keep.any():
                continue
            idx = tpos[keep]
            windows_by_cluster.setdefault(cid, []).append(windows[keep])
            info = g.iloc[idx][[c for c in info_cols if c in g.columns]].copy()
            info["time"] = cal[idx]
            info["estacao"] = est
            info["cluster_id"] = cid
            rows_by_cluster.setdefault(cid, []).append(info)

        frames = []
        for cid, wins in windows_by_cluster.items():
            X = np.concatenate(wins, axis=0)
            pred = self.cluster_models[cid].predict(X, batch_size=1024, verbose=0)
            info = pd.concat(rows_by_cluster[cid], ignore_index=True)
            info["rajada_corrigida"] = inverse_target(scaler_y, np.ravel(pred), clip=True)
            frames.append(info)
        if not frames:
            return pd.DataFrame(columns=_OUT_COLS)
        out = pd.concat(frames, ignore_index=True)
        for c in _OUT_COLS:
            if c not in out.columns:
                out[c] = np.nan
        return out[_OUT_COLS]


if __name__ == "__main__":
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Inferência Espacial LSTM (v2)")
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--shp-dir", default="dataset/shp")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--out-dir", default="output_corrigido_dl")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--smoothing", choices=["none", "gaussian"], default="gaussian")
    args = parser.parse_args()

    t0 = _time.time()
    corr = DLSpatialCorrector(args.raw_dir, args.shp_dir, args.models_dir)
    corr.prepare()

    time_slice = (f"{args.year}-01-01", f"{args.year}-12-31")
    ds = corr.infer_grid(time_slice, smoothing=args.smoothing)

    out_path = Path(args.out_dir) / f"era5_corrigido_{args.year}_{args.smoothing}.nc"
    corr.save(ds, out_path)
    print(f"\nConcluído em {_time.time() - t0:.1f}s")
