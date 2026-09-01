"""Inferência Espacial — Correção de Rajadas ERA5 via LSTM.

Carrega os modelos Keras salvos pela pipeline cluster_lstm, reconstrói as
janelas temporais usando a mesma lógica do treino (`_make_windows` em
`src/pipeline/data/cluster_preprocessor.py`) e interpola as predições para a
grade ERA5. Estruturalmente paralelo a `SpatialCorrector` (joblib): só
`prepare()` e `predict_stations()` são sobrescritos — `infer_grid()`/`save()`/
`_plot_map()`/`_idw_interpolate` são herdados sem mudança.

Uso:
    python -m src.inference.spatial_correction_dl \\
        --raw-dir dataset/raw --shp-dir dataset/shp \\
        --models-dir artifacts/modal/experiments/original/fitted_models \\
        --out-dir output_corrigido_dl --year 2024
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.inference.spatial_correction import SpatialCorrector
from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_climatology
from src.pipelines.common import (
    ERA5_GUST_PROXY, TARGET_VAR, TRAIN_SLICE, build_flat_dataframe,
)

warnings.filterwarnings("ignore")

# Variante dual-input (sequência + estáticas) usada por ClusterTRTrainer —
# não suportada aqui, que só constrói o tensor de sequência. A config ativa
# de produção (experiment_cluster_lstm_modal.yaml) usa "cluster_dual_head_lstm",
# um único tensor de entrada — ver ClusterDualHeadLSTMBuilder/ClusterTrainer.
_UNSUPPORTED_MODEL_NAMES = {"cluster_tr_lstm"}


class DLSpatialCorrector(SpatialCorrector):
    """Herda de SpatialCorrector; sobrescreve prepare() e predict_stations()
    pra lidar com Keras + janelas temporais deslizantes."""

    def prepare(self) -> None:
        """Carrega os modelos Keras salvos e monta o DataFrame flat (mesma
        fonte que SpatialCorrector usa, sem o ClusterPreprocessor completo —
        esse caminho reconstruía janelas de treino/val inteiras só pra
        descartá-las depois, um desperdício de minutos por chamada)."""
        import tensorflow as tf

        meta_path = self.models_dir / "dl_metadata.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {meta_path}")

        self.metadata = joblib.load(meta_path)
        model_name = self.metadata.get("model_name")
        if model_name in _UNSUPPORTED_MODEL_NAMES:
            raise NotImplementedError(
                f"model_name='{model_name}' usa entrada dual (sequência + "
                "estáticas via ClusterTRTrainer.predict(x_dict, xs_dict, ...)) "
                "— DLSpatialCorrector só suporta o tensor único de "
                "'cluster_dual_head_lstm' (a config de produção atual)."
            )

        self.extreme_threshold = self.metadata.get("extreme_threshold")
        if self.extreme_threshold is None:
            self.extreme_threshold = 1.5
            print(
                "[DLSpatialCorrector] AVISO: 'extreme_threshold' ausente no "
                "metadata (artefato salvo antes dessa chave existir) — "
                f"usando default {self.extreme_threshold}."
            )

        print(f"[DLSpatialCorrector] Metadados carregados: lookback={self.metadata['lookback']}")

        keras_files = sorted(self.models_dir.glob("best_model_c*.keras"))
        if not keras_files:
            raise FileNotFoundError("Nenhum modelo .keras encontrado.")

        print(f"[DLSpatialCorrector] Carregando {len(keras_files)} modelo(s) Keras...")
        for kf in keras_files:
            cid = int(kf.stem.replace("best_model_c", ""))
            model = tf.keras.models.load_model(kf, compile=False)
            self.cluster_models[cid] = model
            print(f"  ✓ Cluster {cid} carregado")

        print("\n[DLSpatialCorrector] Carregando dados (load_extended)...")
        ds_inmet, ds_era5 = self.loader.load_extended()

        self._station_lats = ds_inmet.latitude.values.astype(float)
        self._station_lons = ds_inmet.longitude.values.astype(float)
        self._station_ids = ds_inmet.estacao.values

        # Grid real do ERA5-Basin (herdado de SpatialCorrector).
        self._grid_lats, self._grid_lons = self._load_era5_basin_grid()

        print("[DLSpatialCorrector] Atribuindo clusters...")
        station_clusters = assign_station_clusters(ds_inmet, self.shp_dir)

        # Climatologia ERA5 (feature de entrada `era5_clim_wind`) — precisa
        # ser a MESMA fonte usada no treino (ds_era5/ERA5_GUST_PROXY via
        # ClusterPreprocessor), não a climatologia INMET do alvo bruto
        # (bug anterior: usava ds_inmet/TARGET_VAR, um desalinhamento
        # silencioso entre o que o modelo viu no treino e o que via aqui).
        print("[DLSpatialCorrector] Calculando climatologia ERA5...")
        ds_clim = get_climatology(ds_era5, ERA5_GUST_PROXY, slice(*TRAIN_SLICE))

        print("[DLSpatialCorrector] Construindo DataFrame flat...")
        self.df_all = build_flat_dataframe(
            ds_inmet, ds_era5, station_clusters, ds_clim, require_target=False,
        )
        print(f"  Shape: {self.df_all.shape}")

        # Fechar os handles xarray/netCDF4 — ver comentário equivalente em
        # SpatialCorrector.prepare() (spatial_correction.py).
        ds_inmet.close()
        ds_era5.close()

    def predict_stations(self, time_slice: tuple[str, str]) -> pd.DataFrame:
        """Reconstrói as janelas deslizantes por estação (mesma lógica de
        `_make_windows`: features base escalonadas por `scaler_x`, one-hots
        de cluster concatenados SEM escala) e prediz em lote — uma chamada
        Keras por CLUSTER (todas as estações daquele cluster de uma vez),
        não uma por estação."""
        lookback = self.metadata["lookback"]
        features = self.metadata["feature_names"]
        scaler_x = self.metadata["scaler_x"]
        scaler_y = self.metadata["scaler_y"]
        imputer_x = self.metadata.get("imputer_x")

        start_date = pd.to_datetime(time_slice[0]) - pd.Timedelta(days=lookback)
        df_ext = self.df_all[
            (self.df_all["time"] >= start_date) & (self.df_all["time"] <= time_slice[1])
        ].copy()
        df_ext = df_ext.sort_values(["estacao", "time"]).reset_index(drop=True)

        if df_ext.empty:
            return pd.DataFrame(columns=[
                "time", "estacao", "latitude", "longitude", "cluster_id",
                TARGET_VAR, ERA5_GUST_PROXY, "rajada_corrigida",
            ])

        # Mesma separação de _make_windows: features base IMPUTADAS (média do
        # treino) e depois ESCALONADAS por scaler_x; one-hots de cluster
        # concatenados SEM escala/imputação. Escalar as duas juntas (bug
        # anterior) corrompe a entrada — o scaler nunca viu as colunas de
        # cluster.
        base_features = [f for f in features if not str(f).startswith("cluster_")]
        cluster_cols = [f for f in features if str(f).startswith("cluster_")]

        missing_base = [f for f in base_features if f not in df_ext.columns]
        if missing_base:
            print(
                f"  ⚠ {len(missing_base)} features ausentes (fora de cobertura "
                f"era5_18z/bt55 dessa config): {missing_base[:5]}"
                f"{'...' if len(missing_base) > 5 else ''} — imputadas com a "
                "média do treino, igual ao treino (ver ClusterPreprocessor)."
            )
            for f in missing_base:
                df_ext[f] = np.nan

        for c in cluster_cols:
            try:
                target_cid = int(c.split("_", 1)[1])
            except (IndexError, ValueError):
                df_ext[c] = 0.0
                continue
            df_ext[c] = (df_ext["cluster_id"] == target_cid).astype(float)

        x_base_raw = df_ext[base_features]
        if imputer_x is not None:
            x_base_raw = imputer_x.transform(x_base_raw)
        x_base = scaler_x.transform(x_base_raw)
        x_cluster = df_ext[cluster_cols].to_numpy(dtype=float)
        x_all_matrix = np.hstack([x_base, x_cluster]).astype("float32")

        era5_all = df_ext[ERA5_GUST_PROXY].to_numpy(dtype="float32")
        target_all = (
            df_ext[TARGET_VAR].to_numpy(dtype="float32")
            if TARGET_VAR in df_ext.columns
            else np.full(len(df_ext), np.nan, dtype="float32")
        )
        time_all = df_ext["time"].to_numpy()
        lat_all = df_ext["latitude"].to_numpy()
        lon_all = df_ext["longitude"].to_numpy()
        estacao_all = df_ext["estacao"].to_numpy()
        cluster_id_all = df_ext["cluster_id"].to_numpy()

        # Rede de segurança: após a imputação acima, isfinite() só deve
        # disparar em casos genuinamente patológicos (ex.: estação sem
        # nenhuma coluna preenchida). Mesmo critério de `_make_windows`.
        windows_by_cluster: dict[int, list[np.ndarray]] = {}
        target_idx_by_cluster: dict[int, list[int]] = {}

        for _station, df_st in df_ext.groupby("estacao", sort=False):
            idx_arr = df_st.index.to_numpy()
            if len(idx_arr) <= lookback:
                continue
            cid = int(df_st["cluster_id"].iloc[0])
            if cid not in self.cluster_models:
                continue
            vals = x_all_matrix[idx_arr]
            for i in range(lookback, len(idx_arr)):
                window = vals[i - lookback:i]
                if not np.isfinite(window).all():
                    continue
                windows_by_cluster.setdefault(cid, []).append(window)
                target_idx_by_cluster.setdefault(cid, []).append(idx_arr[i])

        results = []
        for cid, windows in windows_by_cluster.items():
            model = self.cluster_models[cid]
            X_3d = np.stack(windows).astype("float32")
            target_idxs = np.asarray(target_idx_by_cluster[cid])

            # LSTM dual-head: cabeça normal decide; amostras acima do
            # threshold são substituídas pela cabeça extrema (mesmo ensemble
            # de ClusterTrainer.predict).
            outputs = model.predict(X_3d, batch_size=1024, verbose=0)
            if isinstance(outputs, list):
                pred_normal = outputs[0].reshape(-1)
                pred_extreme = outputs[1].reshape(-1)
                is_ext = pred_normal > self.extreme_threshold
                pred_final = pred_normal.copy()
                pred_final[is_ext] = pred_extreme[is_ext]
            else:
                pred_final = np.asarray(outputs).reshape(-1)

            # Reconstrução por RAZÃO (não mais anomalia climática — o LSTM
            # foi migrado pra essa convenção, igual ao MLP): abs = razão *
            # era5, âncora no dia-alvo (índice i da janela, não o dia
            # anterior). Mesma lógica de ClusterTrainer.predict.
            era5_target = era5_all[target_idxs]
            era5_safe = np.clip(era5_target, 0.1, None)
            abs_pred = (
                scaler_y.inverse_transform(pred_final.reshape(-1, 1)).flatten()
                * era5_safe
            )
            abs_pred = np.clip(abs_pred, 0, 80)  # range físico

            for j, ti in enumerate(target_idxs):
                t = time_all[ti]
                if pd.to_datetime(t) < pd.to_datetime(time_slice[0]):
                    continue
                results.append({
                    "time": t,
                    "estacao": estacao_all[ti],
                    "latitude": lat_all[ti],
                    "longitude": lon_all[ti],
                    "cluster_id": cluster_id_all[ti],
                    TARGET_VAR: target_all[ti],
                    ERA5_GUST_PROXY: era5_all[ti],
                    "rajada_corrigida": abs_pred[j],
                })

        cols = ["time", "estacao", "latitude", "longitude", "cluster_id",
                TARGET_VAR, ERA5_GUST_PROXY, "rajada_corrigida"]
        return pd.DataFrame(results, columns=cols) if results else pd.DataFrame(columns=cols)


if __name__ == "__main__":
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Inferência Espacial LSTM (V3)")
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
