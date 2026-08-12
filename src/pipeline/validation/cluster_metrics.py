from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS
from src.pipeline.training.cluster_trainer import (
    ClusterTrainResult,
    ClusterTrainer,
)
from src.pipelines.common import compute_metrics


def get_cluster_season_arrays(data, preds, season, cluster_id):
    """Retorna (y_true, y_pred) absolutos p/ um cluster×season no teste, ou None.

    Aplica inverse_transform + multiplicação pelo ERA5 (razão → absoluto),
    mascara pela coluna one-hot do cluster e filtra NaN. Usado tanto por
    ClusterMetricsEvaluator quanto por src.visualization.cluster_plots para
    evitar duplicar a lógica de mascaramento em dois lugares.
    """
    if season not in data.x_test or len(data.x_test[season]) == 0:
        return None
    col = f"cluster_{cluster_id}"
    if col not in data.feature_names:
        return None
    col_idx = data.feature_names.index(col)

    x_all = data.x_test[season].astype("float32")
    y_all = data.y_test[season].astype("float32")
    era5_all = np.array(data.era5_test[season], dtype="float32")
    n = min(len(x_all), len(y_all), len(era5_all))
    y_all = y_all[:n]
    era5_all = era5_all[:n]

    era5_safe = np.clip(era5_all, 0.1, None)
    y_true_abs = data.scaler_y.inverse_transform(y_all).flatten() * era5_safe
    y_pred_abs = preds.get(season)
    if y_pred_abs is None:
        return None
    y_pred_abs = y_pred_abs[:n]

    mask = x_all[:n, -1, col_idx] == 1
    if not np.any(mask):
        return None

    yt = y_true_abs[mask]
    yp = y_pred_abs[mask]
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return None
    return yt[valid], yp[valid]


def get_cluster_season_station_arrays(data, preds, season, cluster_id):
    """Como get_cluster_season_arrays, mas também retorna estacao/latitude/
    longitude/time por amostra (de data.meta_test) — usado pra montar
    predictions_by_station.csv com granularidade real de estação em vez de
    só o agregado por cluster_id. Retorna (y_true, y_pred, estacao,
    latitude, longitude, time) ou None.
    """
    if season not in data.x_test or len(data.x_test[season]) == 0:
        return None
    col = f"cluster_{cluster_id}"
    if col not in data.feature_names:
        return None
    col_idx = data.feature_names.index(col)

    x_all = data.x_test[season].astype("float32")
    y_all = data.y_test[season].astype("float32")
    era5_all = np.array(data.era5_test[season], dtype="float32")
    meta = data.meta_test.get(season, {})
    n = min(len(x_all), len(y_all), len(era5_all))
    y_all = y_all[:n]
    era5_all = era5_all[:n]

    era5_safe = np.clip(era5_all, 0.1, None)
    y_true_abs = data.scaler_y.inverse_transform(y_all).flatten() * era5_safe
    y_pred_abs = preds.get(season)
    if y_pred_abs is None:
        return None
    y_pred_abs = y_pred_abs[:n]

    mask = x_all[:n, -1, col_idx] == 1
    if not np.any(mask):
        return None

    yt = y_true_abs[mask]
    yp = y_pred_abs[mask]
    estacao = np.asarray(meta.get("estacao"))[:n][mask]
    latitude = np.asarray(meta.get("latitude"))[:n][mask]
    longitude = np.asarray(meta.get("longitude"))[:n][mask]
    time = np.asarray(meta.get("time"))[:n][mask]

    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return None
    return (
        yt[valid], yp[valid], estacao[valid],
        latitude[valid], longitude[valid], time[valid],
    )


class ClusterMetricsEvaluator:
    """Avalia RMSE e correlacao de Pearson por cluster x estacao climatica."""

    def evaluate(
        self,
        data: ClusterDataBatch,
        result,
        trainer,
    ) -> pd.DataFrame:
        """Retorna DataFrame com [cluster_id, season, n_samples, R2, RMSE, Bias, Bias_P90, RMSE_P90, Corr]."""
        from src.pipeline.training.cluster_tr_trainer import ClusterTRTrainer

        if isinstance(trainer, ClusterTRTrainer):
            preds = trainer.predict(
                x_dict=data.x_test,
                xs_dict=data.x_static_test,
                models=result.models,
                era5_dict=data.era5_test,
                scaler_y=data.scaler_y,
                feature_names=data.feature_names,
                cluster_ids=data.cluster_ids,
            )
        else:
            preds = trainer.predict(
                x_dict=data.x_test,
                models=result.models,
                era5_dict=data.era5_test,
                scaler_y=data.scaler_y,
                feature_names=data.feature_names,
                cluster_ids=data.cluster_ids,
            )

        # Diagnóstico de cobertura
        trained_keys = {
            (cid, s)
            for cid, sdict in (
                result.models.items() if hasattr(result, "models") else []
            )
            for s in sdict
        }
        print(
            f"[eval] Modelos treinados: {len(trained_keys)} "
            f"(clusters: {sorted({k[0] for k in trained_keys})}, "
            f"seasons: {sorted({k[1] for k in trained_keys})})"
        )
        for season, yp in preds.items():
            n_nan = int(np.isnan(yp).sum()) if yp is not None else -1
            n_tot = len(yp) if yp is not None else 0
            print(f"[eval] Preds {season}: {n_tot} amostras, {n_nan} NaN")

        rows = []
        for season in SEASONS:
            for cluster_id in data.cluster_ids:
                arrays = get_cluster_season_arrays(data, preds, season, cluster_id)
                if arrays is None:
                    continue
                yt, yp = arrays

                core = compute_metrics(yt, yp)
                corr = (
                    float(np.corrcoef(yt, yp)[0, 1])
                    if len(yt) > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "cluster_id": str(cluster_id),
                        "season": season,
                        "n_samples": int(len(yt)),
                        "R2": round(core["R2"], 4),
                        "RMSE": round(core["RMSE"], 3),
                        "Bias": round(core["Bias"], 3),
                        "Bias_P90": round(core["Bias_P90"], 3),
                        "RMSE_P90": round(core["RMSE_P90"], 3),
                        "Corr": round(corr, 3),
                    }
                )
        if not rows:
            print(
                "[eval] AVISO: nenhuma previsão válida — "
                "verifique se os modelos treinados cobrem os seasons do teste."
            )
            return pd.DataFrame(
                columns=[
                    "cluster_id", "season", "n_samples",
                    "R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90", "Corr",
                ]
            )

        return (
            pd.DataFrame(rows)
            .sort_values(["cluster_id", "season"])
            .reset_index(drop=True)
        )
