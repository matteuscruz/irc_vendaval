from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS
from src.pipeline.data.target import inverse_target
from src.pipeline.training.cluster_trainer import cluster_mask
from src.pipelines.common import compute_metrics


def _select(data, preds, season, cluster_id):
    """(índices válidos das janelas de teste do cluster, y_true, y_pred) em
    m/s, ou None. y_true vem do inverso de scaler_y (sem clip)."""
    x_all = data.x_test.get(season)
    y_pred = preds.get(season)
    if x_all is None or len(x_all) == 0 or y_pred is None:
        return None
    mask = cluster_mask(x_all, data.feature_names, cluster_id)
    if mask is None:
        return None
    y_true = inverse_target(data.scaler_y, data.y_test[season])
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(mask), len(y_true), len(y_pred))
    idx = np.flatnonzero(mask[:n])
    idx = idx[np.isfinite(y_true[idx]) & np.isfinite(y_pred[idx])]
    if len(idx) == 0:
        return None
    return idx, y_true[idx], y_pred[idx]


def get_cluster_season_arrays(data, preds, season, cluster_id):
    """Retorna (y_true, y_pred) em m/s p/ um cluster × estação do ano no
    teste, ou None. Usado por ClusterMetricsEvaluator e cluster_plots."""
    sel = _select(data, preds, season, cluster_id)
    return None if sel is None else (sel[1], sel[2])


def get_cluster_season_station_arrays(data, preds, season, cluster_id):
    """Como get_cluster_season_arrays, mais estacao/latitude/longitude/time
    por amostra (de data.meta_test) — base do predictions_by_station.csv.
    Retorna (y_true, y_pred, estacao, latitude, longitude, time) ou None."""
    sel = _select(data, preds, season, cluster_id)
    if sel is None:
        return None
    idx, yt, yp = sel
    meta = data.meta_test.get(season, {})
    return (
        yt, yp,
        np.asarray(meta.get("estacao"))[idx],
        np.asarray(meta.get("latitude"))[idx],
        np.asarray(meta.get("longitude"))[idx],
        np.asarray(meta.get("time"))[idx],
    )


class ClusterMetricsEvaluator:
    """Avalia RMSE e correlação de Pearson por cluster × estação do ano."""

    COLUMNS = ["cluster_id", "season", "n_samples", "R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90", "Corr"]

    def evaluate(self, data: ClusterDataBatch, result, preds: dict) -> pd.DataFrame:
        """`preds` (m/s) já vem calculado pelo chamador — ver
        `cluster_lstm._predict_lstm_experts`, que prediz uma única vez."""
        trained = {
            (cid, s)
            for cid, sdict in (result.models.items() if hasattr(result, "models") else [])
            for s in sdict
        }
        print(
            f"[eval] Modelos treinados: {len(trained)} "
            f"(clusters: {sorted({k[0] for k in trained})})"
        )
        for season, yp in preds.items():
            n_nan = int(np.isnan(yp).sum()) if yp is not None else -1
            print(f"[eval] Preds {season}: {len(yp) if yp is not None else 0} amostras, {n_nan} NaN")

        rows = []
        for season in SEASONS:
            for cluster_id in data.cluster_ids:
                arrays = get_cluster_season_arrays(data, preds, season, cluster_id)
                if arrays is None:
                    continue
                yt, yp = arrays
                core = compute_metrics(yt, yp)
                corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
                rows.append({
                    "cluster_id": str(cluster_id),
                    "season": season,
                    "n_samples": int(len(yt)),
                    "R2": round(core["R2"], 4),
                    "RMSE": round(core["RMSE"], 3),
                    "Bias": round(core["Bias"], 3),
                    "Bias_P90": round(core["Bias_P90"], 3),
                    "RMSE_P90": round(core["RMSE_P90"], 3),
                    "Corr": round(corr, 3),
                })
        if not rows:
            print(
                "[eval] AVISO: nenhuma previsão válida — "
                "verifique se os modelos treinados cobrem os clusters do teste."
            )
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(rows).sort_values(["cluster_id", "season"]).reset_index(drop=True)
