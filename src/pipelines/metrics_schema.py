"""Schema comum de resultados/predições entre as pipelines de cluster
(cluster_lazy, cluster_mlp, cluster_lstm), para permitir comparação direta.

Cada pipeline mantém seus arquivos/colunas específicos; este módulo só define
o núcleo comum que alimenta results.csv / predictions.csv em cada experimento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.common import compute_metrics

CORE_RESULTS_COLUMNS = [
    "pipeline", "experiment", "cluster_id", "season", "split",
    "n_samples", "R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90",
]

CORE_PREDICTIONS_COLUMNS = [
    "pipeline", "experiment", "cluster_id", "season", "split", "y_true", "y_pred",
]


def build_results_row(
    *,
    pipeline: str,
    experiment: str,
    cluster_id,
    season: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Monta uma linha de results.csv com o schema comum entre pipelines."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = compute_metrics(y_true, y_pred)
    row = {
        "pipeline": pipeline,
        "experiment": experiment,
        "cluster_id": str(cluster_id),
        "season": season,
        "split": split,
        "n_samples": int(n_samples) if n_samples is not None else len(y_true),
        "R2": metrics["R2"],
        "RMSE": metrics["RMSE"],
        "Bias": metrics["Bias"],
        "Bias_P90": metrics["Bias_P90"],
        "RMSE_P90": metrics["RMSE_P90"],
    }
    if extra:
        row.update(extra)
    return row


def build_predictions_frame(
    *,
    pipeline: str,
    experiment: str,
    cluster_id,
    season: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Monta um DataFrame de predições com o schema comum entre pipelines."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    df = pd.DataFrame({
        "pipeline": pipeline,
        "experiment": experiment,
        "cluster_id": str(cluster_id),
        "season": season,
        "split": split,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df


def build_station_predictions_frame(
    *,
    estacao: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cluster_id,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time: np.ndarray | None = None,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Monta um DataFrame de predições POR ESTAÇÃO (não só por cluster).

    Usado por lazy/lstm pra alimentar predictions_by_station.csv com a mesma
    granularidade que cluster_mlp.py já produz — antes, essas duas pipelines
    só salvavam agregados por cluster_id (build_predictions_frame acima),
    perdendo a identidade da estação de cada previsão; o dashboard então
    tinha que "espalhar" a média do cluster pra todas as estações-membro,
    em vez de mostrar a previsão real de cada uma. estacao/latitude/
    longitude precisam estar alinhadas posicionalmente com y_true/y_pred
    (mesma ordem de linhas do DataFrame de origem).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    df = pd.DataFrame({
        "estacao": np.asarray(estacao),
        "latitude": np.asarray(latitude, dtype=float),
        "longitude": np.asarray(longitude, dtype=float),
        "cluster_id": str(cluster_id),
        "split": split,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if time is not None:
        df["time"] = np.asarray(time)
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df
