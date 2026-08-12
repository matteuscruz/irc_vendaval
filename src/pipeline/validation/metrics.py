from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.schema import ValidationConfig


@dataclass
class EvalResult:
    r2: float
    rmse: float
    mae: float
    bias: float
    kge: float = 0.0
    nrmse: float = 0.0
    metrics_by_percentile: dict[float, dict] = field(default_factory=dict)
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def _kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 1.0
    beta = float(np.mean(y_pred) / np.mean(y_true)) if np.mean(y_true) != 0 else 1.0
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _nrmse(rmse: float, y_true: np.ndarray) -> float:
    denom = float(np.max(y_true) - np.min(y_true))
    return float(rmse / denom) if denom > 0 else float("nan")


class MetricsEvaluator:
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cfg: ValidationConfig,
    ) -> EvalResult:
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        bias = float(np.mean(y_pred - y_true))
        kge = _kge(y_true, y_pred)
        nrmse = _nrmse(rmse, y_true)

        by_pct: dict[float, dict] = {}
        for p in cfg.percentiles:
            thresh = float(np.quantile(y_true, p))
            mask = y_true >= thresh
            if mask.sum() == 0:
                by_pct[p] = {"r2": float("nan"), "rmse": float("nan"), "bias": float("nan")}
                continue
            by_pct[p] = {
                "r2": float(r2_score(y_true[mask], y_pred[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                "bias": float(np.mean(y_pred[mask] - y_true[mask])),
            }

        return EvalResult(
            r2=r2,
            rmse=rmse,
            mae=mae,
            bias=bias,
            kge=kge,
            nrmse=nrmse,
            metrics_by_percentile=by_pct,
            y_pred=y_pred,
        )
