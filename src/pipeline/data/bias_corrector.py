from __future__ import annotations

import numpy as np

# Features que recebem correção QDM contra INMET gust
_WIND_NAMES: frozenset[str] = frozenset(
    {"10m_u_component_of_wind", "10m_v_component_of_wind", "wind_mag"}
)
_WIND_MAG = "wind_mag"


class ERA5BiasCorrector:
    """
    Quantile Delta Mapping (QDM) de ERA5 wind → escala INMET gust.

    Treina em espaço scaled (pós-RobustScaler): aprende delta(q) por quantil
    comparando ERA5 wind_mag com y_train (anomalia INMET) poolado em todas
    as estações climáticas. Aplica o mesmo delta a u, v, e wind_mag em todas
    as timesteps do lookback.

    Temperatura e pressão não são corrigidas (ERA5 confiável nessas vars).
    """

    def __init__(self) -> None:
        self._era5_q: np.ndarray | None = None
        self._delta_q: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------

    def fit(
        self,
        data: object,  # ClusterDataBatch
        n_quantiles: int = 100,
    ) -> None:
        """
        Aprende o mapeamento quantil ERA5_wind_mag → INMET_gust no
        período de treino (poolado em todas as estações climáticas).
        """
        from src.pipeline.data.cluster_preprocessor import SEASONS

        if _WIND_MAG not in data.feature_names:
            return

        wm_idx = data.feature_names.index(_WIND_MAG)

        era5_parts: list[np.ndarray] = []
        inmet_parts: list[np.ndarray] = []

        for s in SEASONS:
            if s not in data.x_train:
                continue
            # Última timestep (mais recente) do lookback como referência
            era5_parts.append(data.x_train[s][:, -1, wm_idx].astype("float64"))
            inmet_parts.append(data.y_train[s][:, 0].astype("float64"))

        if not era5_parts:
            return

        era5_pooled = np.concatenate(era5_parts)
        inmet_pooled = np.concatenate(inmet_parts)

        n_pts = min(n_quantiles, len(era5_pooled))
        q_pts = np.linspace(0.0, 1.0, n_pts)

        self._era5_q = np.quantile(era5_pooled, q_pts)
        self._delta_q = np.quantile(inmet_pooled, q_pts) - self._era5_q
        self._fitted = True

    # ------------------------------------------------------------------

    def transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """
        Aplica correção QDM às colunas de vento (u, v, wind_mag) em todas
        as timesteps. Retorna cópia corrigida; shape preservado.
        """
        if not self._fitted or self._era5_q is None:
            return X

        wind_idx = [
            i for i, name in enumerate(feature_names) if name in _WIND_NAMES
        ]
        if not wind_idx:
            return X

        X = X.copy()
        for f in wind_idx:
            vals = X[:, :, f]           # (N, lookback)
            delta = np.interp(
                vals.ravel(),
                self._era5_q,
                self._delta_q,
                left=float(self._delta_q[0]),
                right=float(self._delta_q[-1]),
            ).reshape(vals.shape).astype(vals.dtype)
            X[:, :, f] = vals + delta

        return X
