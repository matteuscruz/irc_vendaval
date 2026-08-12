from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler


class StationStaticFeatures:
    """
    Computa features estáticas por estação a partir do período de treino.

    Features produzidas (5 no total):
        lat, lon        — coordenadas geográficas (ds_inmet.latitude/longitude)
        gust_P10        — 10º percentil de daily_wind_gust_max no treino
        gust_P50        — mediana de daily_wind_gust_max no treino
        gust_P90        — 90º percentil de daily_wind_gust_max no treino

    O StandardScaler é ajustado apenas nas estações de treino para evitar
    leakage. Estações de teste são transformadas com os parâmetros do treino.
    """

    FEATURE_NAMES = ["lat", "lon", "gust_P10", "gust_P50", "gust_P90"]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False

    def compute(
        self,
        ds_inmet: xr.Dataset,
        target_var: str,
        train_slice: slice,
        train_stations: np.ndarray,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame com colunas [estacao, lat, lon, gust_P10, gust_P50, gust_P90].
        O scaler é ajustado nas estações de treino; todas as estações são transformadas.

        Deve ser chamado uma única vez por experimento. O scaler fica armazenado
        em self.scaler para uso posterior na inferência.
        """
        lats = ds_inmet.latitude.values
        lons = ds_inmet.longitude.values
        stations = ds_inmet.estacao.values

        gust_train = (
            ds_inmet[target_var]
            .sel(time=train_slice)
            .values  # (n_time, n_estacao)
        )

        rows = []
        for i, sta in enumerate(stations):
            vals = gust_train[:, i]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                p10, p50, p90 = np.nan, np.nan, np.nan
            else:
                p10, p50, p90 = np.percentile(vals, [10, 50, 90])
            rows.append(
                {
                    "estacao": sta,
                    "lat": float(lats[i]),
                    "lon": float(lons[i]),
                    "gust_P10": float(p10),
                    "gust_P50": float(p50),
                    "gust_P90": float(p90),
                }
            )

        df = pd.DataFrame(rows)

        # Ajustar scaler apenas nas estações de treino
        df_train = df[df["estacao"].isin(train_stations)]
        self.scaler.fit(df_train[self.FEATURE_NAMES].values)
        self._fitted = True

        scaled = self.scaler.transform(df[self.FEATURE_NAMES].values)
        df_scaled = pd.DataFrame(scaled, columns=self.FEATURE_NAMES)
        df_scaled.insert(0, "estacao", df["estacao"].values)

        return df_scaled

    def n_features(self) -> int:
        return len(self.FEATURE_NAMES)
