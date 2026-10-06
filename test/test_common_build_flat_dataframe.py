"""Teste de regressão pro fix de `build_flat_dataframe(require_target=...)`.

Bug real: a função descartava (`dropna`) toda linha sem alvo INMET,
compartilhada entre treino (onde isso é correto) e inferência/correção
(onde é errado — o modelo só usa features do ERA5 pra prever, o alvo INMET
não é insumo). Na prática isso zerava a grade corrigida em anos com pouca
estação INMET ativa (ex.: 2000-2006), justamente os anos que a extrapolação
deveria cobrir. Ver `src/inference/spatial_correction.py` e
`spatial_correction_dl.py` (agora chamam com `require_target=False`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.pipelines.common import TARGET_VAR, build_flat_dataframe


def _tiny_dataset():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    stations = ["A1", "A2"]

    # A2 sem observação INMET nos 5 primeiros dias (simula estação
    # inexistente/inativa naquele período, como em 2000-2006 no dataset real).
    gust = np.array([
        [1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan], [5.0, np.nan],
        [6.0, 7.0], [6.5, 7.5], [7.0, 8.0], [7.5, 8.5], [8.0, 9.0],
    ])
    ds_inmet = xr.Dataset(
        {TARGET_VAR: (("time", "estacao"), gust)},
        coords={
            "time": dates, "estacao": stations,
            "latitude": ("estacao", [-20.0, -21.0]),
            "longitude": ("estacao", [-45.0, -46.0]),
        },
    )
    wind = np.random.default_rng(0).normal(5, 1, size=(10, 2))
    ds_era5 = xr.Dataset(
        {"wind_mag_max": (("time", "estacao"), wind)},
        coords={"time": dates, "estacao": stations},
    )
    station_clusters = pd.DataFrame({"estacao": stations, "cluster_id": [1, 1]})
    ds_clim = xr.DataArray(
        np.full((366, 2), 5.0),
        dims=("dayofyear", "estacao"),
        coords={"dayofyear": np.arange(1, 367), "estacao": stations},
    )
    return ds_inmet, ds_era5, station_clusters, ds_clim


def test_require_target_true_drops_rows_without_inmet_observation():
    ds_inmet, ds_era5, station_clusters, ds_clim = _tiny_dataset()
    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim, require_target=True)
    assert len(df) == 15  # 20 linhas - 5 de A2 sem observação
    assert df[TARGET_VAR].notna().all()


def test_require_target_false_keeps_rows_without_inmet_observation():
    ds_inmet, ds_era5, station_clusters, ds_clim = _tiny_dataset()
    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim, require_target=False)
    assert len(df) == 20  # nenhuma linha descartada por falta de alvo
    assert df[TARGET_VAR].isna().sum() == 5


def test_default_still_requires_target_backward_compatible():
    """Chamadores de treino (cluster_lazy/mlp/lstm) não passam
    `require_target` explicitamente — o default precisa continuar True."""
    ds_inmet, ds_era5, station_clusters, ds_clim = _tiny_dataset()
    df_default = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim)
    df_explicit_true = build_flat_dataframe(
        ds_inmet, ds_era5, station_clusters, ds_clim, require_target=True,
    )
    assert len(df_default) == len(df_explicit_true) == 15
