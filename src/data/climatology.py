from __future__ import annotations

import xarray as xr


def get_climatology(
    ds_target: xr.Dataset,
    target_var: str,
    train_slice: slice,
) -> xr.DataArray:
    """Média histórica por dia-do-ano, calculada apenas no período de treino."""
    return (
        ds_target[target_var]
        .sel(time=train_slice)
        .groupby("time.dayofyear")
        .mean(dim="time")
    )
