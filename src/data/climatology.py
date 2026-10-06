from __future__ import annotations

import numpy as np
import pandas as pd
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


def _harmonic_design(doy, n_harmonics: int) -> np.ndarray:
    phase = 2 * np.pi * (np.asarray(doy, dtype=float) - 1) / 365.25
    cols = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(k * phase), np.sin(k * phase)]
    return np.column_stack(cols)


def get_harmonic_climatology(
    ds_target: xr.Dataset,
    target_var: str,
    train_times,
    n_harmonics: int = 3,
    min_samples: int = 30,
) -> xr.DataArray:
    """Climatologia por dia-do-ano via série harmônica ajustada em `train_times`.

    `get_climatology` (média por dia-do-ano) não serve para o split por blocos
    de mês: ajustada só nos meses de treino, deixa os dias-do-ano dos meses de
    teste SEM valor, e `build_flat_dataframe` descarta linhas com climatologia
    NaN — apagando o teste inteiro. A série harmônica (constante + n pares
    cosseno/seno) é contínua no ano e fica definida em todos os dias 1..366,
    mesmo sem amostra de treino naquele mês.

    Ajuste por mínimos quadrados independente por ponto (estação ou célula),
    ignorando NaN. Pontos com menos de `min_samples` valores válidos ficam NaN.
    Retorna DataArray (dayofyear=1..366, <dims de ponto>) — mesmo formato de
    `get_climatology`, então os consumidores seguem com `.sel(dayofyear=...)`.
    """
    da = ds_target[target_var]
    if "time" not in da.dims:
        raise ValueError(f"{target_var} precisa ter a dimensão 'time'")
    point_dims = [d for d in da.dims if d != "time"]

    times = pd.DatetimeIndex(da["time"].values)
    mask = np.isin(times.values, pd.DatetimeIndex(train_times).values)
    da_tr = da.isel(time=np.flatnonzero(mask)).transpose("time", *point_dims)

    y = np.asarray(da_tr.values, dtype=float)
    point_shape = y.shape[1:]
    y = y.reshape(len(y), -1)
    X = _harmonic_design(pd.DatetimeIndex(da_tr["time"].values).dayofyear, n_harmonics)
    n_coef = X.shape[1]

    # Equações normais por ponto, vetorizadas: A_p = Σ_t w_tp x_t x_tᵀ,
    # b_p = Σ_t w_tp y_tp x_t, com w_tp = 1 onde y é finito.
    valid = np.isfinite(y)
    xx = (X[:, :, None] * X[:, None, :]).reshape(len(X), n_coef * n_coef)
    A = (valid.T.astype(float) @ xx).reshape(-1, n_coef, n_coef)
    b = np.where(valid, y, 0.0).T @ X
    ok = valid.sum(axis=0) >= max(min_samples, n_coef)

    coef = np.full((y.shape[1], n_coef), np.nan)
    if ok.any():
        coef[ok] = (np.linalg.pinv(A[ok]) @ b[ok][:, :, None])[:, :, 0]

    doy_grid = np.arange(1, 367)
    clim = (_harmonic_design(doy_grid, n_harmonics) @ coef.T).reshape(366, *point_shape)

    coords = {"dayofyear": doy_grid}
    for name, c in da_tr.coords.items():
        if name != "time" and set(c.dims) <= set(point_dims):
            coords[name] = c
    return xr.DataArray(clim, dims=("dayofyear", *point_dims), coords=coords, name=target_var)
