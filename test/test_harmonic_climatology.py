"""Climatologia harmônica (src/data/climatology.py::get_harmonic_climatology)."""
import numpy as np
import pandas as pd
import xarray as xr

from src.data.climatology import get_harmonic_climatology
from src.pipeline.data.splits import MonthBlockSplit

DAYS = pd.date_range("2010-01-01", "2019-12-31", freq="D")


def _dataset(values_fn):
    phase = 2 * np.pi * (DAYS.dayofyear.values - 1) / 365.25
    base = values_fn(phase)
    return xr.Dataset(
        {"v": (("time", "estacao"), np.stack([base, base + 1.0], axis=1))},
        coords={
            "time": DAYS,
            "estacao": ["A", "B"],
            "latitude": ("estacao", [-25.0, -26.0]),
        },
    )


def _train_times():
    return DAYS[MonthBlockSplit().label(DAYS) == "train"]


def test_every_day_of_year_defined_even_for_test_months():
    clim = get_harmonic_climatology(
        _dataset(lambda p: 5 + 2 * np.cos(p) + np.sin(2 * p)), "v", _train_times(),
    )
    assert clim.dims == ("dayofyear", "estacao")
    assert clim.sizes["dayofyear"] == 366
    assert np.isfinite(clim.values).all()
    assert "latitude" in clim.coords


def test_recovers_smooth_seasonal_cycle():
    clim = get_harmonic_climatology(
        _dataset(lambda p: 5 + 2 * np.cos(p) + np.sin(2 * p)), "v", _train_times(),
    )
    phase = 2 * np.pi * (np.arange(1, 367) - 1) / 365.25
    expected = 5 + 2 * np.cos(phase) + np.sin(2 * phase)
    np.testing.assert_allclose(clim.sel(estacao="A").values, expected, atol=1e-6)
    np.testing.assert_allclose(clim.sel(estacao="B").values, expected + 1, atol=1e-6)


def test_test_month_values_do_not_influence_fit():
    ds = _dataset(lambda p: 5 + 2 * np.cos(p))
    clim_a = get_harmonic_climatology(ds, "v", _train_times())
    ds_b = ds.copy(deep=True)
    ds_b["v"].values[MonthBlockSplit().label(DAYS) == "test"] = 999.0
    clim_b = get_harmonic_climatology(ds_b, "v", _train_times())
    np.testing.assert_allclose(clim_a.values, clim_b.values)


def test_point_with_too_few_samples_is_nan():
    ds = _dataset(lambda p: 5 + np.cos(p))
    ds["v"].values[:, 1] = np.nan
    ds["v"].values[:10, 1] = 1.0
    clim = get_harmonic_climatology(ds, "v", _train_times(), min_samples=30)
    assert np.isnan(clim.sel(estacao="B").values).all()
    assert np.isfinite(clim.sel(estacao="A").values).all()
