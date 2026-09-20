"""ERA5BasinLoader com `interp_method` (src/data/era5_basin_loader.py) e nome do
cache do merge por método (NetCDFLoader.merged_cache_name)."""
import numpy as np
import pandas as pd
import xarray as xr

from src.data.era5_basin_loader import ERA5BasinLoader
from src.data.netcdf_loader import NetCDFLoader

LATS = np.array([-24.0, -24.25, -24.5, -24.75])  # decrescente, como no arquivo real
LONS = np.array([-51.0, -50.75, -50.5, -50.25])
TIMES = pd.date_range("2020-01-01", periods=4, freq="D")
ST_LATS = np.array([-24.1, -24.6])
ST_LONS = np.array([-50.9, -50.4])


def _linear(lat, lon):
    return 1.0 + 2.0 * lat + 3.0 * lon


def _write_basin(raw_dir):
    lat2, lon2 = np.meshgrid(LATS, LONS, indexing="ij")
    data = np.stack([_linear(lat2, lon2) + t for t in range(len(TIMES))])
    xr.Dataset(
        {
            "ws_max": (("time", "latitude", "longitude"), data),
            "ws_mean": (("time", "latitude", "longitude"), data / 2),
        },
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    ).to_netcdf(raw_dir / ERA5BasinLoader.FILENAME)


def _inmet():
    ids = [f"A{i}" for i in range(len(ST_LATS))]
    return xr.Dataset(
        {"daily_wind_gust_max": (("time", "estacao"), np.zeros((len(TIMES), len(ids))))},
        coords={
            "time": TIMES,
            "estacao": ids,
            "latitude": ("estacao", ST_LATS),
            "longitude": ("estacao", ST_LONS),
        },
    )


def test_bilinear_values_and_separate_batch_cache(tmp_path):
    _write_basin(tmp_path)
    out = ERA5BasinLoader(str(tmp_path), interp_method="bilinear").load(_inmet())
    expected = _linear(ST_LATS, ST_LONS)
    np.testing.assert_allclose(out["ws_max_basin"].isel(time=0).values, expected, rtol=1e-10)
    np.testing.assert_allclose(out["ws_max_basin"].isel(time=2).values, expected + 2, rtol=1e-10)
    assert (tmp_path / "_era5_basin_batches_bilinear").is_dir()
    assert not (tmp_path / "_era5_basin_batches").exists()


def test_nearest_default_unchanged(tmp_path):
    _write_basin(tmp_path)
    out = ERA5BasinLoader(str(tmp_path)).load(_inmet())
    # nós mais próximos: (-24.0, -51.0) e (-24.5, -50.5)
    expected = _linear(np.array([-24.0, -24.5]), np.array([-51.0, -50.5]))
    np.testing.assert_allclose(out["ws_max_basin"].isel(time=0).values, expected)
    assert (tmp_path / "_era5_basin_batches").is_dir()


def test_merged_cache_name_per_method():
    assert NetCDFLoader.merged_cache_name("nearest") == "era5_merged_cache.nc"
    assert NetCDFLoader.merged_cache_name("bilinear") == "era5_merged_cache_bilinear.nc"
