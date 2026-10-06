"""Extração grade→ponto nearest/bilinear (src/data/interp.py)."""
import numpy as np
import xarray as xr

from src.data.interp import build_point_indexer, extract_points

LATS = np.arange(-13.75, -15.26, -0.25)  # decrescente, como no ERA5-Basin
LONS = np.arange(-58.0, -56.49, 0.25)
PT_LATS = np.array([-13.9, -14.62, -15.1])
PT_LONS = np.array([-57.93, -57.1, -56.6])


def _field(lat, lon):
    return 2.0 + 3.0 * lat + 5.0 * lon


def _grid():
    lat2, lon2 = np.meshgrid(LATS, LONS, indexing="ij")
    base = _field(lat2, lon2)
    return xr.Dataset(
        {"v": (("time", "latitude", "longitude"), np.stack([base, 2 * base]))},
        coords={"time": [0, 1], "latitude": LATS, "longitude": LONS},
    )


def _extract(ds, lats, lons, method="bilinear"):
    indexer = build_point_indexer(ds.latitude.values, ds.longitude.values, lats, lons, method)
    return extract_points(ds, indexer, "estacao", [f"S{i}" for i in range(len(lats))])


def test_bilinear_exact_on_linear_field_with_descending_latitude():
    out = _extract(_grid(), PT_LATS, PT_LONS)
    assert out["v"].dims == ("time", "estacao")
    np.testing.assert_allclose(out["v"].sel(time=0).values, _field(PT_LATS, PT_LONS), rtol=1e-12)
    np.testing.assert_allclose(out["v"].sel(time=1).values, 2 * _field(PT_LATS, PT_LONS), rtol=1e-12)


def test_nan_corner_falls_back_to_nearest_valid_node():
    ds = _grid()
    ds["v"].values[:, 2, 2] = np.nan  # nó (-14.25, -57.5), canto distante da célula
    out = _extract(ds, np.array([-14.02]), np.array([-57.73]))
    nearest = ds["v"].sel(latitude=-14.0, longitude=-57.75).values
    np.testing.assert_allclose(out["v"].values[:, 0], nearest)


def test_nearest_mode_matches_xarray_sel_nearest():
    ds = _grid()
    out = _extract(ds, PT_LATS, PT_LONS, method="nearest")
    ref = ds.sel(
        latitude=xr.DataArray(PT_LATS, dims="p"),
        longitude=xr.DataArray(PT_LONS, dims="p"),
        method="nearest",
    )
    np.testing.assert_allclose(out["v"].values, ref["v"].values)


def test_points_outside_grid_are_nan():
    out = _extract(_grid(), np.array([-10.0, -14.1]), np.array([-57.0, -57.0]))
    assert np.isnan(out["v"].values[:, 0]).all()
    assert np.isfinite(out["v"].values[:, 1]).all()


def test_subset_matches_full_extraction():
    ds = _grid()
    full = build_point_indexer(LATS, LONS, PT_LATS, PT_LONS)
    part = extract_points(ds, full.subset(slice(1, 3)), "estacao", ["S1", "S2"])
    ref = _extract(ds, PT_LATS, PT_LONS)
    np.testing.assert_allclose(part["v"].values, ref["v"].values[:, 1:3])
