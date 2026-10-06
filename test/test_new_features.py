"""Features ERA5 novas (src/data/new_features.py) e a regra sem imputação das
pipelines (src/pipelines/common.py::select_complete_rows)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.new_features import CACHE_FILENAME, load_new_features_grid
from src.pipelines.common import resolve_feature_groups, select_complete_rows


def test_grid_daily_utc_aggregation_and_incomplete_variable_dropped(synthetic_nf_raw_dir):
    grid = load_new_features_grid(synthetic_nf_raw_dir)
    names = set(grid.data_vars)

    assert {"nf_ws10_max", "nf_ws10_mean", "nf_ws10_std", "nf_cape_max", "nf_cape_mean", "nf_sdor"} <= names
    assert not any("ws100" in n or "u100" in n for n in names)  # u100 só no último ano
    assert not {"nf_u10_mean", "nf_v10_mean"} & names          # par u/v vira magnitude
    assert grid["nf_sdor"].dims == ("latitude", "longitude")

    region = synthetic_nf_raw_dir / "new_features" / "cluster9" / "sl"
    u = xr.open_dataset(region / "u10_2017.nc")["u10"].isel(latitude=1, longitude=2)
    v = xr.open_dataset(region / "v10_2017.nc")["v10"].isel(latitude=1, longitude=2)
    ws = np.hypot(u, v).to_series()
    day = pd.Timestamp("2017-03-10")
    expected = ws[(ws.index >= day) & (ws.index < day + pd.Timedelta(days=1))].max()
    got = grid["nf_ws10_max"].isel(latitude=1, longitude=2).sel(time=day)
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_grid_cache_is_rebuilt_when_files_change(synthetic_nf_raw_dir):
    load_new_features_grid(synthetic_nf_raw_dir)
    cache = synthetic_nf_raw_dir / "new_features" / CACHE_FILENAME
    first = xr.load_dataset(cache).attrs["fingerprint"]

    static = synthetic_nf_raw_dir / "new_features" / "cluster9" / "static"
    ds = xr.load_dataset(static / "sdor.nc")
    ds.rename({"sdor": "slor"}).to_netcdf(static / "slor.nc")

    grid = load_new_features_grid(synthetic_nf_raw_dir)
    assert "nf_slor" in grid
    assert xr.load_dataset(cache).attrs["fingerprint"] != first


def test_grid_outside_basin_grid_is_rejected(synthetic_nf_raw_dir):
    path = synthetic_nf_raw_dir / "new_features" / "cluster9" / "static" / "sdor.nc"
    ds = xr.load_dataset(path)
    ds.assign_coords(latitude=ds.latitude + 0.1).to_netcdf(path)
    with pytest.raises(ValueError, match="mesma dimensão espacial"):
        load_new_features_grid(synthetic_nf_raw_dir, use_cache=False)


def test_basin_mask_applied_to_new_features(synthetic_nf_raw_dir):
    basin_path = synthetic_nf_raw_dir / "ERA5_Features_Basin_2000_2026.nc"
    basin = xr.load_dataset(basin_path)
    for v in basin.data_vars:  # máscara da bacia vale para todas as variáveis
        basin[v].loc[dict(latitude=-23.0, longitude=-45.0)] = np.nan
    basin.to_netcdf(basin_path)

    grid = load_new_features_grid(synthetic_nf_raw_dir, use_cache=False)
    assert grid["nf_cape_max"].sel(latitude=-23.0, longitude=-45.0).isnull().all()
    assert grid["nf_sdor"].sel(latitude=-23.0, longitude=-45.0).isnull()
    assert grid["nf_cape_max"].sel(latitude=-21.0, longitude=-45.0).notnull().all()


def _frame():
    return pd.DataFrame({
        "estacao": ["A", "A", "B", "B", "C", "C"],
        "cluster_id": [1, 1, 1, 1, 2, 2],
        "x": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0],
        "nf_cape_max": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
    })


def test_select_complete_rows_keeps_only_fully_covered_clusters():
    out = select_complete_rows(_frame(), ["x", "nf_cape_max"])
    assert set(out["cluster_id"]) == {1}
    assert len(out) == 3  # linha A com x NaN descartada


def test_select_complete_rows_drops_partially_covered_cluster():
    df = _frame()
    df.loc[df["estacao"] == "B", "nf_cape_max"] = np.nan
    with pytest.raises(ValueError, match="nenhum cluster 100% coberto"):
        select_complete_rows(df, ["x", "nf_cape_max"])


def test_select_complete_rows_without_new_features_only_drops_nan_rows():
    out = select_complete_rows(_frame(), ["x"])
    assert set(out["cluster_id"]) == {1, 2} and len(out) == 5


def test_select_complete_rows_rejects_missing_column():
    with pytest.raises(ValueError, match="ausentes"):
        select_complete_rows(_frame(), ["x", "ws_p90_basin"])


def test_resolve_new_features_group():
    cols = ["wind_mag_max", "nf_ws10_max", "nf_cape_max"]
    feats = resolve_feature_groups("original,new_features", cols)
    assert feats[-2:] == ["nf_cape_max", "nf_ws10_max"]
    with pytest.raises(ValueError, match="nenhuma feature correspondente"):
        resolve_feature_groups("original,new_features", ["wind_mag_max"])
    with pytest.raises(ValueError, match="colunas do DataFrame"):
        resolve_feature_groups("new_features")
    with pytest.raises(ValueError, match="inválido"):
        resolve_feature_groups("original,bt55")
    assert resolve_feature_groups(None) == resolve_feature_groups("original")


def test_resolve_new_features_static_and_dynamic_subgroups():
    cols = ["wind_mag_max", "nf_ws10_max", "nf_cape_max", "nf_sdor", "nf_orog_height"]
    static = resolve_feature_groups("new_features_static", cols)
    dynamic = resolve_feature_groups("new_features_dynamic", cols)

    assert static == ["nf_orog_height", "nf_sdor"]
    assert dynamic == ["nf_cape_max", "nf_ws10_max"]
    assert set(static) & set(dynamic) == set()
    assert set(static) | set(dynamic) == set(resolve_feature_groups("new_features", cols)) - {"wind_mag_max"}

    # Só as estáticas cobertas entram — pedir new_features_static sem nenhuma
    # coluna nf_ estática carregada é erro, mesmo com dinâmicas presentes.
    with pytest.raises(ValueError, match="nenhuma feature correspondente"):
        resolve_feature_groups("new_features_static", ["nf_ws10_max"])


def test_lstm_daily_batch_with_new_features_trains_only_covered_cluster(
    synthetic_nf_raw_dir, synthetic_shp_dir,
):
    from src.pipeline.data.cluster_preprocessor import ClusterPreprocessor

    data = ClusterPreprocessor(
        str(synthetic_nf_raw_dir), str(synthetic_shp_dir), lookback=3,
        feature_groups="original,new_features", seed=0,
    ).run()
    assert data.cluster_ids == [9]
    assert "nf_ws10_max" in data.feature_names and "nf_sdor" in data.feature_names
    for split in ("x_train", "x_val", "x_test"):
        for arr in getattr(data, split).values():
            assert np.isfinite(arr).all()
