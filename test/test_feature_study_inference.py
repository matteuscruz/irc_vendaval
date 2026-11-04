"""Inferência de sanidade na grade (`diagnostics/inference.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from src.feature_study.diagnostics import inference as inf


def _grade(tmp_path, row_group_size):
    """2 células × 3 dias × 24 h; o pico de cada dia é a hora `dia + 5` com valor conhecido."""
    linhas = []
    for lat, lon in [(-30.0, -51.25), (-30.25, -51.0)]:
        for h in pd.date_range("2025-01-01", periods=72, freq="h"):
            pico = h.hour == h.day + 5
            linhas.append({"lat": lat, "lon": lon, "time": h, "gust10fg": 20.0 + h.day if pico else 5.0,
                           "w10": float(h.hour)})
    df = pd.DataFrame(linhas)
    p = tmp_path / "g.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, row_group_size=row_group_size)
    return p, df


def test_peak_row_is_the_hour_of_max_gust_even_across_row_groups(tmp_path):
    # grupos de 50 linhas partem dias e células no meio
    p, _ = _grade(tmp_path, 50)
    d = inf.daily_peak_rows(p, ["w10"])
    assert len(d) == 6
    assert (d["gust10fg"] == 20.0 + d["time"].dt.day).all()
    assert (d["hora_pico"].dt.hour == d["time"].dt.day + 5).all()
    assert (d["w10"] == d["hora_pico"].dt.hour).all()
    assert set(d["season"]) == {"DJF"}


def test_day_with_a_missing_gust_hour_is_dropped(tmp_path):
    p, df = _grade(tmp_path, 1000)
    df.loc[(df.lat == -30.0) & (df.time == "2025-01-02 10:00"), "gust10fg"] = np.nan
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, row_group_size=50)
    d = inf.daily_peak_rows(p, ["w10"])
    assert len(d) == 5
    assert not ((d.lat == -30.0) & (d.time == "2025-01-02")).any()


def test_sample_code_separates_train_test_outside_and_new_year():
    t = pd.Series(pd.to_datetime(["2010-02-01", "2010-01-01", "2010-02-01", "2025-02-01"]))
    est = pd.Series([True, True, False, True])
    assert inf.amostra(t, est).tolist() == [inf.TREINO, inf.TESTE, inf.FORA, inf.NOVO]


def test_netcdf_has_the_grid_dims_and_is_zlib_5(tmp_path):
    p, _ = _grade(tmp_path, 1000)
    d = inf.daily_peak_rows(p, ["w10"])
    d = d.rename(columns={"gust10fg": "era5_gust_max"}).assign(pred_gust_max=lambda x: x.era5_gust_max - 1,
                                                               amostra=inf.NOVO)
    out = inf.write_netcdf(inf.to_dataset(d, {"nota": "x"}), tmp_path / "r.nc")
    ds = xr.open_dataset(out)
    assert dict(ds.sizes) == {"time": 3, "lat": 2, "lon": 2}
    assert {"era5_gust_max", "pred_gust_max", "delta", "hora_pico_utc", "amostra"} <= set(ds.data_vars)
    assert ds["pred_gust_max"].encoding["zlib"] and ds["pred_gust_max"].encoding["complevel"] == 5
    np.testing.assert_allclose(ds["delta"].values[~np.isnan(ds["delta"].values)], -1)


def test_rows_with_a_missing_feature_get_no_prediction():
    class Pipe:
        def predict(self, x):
            return np.full(len(x), 10.0)
    tr = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    g = pd.DataFrame({"a": [1.0, np.nan, 2.0], "season": ["DJF"] * 3})
    p = inf.predict_grid(g, {"DJF": {"pipe": Pipe(), "train": tr}}, ["a"])
    assert np.isnan(p[1]) and (p[[0, 2]] == 10).all()
