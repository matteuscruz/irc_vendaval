"""Base de treino por estação (`data/station_dataset.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.data import station_dataset as sdb
from src.pipelines.common import TARGET_VAR


def _dados(tmp_path):
    dias = pd.date_range("2018-01-01", "2025-06-30", freq="D")
    est = ["E1", "E2"]
    r = np.random.default_rng(0)
    xr.Dataset({TARGET_VAR: (("time", "estacao"), r.uniform(5, 20, (len(dias), 2)))},
               coords={"time": dias, "estacao": est, "latitude": ("estacao", [-30.0, -30.5]),
                       "longitude": ("estacao", [-51.0, -51.5])}).to_netcdf(tmp_path / "inmet.nc")
    linhas = [{"lat": la, "lon": lo, "time": d, "hora_pico": d + pd.Timedelta(hours=15), "season": "DJF",
               "gust10fg": 10.0, "w10": 5.0}
              for la, lo in [(-30.0, -51.0), (-30.5, -51.5)] for d in dias]
    st = pd.DataFrame({"estacao": est, "cluster_id": [1, 2], "grid_lat": [-30.0, -30.5], "grid_lon": [-51.0, -51.5]})
    return pd.DataFrame(linhas), st, tmp_path / "inmet.nc"


def test_split_is_the_study_split_and_2025_stays_out(tmp_path):
    grade, st, inm = _dados(tmp_path)
    b = sdb.build(grade, st, inm, ["gust10fg", "w10"])
    h = b[b.time.dt.year <= 2024]
    assert set(h[h._split == "test"].month) == {1, 4, 7, 10}
    assert not set(h[h._split.isin(["train", "val"])].month) & {1, 4, 7, 10}
    assert (h._split == "val").any()
    assert (b[b.time.dt.year == 2025]._split == sdb.NOVO).all()
    assert set(b.cluster_id) == {1, 2} and np.allclose(b.era5_gust_max, 10.0)


def test_split_is_calendar_only_so_both_stations_share_it(tmp_path):
    grade, st, inm = _dados(tmp_path)
    b = sdb.build(grade, st, inm, ["gust10fg", "w10"])
    p = b.pivot_table(index="time", columns="estacao", values="_split", aggfunc="first").dropna()
    assert (p["E1"] == p["E2"]).all()


def test_rows_with_missing_feature_are_dropped(tmp_path):
    grade, st, inm = _dados(tmp_path)
    grade.loc[grade.index[:10], "w10"] = np.nan
    b = sdb.build(grade, st, inm, ["gust10fg", "w10"])
    assert b.w10.notna().all()
