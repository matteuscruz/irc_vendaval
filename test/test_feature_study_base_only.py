"""BASE na hora do pico (`src/feature_study/base_only.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.data import base_only as bo


def _nc(tmp_path):
    t = pd.date_range("2020-01-01", periods=48, freq="h")
    # ws_h[hora, estação] = 100*estação + índice da hora: identifica célula e instante
    v = {n: (("time", "estacao"), np.add.outer(np.arange(48.0), 100 * np.arange(2))) for n in bo.BASE_VARS}
    xr.Dataset(v, coords={"time": t, "estacao": ["A1", "A2"]}).to_netcdf(tmp_path / bo.BASE_FILE)


def test_each_variable_is_read_at_the_peak_hour_of_its_own_station_and_day(tmp_path):
    """Hora errada ou estação trocada dariam um modelo plausível e errado."""
    _nc(tmp_path)
    keys = pd.DataFrame({"estacao": ["A1", "A2", "A1"],
                         "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
                         "hora_pico_utc": [5, 3, 23]})
    r = bo.base_at_peak(tmp_path, keys)
    assert r["base_ws_h"].tolist() == [5.0, 100 + 27.0, 47.0]


def test_missing_instant_or_station_stays_nan_instead_of_being_filled(tmp_path):
    _nc(tmp_path)
    keys = pd.DataFrame({"estacao": ["A9", "A1"], "time": pd.to_datetime(["2020-01-01", "2030-01-01"]),
                         "hora_pico_utc": [0, 0]})
    assert bo.base_at_peak(tmp_path, keys)["base_ws_h"].isna().all()


def test_base_columns_are_the_twelve_old_variables_with_no_coordinates():
    cols = bo.base_columns()
    assert len(cols) == 12 and not {"base_latitude", "base_longitude"} & set(cols)
