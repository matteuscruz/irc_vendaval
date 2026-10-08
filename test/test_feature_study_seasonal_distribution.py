"""Distribuição sazonal por método (`diagnostics/seasonal_distribution.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.diagnostics import seasonal_distribution as sd
from test.test_feature_study_predictor_curves import _estudo

CH = {s: "CatBoostRegressor" for s in sd.SEASONS}


def _com_dias(tmp_path):
    """O estudo mínimo não tem `estacao`/`time`; acrescenta: 4 estações por dia."""
    ml = _estudo(tmp_path, "ml", "CatBoostRegressor", 0.0)
    ls = _estudo(tmp_path, "lstm", "LSTM", 0.0)
    for d in (ml, ls):
        t = pd.read_parquet(d / "data/test.parquet")
        t["estacao"] = ["A", "B", "C", "D"] * (len(t) // 4)
        t["time"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(t)) // 4, unit="D")
        t.to_parquet(d / "data/test.parquet", index=False)
    return ml, ls


def test_each_label_is_read_from_its_own_arm_and_turned_back_into_a_prediction(tmp_path):
    ml, ls = _com_dias(tmp_path)
    df = sd.load_series(ml, ls, CH, {"ML, base": "base", "ML, sel": "sel__val12"}, {"LSTM, base": "base"})
    assert {"ML, base", "ML, sel", "LSTM, base", sd.INMET, sd.ERA5} <= set(df.columns)
    assert len(df) == 800 and df[["ML, base", "ML, sel", "LSTM, base"]].notna().all().all()
    # resíduo de média ~0: a previsão fica perto do observado
    assert abs((df["ML, base"] - df[sd.INMET]).mean()) < 0.3


def test_a_missing_arm_fails_loudly(tmp_path):
    ml, ls = _com_dias(tmp_path)
    with pytest.raises(FileNotFoundError, match="não foi ajustado"):
        sd.load_series(ml, ls, CH, {"ML, grupo 2": "add_grp__grupo2"}, {})


def test_daily_mean_averages_the_stations_present_each_day(tmp_path):
    ml, ls = _com_dias(tmp_path)
    df = sd.load_series(ml, ls, CH, {"ML, base": "base"}, {})
    d = sd.daily_station_mean(df, [sd.INMET, "ML, base"])
    assert len(d) == 800 // 4 and set(d["season"]) == set(sd.SEASONS)
    um = df[(df.season == "DJF")].groupby("time")[sd.INMET].mean()
    got = d[d.season == "DJF"].set_index("time")[sd.INMET]
    np.testing.assert_allclose(got.loc[um.index], um)


def test_summary_has_one_row_per_scope_and_method(tmp_path):
    ml, ls = _com_dias(tmp_path)
    df = sd.load_series(ml, ls, CH, {"ML, base": "base"}, {})
    d = sd.daily_station_mean(df, [sd.INMET, "ML, base"])
    r = sd.summary(d, [sd.INMET, "ML, base"])
    assert len(r) == 5 * 2 and set(r["escopo"]) == {"Todos", *sd.SEASONS}
    assert (r["maximo"] >= r["p90"]).all() and (r["p90"] >= r["mediana"]).all()
