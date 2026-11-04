"""LSTM nos desenhos regionais (`diagnostics/regional_lstm.py`). Pequeno e rápido: poucas épocas, dado sintético."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src.feature_study.diagnostics import regional as rg  # noqa: E402
from src.feature_study.diagnostics import regional_lstm as rl  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402


def _dados(n=480, T=6, F=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, F)).astype("float32")
    y = 10 + 2 * X[:, -1, 0] + r.normal(0, 0.3, n)
    base = pd.DataFrame({"row_id": np.arange(n), "season": "DJF", "cluster_id": np.repeat([1, 2], n // 2),
                         "_split": np.tile(["train"] * 6 + ["val"] * 2 + ["test"] * 2, n // 10),
                         "janela_ok": True, TARGET_VAR: y})
    return X, base


def test_lstm_trains_only_on_the_design_clusters_and_predicts_with_eqm():
    X, base = _dados()
    X[0, 2, 1] = np.nan                                          # janela com lacuna: fica sem previsão
    fit = rl.fit_season(X, base, (1,), "DJF", epochs=2, patience=1, units=8)
    assert fit["n_train"] == ((base.cluster_id == 1) & (base._split == "train")).sum()
    idx = np.flatnonzero((base._split == "test").to_numpy())
    b, m = rl.predict_rows(fit, X, idx)
    assert np.isfinite(b).all() and np.isfinite(m).all()
    b0, _ = rl.predict_rows(fit, X, np.array([0, 1]))
    assert np.isnan(b0[0]) and np.isfinite(b0[1])


def test_designs_cover_local_regional_global_and_loco():
    d = {x["chave"]: x for x in rg.designs()}
    assert {"B1", "B12", "C", "G", "L12"} <= set(d)
    assert d["C"]["train"] == rg.RS_SC and d["G"]["train"] == rg.GLOBAL
    assert 12 not in d["L12"]["train"] and d["L12"]["avalia"] == (12,)
    assert len(d) == 13 and d["GA"].get("anual")


def test_annual_lstm_uses_all_seasons_and_the_day_of_year():
    X, base = _dados()
    base["season"] = np.tile(["DJF", "MAM", "JJA", "SON"], len(base) // 4)
    base["time"] = pd.Timestamp("2010-01-01") + pd.to_timedelta(np.arange(len(base)), unit="D")
    extra = rl.doy_features(base)
    fit = rl.fit_season(X, base, (1, 2), None, extra=extra, epochs=2, patience=1, units=8)
    assert fit["n_train"] == (base._split == "train").sum()
    b, _ = rl.predict_rows(fit, X, np.arange(10), extra=extra)
    assert np.isfinite(b).all()


def test_saved_lstm_reloads_with_the_same_predictions(tmp_path):
    X, base = _dados()
    fit = rl.fit_season(X, base, (1, 2), "DJF", epochs=1, patience=1, units=4)
    rl.save_fit(fit, tmp_path / "m" / "B1__DJF")
    back = rl.load_fit(tmp_path / "m" / "B1__DJF")
    idx = np.arange(20)
    np.testing.assert_allclose(rl.predict_rows(fit, X, idx)[1], rl.predict_rows(back, X, idx)[1], rtol=1e-5)
