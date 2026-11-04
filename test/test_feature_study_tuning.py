"""Busca de hiperparâmetros mirando extremos (`diagnostics/tuning.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_study.diagnostics import tuning as tn


def _dados(n=600, seed=0):
    r = np.random.default_rng(seed)
    x = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n)})
    y = 10 + 2 * x.a + np.exp(x.b) + r.normal(scale=0.5, size=n)
    return x, y.to_numpy()


def _pipe(x, y):
    return Pipeline([("preprocessor", StandardScaler()),
                     ("regressor", HistGradientBoostingRegressor(random_state=0))]).fit(x, y)


def test_tail_metrics_detect_a_model_that_cuts_the_peaks():
    y = np.r_[np.linspace(5, 15, 95), np.full(5, 30.0)]
    corta = np.where(y > 20, 15.0, y)
    m = tn.tail_metrics(y, corta, lim95=20, lim99=25)
    assert m["POD_P95"] == 0 and m["vies_P99"] < -10 and m["RMSE_P90"] > m["RMSE"]
    assert tn.tail_metrics(y, y, 20, 25)["POD_P99"] == 1


def test_fit_variant_changes_only_the_model_step_and_keeps_the_original_untouched():
    x, y = _dados()
    base = _pipe(x, y)
    p = tn.fit_variant(base, x, y, {"loss": "quantile", "quantile": 0.9}, peso_cauda=3.0, lim_cauda=np.quantile(y, 0.9))
    assert p.named_steps["regressor"].loss == "quantile"
    assert base.named_steps["regressor"].loss == "squared_error"
    # quantil 0,9 com peso na cauda prevê mais alto que o padrão
    assert p.predict(x).mean() > base.predict(x).mean()


def test_tune_runs_and_returns_params_inside_the_search_space():
    x, y = _dados()
    base = _pipe(x[:400], y[:400])
    study, params, peso, lim = tn.tune(base, "HistGradientBoostingRegressor", x[:400], y[:400], x[400:], y[400:],
                                       n_trials=3)
    assert len(study.trials) == 3 and 1.0 <= peso <= 5.0
    assert params["loss"] in ("squared_error", "quantile") and 15 <= params["max_leaf_nodes"] <= 127


def test_wide_search_also_tunes_the_tail_threshold_and_a_heavier_weight():
    x, y = _dados()
    base = _pipe(x[:400], y[:400])
    study, params, peso, lim = tn.tune(base, "HistGradientBoostingRegressor", x[:400], y[:400], x[400:], y[400:],
                                       n_trials=3, amplo=True)
    assert 1.0 <= peso <= 10.0 and "max_features" in params
    assert lim in {float(np.quantile(y[:400], q)) for q in (0.8, 0.9, 0.95)}
