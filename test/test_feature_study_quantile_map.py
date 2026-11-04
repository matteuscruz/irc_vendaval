"""Mapeamento de quantis (`diagnostics/quantile_map.py`)."""
from __future__ import annotations

import numpy as np

from src.feature_study.diagnostics import quantile_map as qm


def test_mapping_restores_the_observed_spread_and_keeps_the_order():
    r = np.random.default_rng(0)
    obs = r.gamma(6, 1.6, 5000)                       # assimétrica, como rajada
    pred = obs.mean() + 0.6 * (obs - obs.mean()) + r.normal(0, 0.5, 5000)   # previsão comprimida
    m = qm.apply(pred, qm.fit(pred, obs))
    assert abs(m.std() - obs.std()) < 0.1 * obs.std() < abs(pred.std() - obs.std())
    assert abs(np.quantile(m, 0.99) - np.quantile(obs, 0.99)) < 0.5
    assert np.corrcoef(m, obs)[0, 1] >= np.corrcoef(pred, obs)[0, 1] - 1e-3
    o = np.argsort(pred)
    assert np.all(np.diff(m[o]) >= -1e-9)             # monotônico


def test_values_outside_the_fitted_range_shift_by_the_edge_offset_and_nan_stays_nan():
    xq, yq = np.array([5.0, 10.0, 15.0]), np.array([4.0, 10.0, 18.0])
    out = qm.apply(np.array([2.0, 10.0, 20.0, np.nan]), (xq, yq))
    np.testing.assert_allclose(out[:3], [1.0, 10.0, 23.0])
    assert np.isnan(out[3])
