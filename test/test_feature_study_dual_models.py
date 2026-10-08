"""ML campeão e LSTM com todas as features e com as selecionadas (`diagnostics/dual_models.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.diagnostics import dual_models as dm


def _frames(n_tr=300, n_va=80, n_te=100):
    rng = np.random.default_rng(0)

    def f(n, seed):
        r = np.random.default_rng(seed)
        d = pd.DataFrame({"util": r.normal(size=n), "ruido": r.normal(size=n)})
        d["daily_wind_gust_max"] = 8 + 2 * d["util"] + r.normal(0, 0.3, n)
        return d
    return f(n_tr, 1), f(n_va, 2), f(n_te, 3), rng


def test_the_ml_model_learns_from_the_informative_columns_only():
    tr, va, te, _ = _frames()
    _, p_com = dm.ml_predict("Ridge", tr, va, te, ["util", "ruido"])
    _, p_sem = dm.ml_predict("Ridge", tr, va, te, ["ruido"])
    rmse = lambda p: float(np.sqrt(np.mean((p - te["daily_wind_gust_max"]) ** 2)))  # noqa: E731
    assert rmse(p_com) < 0.3 * rmse(p_sem)
    assert p_com.min() >= 0 and p_com.max() <= 80                          # faixa física


def test_selection_rule_keeps_what_beats_the_noise_floor_in_enough_seasons():
    def d(util, ruido, outra):
        return pd.DataFrame({"nome": ["util"] * 2 + ["ctrl_noise"] * 2 + ["outra"] * 2,
                             "d_rmse": [util, util, ruido, ruido, outra, outra]})
    imp = {"DJF": d(0.5, 0.01, 0.2), "MAM": d(0.4, 0.00, 0.2), "JJA": d(0.6, -0.01, -0.1), "SON": d(0.5, 0.0, -0.1)}
    r = dm.select_by_rule(imp)
    assert r.loc["util", "decisão"] == "manter"
    assert r.loc["outra", "decisão"] == "descartar"                         # positiva em só 2 de 4 trimestres
    assert abs(r.attrs["piso"] - 0.01) < 1e-9 and "ctrl_noise" not in r.index


def test_consensus_separates_what_both_models_choose_from_what_only_one_does():
    c = dm.consenso(["a", "b"], ["b", "c"], ["a", "b", "c", "d"]).set_index("variável")["veredito"]
    assert c.to_dict() == {"a": "só ML", "b": "ambos", "c": "só LSTM", "d": "nenhum"}
