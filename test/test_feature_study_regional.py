"""Treino e avaliação por cluster (`diagnostics/regional.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.diagnostics import regional as rg
from src.pipelines.common import TARGET_VAR


def _base(n=1600, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(10, 3, n)
    d = pd.DataFrame({"gust10fg": x, "w10": x * 0.5 + r.normal(0, 1, n),
                      "season": np.tile(rg.SEASONS, n // 4), "cluster_id": np.repeat([1, 2], n // 2),
                      "estacao": np.tile(["A", "B", "C", "D"], n // 4),
                      "_split": np.tile(["train"] * 6 + ["val"] * 2 + ["test"] * 2, n // 10)[:n]})
    d[TARGET_VAR] = 0.8 * x + 2 + r.normal(0, 1, n)
    d["era5_gust_max"] = d.gust10fg
    d["row_id"] = np.arange(n)
    return d


CH = {s: "LinearRegression" for s in rg.SEASONS}


def test_fit_set_uses_train_and_fits_eqm_on_val():
    d = _base()
    f = rg.fit_set(d, ["gust10fg", "w10"], CH)
    assert set(f) == set(rg.SEASONS)
    for s, v in f.items():
        assert v["n_train"] == ((d.season == s) & (d._split == "train")).sum()
        assert v["n_val"] == ((d.season == s) & (d._split == "val")).sum()
        assert len(v["mapa"][0]) == 199


def test_loco_training_never_sees_the_held_out_cluster():
    d = _base()
    f = rg.fit_set(d[d.cluster_id != 2], ["gust10fg", "w10"], CH)
    assert sum(v["n_train"] for v in f.values()) == ((d.cluster_id != 2) & (d._split == "train")).sum()


def test_evaluate_reports_only_test_rows_per_cluster_and_version():
    d = _base()
    f = rg.fit_set(d, ["gust10fg", "w10"], CH)
    b, m = rg.predict(d, f, ["gust10fg", "w10"])
    mt = rg.evaluate(d, {"modelo": b, "Corrigido ML": m}, rotulo={"desenho": "x"})
    assert set(mt.versao) == {"ERA5", "modelo", "Corrigido ML"} and set(mt.cluster) == {1, 2}
    tot = mt[(mt.trimestre == "Todos") & (mt.versao == "ERA5")]
    assert tot.n.sum() == (d._split == "test").sum()
    assert (mt.desenho == "x").all()


def test_single_annual_model_covers_all_seasons_with_one_eqm():
    d = _base()
    d["time"] = pd.Timestamp("2010-01-01") + pd.to_timedelta(np.arange(len(d)) % 365, unit="D")
    d2 = rg.with_doy(d)
    cols = ["gust10fg", "w10", *rg.DOY_COLS]
    f = rg.fit_single(d2, cols, model="LinearRegression")
    assert list(f) == [rg.ANUAL] and f[rg.ANUAL]["n_train"] == (d._split == "train").sum()
    b, m = rg.predict(d2.assign(season=rg.ANUAL), f, cols)
    assert np.isfinite(b).all() and np.isfinite(m).all()
    assert {"GA"} <= {x["chave"] for x in rg.designs() if x.get("anual")}
