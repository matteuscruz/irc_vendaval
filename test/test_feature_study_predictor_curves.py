"""Calibração e excedência dos cinco previsores (`diagnostics/predictor_curves.py`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_study.diagnostics import predictor_curves as pc


def _estudo(tmp_path, nome, modelo, deslocamento):
    """Estudo mínimo: 4 trimestres, 200 linhas cada; a previsão = observado + deslocamento (resid = deslocamento)."""
    rng = np.random.default_rng(0)
    d = tmp_path / nome
    (d / "data").mkdir(parents=True)
    linhas = []
    for i, s in enumerate(pc.SEASONS):
        linhas.append(pd.DataFrame({"row_id": np.arange(200) + 1000 * i, "season": s,
                                    "daily_wind_gust_max": rng.gamma(4, 2, 200)}))
    t = pd.concat(linhas, ignore_index=True)
    t["era5_gust_max"] = t["daily_wind_gust_max"] + rng.normal(0, 3, len(t))
    t.to_parquet(d / "data/test.parquet", index=False)
    tag = "full" if nome == "ml" else "full_lstm"
    for s in pc.SEASONS:
        for arm in ("base", "sel__val12"):
            p = d / "units" / tag
            p.mkdir(parents=True, exist_ok=True)
            n = (t["season"] == s).sum()
            ruido = rng.normal(0, 1.0 if arm == "base" else 0.5, n) + deslocamento
            pd.DataFrame({"row_id": t.loc[t.season == s, "row_id"].to_numpy(), modelo: ruido.astype("float32")}
                         ).to_parquet(p / f"resid__{s}__{arm}.parquet", index=False)
    return d


def test_the_five_predictors_share_the_same_rows_and_have_the_observed_next_to_them(tmp_path):
    ml = _estudo(tmp_path, "ml", "CatBoostRegressor", 0.0)
    ls = _estudo(tmp_path, "lstm", "LSTM", 0.0)
    df = pc.load_predictors(ml, ls, {s: "CatBoostRegressor" for s in pc.SEASONS})
    assert set(pc.PREVISORES) | {"row_id", "season", "y"} == set(df.columns)
    assert len(df) == 800 and df[list(pc.PREVISORES)].notna().all().all()
    # previsão = resíduo + observado: com resíduo de média 0, a previsão fica perto do observado
    assert abs((df["ML, base"] - df["y"]).mean()) < 0.2


def test_different_targets_between_the_two_studies_are_refused(tmp_path):
    ml = _estudo(tmp_path, "ml", "CatBoostRegressor", 0.0)
    ls = _estudo(tmp_path, "lstm", "LSTM", 0.0)
    t = pd.read_parquet(ls / "data/test.parquet")
    t["daily_wind_gust_max"] += 1.0
    t.to_parquet(ls / "data/test.parquet", index=False)
    with pytest.raises(ValueError, match="populações diferentes"):
        pc.load_predictors(ml, ls, {s: "CatBoostRegressor" for s in pc.SEASONS})


def test_a_missing_arm_fails_loudly_instead_of_dropping_the_predictor(tmp_path):
    ml = _estudo(tmp_path, "ml", "CatBoostRegressor", 0.0)
    ls = _estudo(tmp_path, "lstm", "LSTM", 0.0)
    (ls / "units/full_lstm/resid__DJF__sel__val12.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="ainda não foi ajustado"):
        pc.load_predictors(ml, ls, {s: "CatBoostRegressor" for s in pc.SEASONS})


def test_curves_cover_every_scope_and_predictor_and_a_biased_predictor_is_miscalibrated(tmp_path):
    ml = _estudo(tmp_path, "ml", "CatBoostRegressor", 0.0)
    ls = _estudo(tmp_path, "lstm", "LSTM", 3.0)                      # a LSTM superestima em 3 m/s
    df = pc.load_predictors(ml, ls, {s: "CatBoostRegressor" for s in pc.SEASONS})
    cal, exc = pc.curves(df)
    assert set(cal["escopo"]) == {pc.TODOS, *pc.SEASONS} and set(cal["previsor"]) == set(pc.PREVISORES)
    assert set(exc["limiar"]) == {"P90", "P95", "P99"} and len(exc) == 5 * 5 * 3
    c = cal[(cal.escopo == pc.TODOS)].assign(desvio=lambda d: d.obs_media - d.prev_media).groupby("previsor").desvio.mean()
    assert c["LSTM, base"] < -2.0 and abs(c["ML, base"]) < 0.5       # quem superestima tem observado < previsto
    pod = exc[(exc.escopo == pc.TODOS) & (exc.limiar == "P90")].set_index("previsor").pod
    assert pod["LSTM, base"] > pod["ML, base"]                        # superestimar acusa mais extremos (e mais falsos)


def test_champions_are_read_from_the_frozen_top_models(tmp_path):
    (tmp_path / "top_models.json").write_text(json.dumps({"by_season": {"DJF": ["A", "B"], "MAM": ["C"]}}))
    assert pc.champions(tmp_path) == {"DJF": "A", "MAM": "C"}
