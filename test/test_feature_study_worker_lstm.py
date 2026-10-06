"""Worker da LSTM (`src/feature_study/core/worker_lstm.py`).

Dados sintéticos com resposta conhecida: o alvo depende só da feature informativa.
Modelo minúsculo (poucas unidades e épocas) para a suíte ficar rápida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.core import worker_lstm as wl

HP = {"units": 8, "epochs": 30, "batch_size": 64, "patience": 10, "learning_rate": 0.01}


def _data(n, T=3, seed=0, nan_rows=()):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, T, 2)).astype("float32")           # [informativa, ruído]
    y = 8.0 + 2.0 * x[:, -1, 0] + rng.normal(0, 0.2, n)
    for i in nan_rows:
        x[i, 0, 1] = np.nan
    return x, y


def _fit(x_cols, nan_te=()):
    tr, va, te = _data(500, seed=1), _data(120, seed=2), _data(150, seed=3, nan_rows=nan_te)
    sl = np.s_[:, :, x_cols]
    return wl.fit_arm_lstm(tr[0][sl], tr[1], va[0][sl], va[1], te[0][sl], te[1], np.arange(150) + 1000, seed=7, **HP)


def test_output_has_the_same_schema_as_the_tabular_worker():
    metrics, resid, secs = _fit([0, 1])
    assert set(metrics["model"]) == {wl.LSTM_NAME} and set(metrics["split"]) == {"val", "test"}
    assert {"R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90", "n", "n_nonfinite"} <= set(metrics.columns)
    assert list(resid.columns) == ["row_id", wl.LSTM_NAME] and resid["row_id"].iloc[0] == 1000
    assert secs > 0 and np.isfinite(resid[wl.LSTM_NAME]).all()


def test_the_arm_with_the_informative_feature_is_better():
    """A LSTM só vale como segundo olhar se enxergar o sinal quando ele existe."""
    com, _, _ = _fit([0, 1])
    sem, _, _ = _fit([1])
    rmse = lambda m: float(m[m.split == "test"].RMSE.iloc[0])  # noqa: E731
    assert rmse(com) < 0.5 * rmse(sem)


def test_windows_with_a_gap_are_dropped_not_filled():
    """Sem imputação: a linha com hora faltando sai da avaliação, e o resíduo só tem as demais."""
    metrics, resid, _ = _fit([0, 1], nan_te=(0, 5, 9))
    assert len(resid) == 147 and not {1000, 1005, 1009} & set(resid["row_id"])
    assert int(metrics[metrics.split == "test"].n.iloc[0]) == 147


def test_the_same_seed_gives_the_same_prediction():
    a, ra, _ = _fit([0, 1])
    b, rb, _ = _fit([0, 1])
    np.testing.assert_allclose(ra[wl.LSTM_NAME], rb[wl.LSTM_NAME], atol=1e-4)


def test_models_mode_changes_with_any_hyperparameter():
    assert wl.models_mode() != wl.models_mode(units=16)
    assert wl.models_mode() != wl.models_mode(window_hours=6)
    assert wl.models_mode() == wl.models_mode()


def test_sequence_and_constant_columns_are_separated_and_repeated_per_step():
    seq, static = wl.split_features(["w10", "cape", "latitude", "ctrl_noise"], ["w10", "cape", "mcpr"])
    assert seq == ["w10", "cape"] and static == ["latitude", "ctrl_noise"]
    x = wl.stack_inputs(np.zeros((4, 5, 2), "float32"), np.arange(8, dtype="float32").reshape(4, 2))
    assert x.shape == (4, 5, 4) and (x[1, :, 2] == 2).all() and (x[1, :, 3] == 3).all()
    assert wl.stack_inputs(np.zeros((4, 5, 2), "float32"), np.empty((4, 0), "float32")).shape == (4, 5, 2)


def test_valid_rows_flags_any_gap_in_the_window():
    x = np.ones((3, 4, 2), "float32")
    x[1, 2, 0] = np.nan
    assert wl.valid_rows(x).tolist() == [True, False, True]


def test_the_lstm_runs_from_prepare_to_aggregate_on_the_same_arms_and_rows(tmp_path):
    """Ponta a ponta: o `prepare` do estudo, a LSTM em todos os trimestres e o MESMO `aggregate` do
    LazyPredict. É o elo que garante que a LSTM entra na análise sem caso especial."""
    from src.feature_study.core.analysis import load_arms, run_aggregate
    from src.feature_study.core.prepare import prepare
    from test.conftest import build_synthetic_raw_dir
    from test.test_feature_study_groups_source import DIAS, _escreve

    raw = build_synthetic_raw_dir(tmp_path / "raw", DIAS)
    _escreve(raw)
    estudo = tmp_path / "estudo"
    prepare(str(raw), estudo, cluster_id=9, arm_sets=("anchors", "groups", "controls"))
    arms = [a for a in load_arms(estudo / "data") if a.name in ("base", "full", "drop_grp__grupo2", "ctrl__noise")]
    for season in ("DJF", "MAM", "JJA", "SON"):
        wl.run_unit_lstm(estudo / "data", estudo, raw, "full", season, arms, out_tag="full_lstm", seed=42,
                         window_hours=4, units=4, epochs=2, batch_size=128, patience=1)

    unit = estudo / "units" / "full_lstm"
    assert (unit / "metrics__DJF__base.parquet").exists() and (unit / "resid__DJF__full.parquet").exists()
    m = pd.read_parquet(unit / "metrics__DJF__full.parquet")
    assert set(m["models_mode"]) == {wl.models_mode(window_hours=4, units=4, epochs=2, batch_size=128, patience=1)}
    assert set(m["model"]) == {wl.LSTM_NAME}

    # idempotente: a segunda rodada pula o que já existe
    antes = (unit / "metrics__DJF__full.parquet").stat().st_mtime_ns
    wl.run_unit_lstm(estudo / "data", estudo, raw, "full", "DJF", arms, out_tag="full_lstm", seed=42,
                     window_hours=4, units=4, epochs=2, batch_size=128, patience=1)
    assert (unit / "metrics__DJF__full.parquet").stat().st_mtime_ns == antes

    res = run_aggregate(estudo, estudo / "data", ["full_lstm"], label="lstm", n_boot=50)
    assert {"full_vs_base", "drop_grp__grupo2", "ctrl__noise"} <= set(res["effects"]["comparison"])
    assert (estudo / "summary" / "lstm" / "effects.csv").exists()


def test_permutation_importance_blames_the_informative_column_and_its_group():
    x_tr, y_tr = _data(500, seed=1)
    x_va, y_va = _data(120, seed=2)
    x_te, y_te = _data(300, seed=3)
    keep: dict = {}
    wl.fit_arm_lstm(x_tr, y_tr, x_va, y_va, x_te, y_te, np.arange(300), seed=7, model_out=keep, **HP)
    imp = wl.permutation_importance(keep["predict"], keep["x_test_scaled"], keep["y_test"],
                                    ["util", "ctrl_noise"], {"util": "grupo2"}, n_repeats=2, seed=1)
    col = imp[imp.nivel == "coluna"].groupby("nome").d_rmse.mean()
    assert col["util"] > 10 * max(col["ctrl_noise"], 1e-3)
    grp = imp[imp.nivel == "grupo"].groupby("nome").d_rmse.mean()
    assert grp["grupo2"] == pytest.approx(col["util"], rel=0.5) and "ruído (controle)" in grp.index
    assert set(imp.columns) >= {"nivel", "nome", "grupo", "n_colunas", "repeticao", "d_rmse", "d_rmse_p90"}


def test_the_full_arm_also_writes_its_importance(tmp_path):
    from src.feature_study.core.analysis import load_arms
    from src.feature_study.core.prepare import prepare
    from test.conftest import build_synthetic_raw_dir
    from test.test_feature_study_groups_source import DIAS, _escreve

    raw = build_synthetic_raw_dir(tmp_path / "raw", DIAS)
    _escreve(raw)
    estudo = tmp_path / "estudo"
    prepare(str(raw), estudo, cluster_id=9, arm_sets=("anchors",))
    arms = load_arms(estudo / "data")
    wl.run_unit_lstm(estudo / "data", estudo, raw, "full", "DJF", arms, out_tag="full_lstm", seed=42,
                     window_hours=4, units=4, epochs=2, batch_size=128, patience=1, importance_repeats=1)
    imp = pd.read_parquet(estudo / "units" / "full_lstm" / "importance__DJF__full.parquet")
    assert {"grupo", "coluna"} == set(imp["nivel"]) and {"grupo1", "grupo2", "grupo3", "grupo4"} <= set(imp["nome"])
    assert not (estudo / "units" / "full_lstm" / "importance__DJF__base.parquet").exists()   # só o arm full


def test_every_arm_is_evaluated_on_the_same_rows_even_if_one_arm_has_columns_with_gaps(tmp_path):
    """Um arm com colunas de sequência que têm lacunas NÃO pode ficar com outro conjunto de linhas
    que o `base`: o `compute_effects` descartaria o par, e a comparação sumiria sem erro."""
    from src.feature_study.core.analysis import load_arms
    from src.feature_study.core.prepare import prepare
    from test.conftest import build_synthetic_raw_dir
    from test.test_feature_study_groups_source import DIAS, _escreve

    raw = build_synthetic_raw_dir(tmp_path / "raw", DIAS)
    _escreve(raw)
    # lacuna só numa coluna do grupo 3 (ausente do arm `base`): se o critério fosse por arm,
    # o `full` perderia linhas e o `base` não
    from src.feature_study.data import groups_source as gs
    g3 = pd.read_parquet(raw / gs.SUBDIR / gs.GROUP_FILES["grupo3"])
    g3.loc[g3.index[700:703], "w850"] = np.nan
    g3.to_parquet(raw / gs.SUBDIR / gs.GROUP_FILES["grupo3"], index=False)

    estudo = tmp_path / "estudo"
    prepare(str(raw), estudo, cluster_id=9, arm_sets=("anchors",))
    arms = load_arms(estudo / "data")
    wl.run_unit_lstm(estudo / "data", estudo, raw, "full", "DJF", arms, out_tag="full_lstm", seed=42,
                     window_hours=4, units=4, epochs=2, batch_size=128, patience=1, importance_arms=())
    unit = estudo / "units" / "full_lstm"
    ids = {a.name: set(pd.read_parquet(unit / f"resid__DJF__{a.name}.parquet")["row_id"]) for a in arms}
    assert ids["base"] == ids["full"] and len(ids["base"]) > 0
