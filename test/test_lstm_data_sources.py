"""Fontes de dados da LSTM v2 — diária (`ClusterPreprocessor`) e horária
(`build_hourly_batch`) — sobre os fixtures sintéticos diários contínuos.
Sem TensorFlow: só construção de janelas, split, purga e anulação de lags."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.pipeline.data.cluster_preprocessor import SPLITS, ClusterPreprocessor
from src.pipeline.data.hourly_builder import build_hourly_batch
from src.pipeline.data.lstm_sources import validate_lstm_config
from src.pipeline.data.target import inverse_target
from src.pipelines.common import INMET_DERIVED_FEATURES

TEST_MONTHS = {1, 4, 7, 10}
AFTER_TEST = {2, 5, 8, 11}


def _times(meta_by_season) -> pd.DatetimeIndex:
    parts = [np.asarray(m["time"], dtype="datetime64[ns]") for m in meta_by_season.values() if len(m["time"])]
    return pd.DatetimeIndex(np.concatenate(parts)) if parts else pd.DatetimeIndex([])


def _stack(d) -> np.ndarray:
    return np.concatenate([v for v in d.values() if len(v)], axis=0)


def _assert_targets_match_inmet(data, raw_dir) -> None:
    with xr.open_dataset(raw_dir / "INMET_Stratified.nc") as ds:
        obs = ds["daily_wind_gust_max"].load().to_series()
    obs.index = obs.index.set_names(["time", "estacao"])
    for season, meta in data.meta_test.items():
        if not len(meta["time"]):
            continue
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(meta["time"]), meta["estacao"]], names=["time", "estacao"],
        )
        y = inverse_target(data.scaler_y, data.y_test[season])
        np.testing.assert_allclose(y, obs.reindex(idx).to_numpy(), rtol=1e-5)


def test_daily_batch(synthetic_daily_raw_dir, synthetic_shp_dir):
    data = ClusterPreprocessor(
        str(synthetic_daily_raw_dir), str(synthetic_shp_dir),
        lookback=3, feature_groups="original", split_cfg={"val_fraction": 0.25}, seed=0,
    ).run()

    assert data.resolution == "daily" and data.window_length == 3
    assert data.climatology["method"] == "harmonic"
    n_feat = len(data.feature_names)
    for sp in SPLITS:
        x = getattr(data, f"x_{sp}")
        assert sum(len(v) for v in x.values()) > 0, f"split {sp} vazio"
        for v in x.values():
            assert v.shape[1:] == (3, n_feat)

    # Rótulos: teste só Jan/Abr/Jul/Out; validação = blocos (ano, mês) sorteados
    t_test, t_train, t_val = (_times(getattr(data, f"meta_{sp}")) for sp in ("test", "train", "val"))
    assert set(t_test.month) <= TEST_MONTHS
    assert not (set(t_train.month) | set(t_val.month)) & TEST_MONTHS
    val_units = set(data.split.val_units)
    assert set(zip(t_val.year, t_val.month)) <= val_units
    assert not set(zip(t_train.year, t_train.month)) & val_units

    # Purga (L=3): alvo nos dias 1-2 do mês cruza o mês anterior (outro rótulo)
    assert (t_test.day >= 3).all()
    after = np.isin(t_train.month, list(AFTER_TEST))
    assert (t_train[after].day >= 3).all()

    # One-hot do cluster no último passo bate com a identidade da janela
    for season, meta in data.meta_test.items():
        for i in range(0, len(meta["time"]), 97):
            col = data.feature_names.index(f"cluster_{meta['cluster_id'][i]}")
            assert data.x_test[season][i, -1, col] == 1

    _assert_targets_match_inmet(data, synthetic_daily_raw_dir)
    assert not set(INMET_DERIVED_FEATURES) & set(data.feature_names)


def test_hourly_batch(synthetic_hourly_raw_dir, synthetic_shp_dir):
    data = build_hourly_batch(
        str(synthetic_hourly_raw_dir), str(synthetic_shp_dir),
        split_cfg={"val_fraction": 0.5}, seed=0,
    )
    assert data.resolution == "hourly" and data.window_length == 24
    assert data.cluster_ids == [9] and data.feature_names[-1] == "cluster_9"
    assert sum(len(v) for v in data.x_train.values()) > 0
    x_test = _stack(data.x_test)
    assert x_test.shape[1:] == (24, len(data.feature_names))
    t_test = _times(data.meta_test)
    assert set(t_test.month) <= TEST_MONTHS
    assert set(t_test.year) <= {2018, 2019}

    # contexto diário repetido nas 24 horas; variável horária varia
    j_ctx = data.feature_names.index("day_sin")
    j_hour = data.feature_names.index("ws_h")
    assert np.allclose(x_test[:, :, j_ctx].std(axis=1), 0, atol=1e-6)
    assert (x_test[:, :, j_hour].std(axis=1) > 0).all()

    _assert_targets_match_inmet(data, synthetic_hourly_raw_dir)


@pytest.mark.parametrize("bad, match", [
    ({"data": {"train_slice": ["2008-01-01", "2018-12-31"]}}, "train_slice"),
    ({"model": {"name": "cluster_dual_head_lstm"}}, "cluster_dual_head_lstm"),
    ({"model": {"params": {"extreme_threshold": 1.5}}}, "extreme_threshold"),
    ({"data": {"resolution": "monthly"}}, "resolution"),
    ({"data": {"interp_method": "cubic"}}, "interp_method"),
    ({"data": {"split": {"purge": "none"}}}, "purge"),
    ({"data": {"exclude_observation_features": False}}, "exclude_observation_features"),
])
def test_validate_lstm_config_rejects(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_lstm_config(bad)


def test_validate_lstm_config_accepts_v2_defaults():
    cfg = {"data": {"resolution": "hourly", "interp_method": "bilinear"},
           "model": {"name": "cluster_lstm", "params": {"units": 96}}}
    assert validate_lstm_config(cfg) is cfg
