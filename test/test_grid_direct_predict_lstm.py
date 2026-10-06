"""Caminho LSTM v2 de src/inference/grid_direct_predict.py: janela [D-L+1..D]
com padding de borda, saída única já em m/s (sem razão × proxy), climatologia
harmônica recalculada por célula e recusa de artefato dual-head legado."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.inference import grid_direct_predict as gdp
from src.inference.dl_metadata import (
    METADATA_FILENAME,
    LegacyLSTMArtifactError,
    build_dl_metadata,
)
from src.pipeline.data.splits import MonthBlockSplit
from src.pipeline.data.windowing import WindowSpec

N_CELLS, N_TIMES, L = 3, 800, 7
BASE = ["wind_mag_max", "era5_clim_wind"]


class _LastStepModel:
    """"Prediz" a 1ª feature escalonada no ÚLTIMO passo da janela — se a
    janela terminar no dia D, a saída desescalonada é o vento do próprio D."""

    def __init__(self):
        self.shapes = []

    def predict(self, x, batch_size=None, verbose=0):
        self.shapes.append(x.shape)
        return x[:, -1, :1]


def _grid_frame():
    rng = np.random.default_rng(0)
    times = pd.date_range("2018-01-01", periods=N_TIMES, freq="D")
    wind = rng.uniform(2, 12, (N_CELLS, N_TIMES))
    return pd.DataFrame({
        "cell": np.repeat(np.arange(N_CELLS), N_TIMES),
        "time": np.tile(times, N_CELLS),
        "wind_mag_max": wind.ravel(),
        "era5_clim_wind": 999.0,  # versão de grid_features — deve ser substituída
    }), wind


def _meta(df):
    fit = df[BASE].assign(era5_clim_wind=np.random.default_rng(1).uniform(3, 9, len(df)))
    return build_dl_metadata(
        scaler_x=RobustScaler().fit(fit),
        scaler_y=RobustScaler().fit(fit[["wind_mag_max"]]),
        feature_names=BASE + ["cluster_1", "cluster_2"],
        window=WindowSpec("daily", L),
        split=MonthBlockSplit(),
        interp_method="bilinear",
        climatology={"method": "harmonic", "n_harmonics": 3},
        hourly=None,
        model_params={},
        seed=0,
    )


def test_window_ends_on_target_day_and_output_in_ms():
    df, wind = _grid_frame()
    model = _LastStepModel()
    pred = gdp._predict_lstm(model, _meta(df), df, cid=1, n_cells=N_CELLS, n_times=N_TIMES)

    assert pred.shape == (N_CELLS * N_TIMES,)
    np.testing.assert_allclose(pred, wind.ravel(), rtol=1e-5)
    assert all(s[1:] == (L, len(BASE) + 2) for s in model.shapes)


def test_window_with_missing_feature_is_not_predicted():
    df, wind = _grid_frame()
    df.loc[(df["cell"] == 1) & (df.index % N_TIMES == 400), "wind_mag_max"] = np.nan
    model = _LastStepModel()
    pred = gdp._predict_lstm(model, _meta(df), df, cid=1, n_cells=N_CELLS, n_times=N_TIMES)
    cell1 = pred.reshape(N_CELLS, N_TIMES)[1]
    assert np.isnan(cell1[400:400 + L]).all()  # toda janela que contém o dia 400
    assert np.isfinite(cell1[:400]).all() and np.isfinite(cell1[400 + L:]).all()


def test_classic_row_with_missing_feature_is_not_predicted():
    class _Echo:
        def predict(self, x):
            return x.iloc[:, 0].to_numpy()

    fit = pd.DataFrame({"a": [1.0, 2.0, 3.0], "nf_cape_max": [1.0, 5.0, 9.0]})
    art = {"model": _Echo(), "scaler": RobustScaler().fit(fit),
           "features": ["a", "nf_cape_max"], "target_kind": "absolute"}
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "nf_cape_max": [1.0, np.nan, 9.0],
                       "wind_mag_max": 5.0})
    pred = gdp._predict_classic(art, df)
    assert np.isnan(pred[1]) and np.isfinite(pred[[0, 2]]).all()


def test_harmonic_climatology_replaces_grid_version():
    df, _ = _grid_frame()
    clim = gdp._harmonic_clim_for_cells(df, _meta(df), N_CELLS, N_TIMES)
    assert clim.shape == (N_CELLS * N_TIMES,)
    assert np.isfinite(clim).all()  # inclusive nos meses de teste (sem amostra de treino)
    assert (np.abs(clim - 7.0) < 3.0).all()  # perto da média do vento U(2, 12)


def test_legacy_dual_head_artifact_is_refused(tmp_path):
    joblib.dump({"model_name": "cluster_dual_head_lstm", "lookback": 7,
                 "extreme_threshold": 1.5}, tmp_path / METADATA_FILENAME)
    (tmp_path / "best_model_c1.keras").write_bytes(b"")
    with pytest.raises(LegacyLSTMArtifactError, match="retreine"):
        gdp._load_lstm(tmp_path, 1)


def test_missing_metadata_means_no_lstm(tmp_path):
    assert gdp._load_lstm(tmp_path, 1) == (None, None)
