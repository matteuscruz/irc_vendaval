"""Metadados v2 da LSTM e guardas contra artefato legado / resolução errada
(src/inference/dl_metadata.py)."""
import joblib
import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from src.inference.dl_metadata import (
    METADATA_FILENAME,
    LegacyLSTMArtifactError,
    UnsupportedResolutionError,
    build_dl_metadata,
    load_dl_metadata,
)
from src.pipeline.data.splits import MonthBlockSplit
from src.pipeline.data.windowing import WindowSpec


def _meta(resolution="daily", feature_names=None):
    x = np.random.default_rng(0).normal(size=(20, 3))
    return build_dl_metadata(
        scaler_x=RobustScaler().fit(x),
        scaler_y=RobustScaler().fit(x[:, :1]),
        feature_names=feature_names or ["wind_mag_max", "t2m_range", "cluster_3"],
        window=WindowSpec(resolution, 7 if resolution == "daily" else 24),
        split=MonthBlockSplit(val_units=((2016, 2),)),
        interp_method="bilinear",
        climatology={"method": "harmonic", "n_harmonics": 3},
        hourly=None,
        model_params={"units": 96, "dropout": 0.3},
        seed=42,
    )


def test_round_trip_and_derived_fields(tmp_path):
    joblib.dump(_meta(), tmp_path / METADATA_FILENAME)
    meta = load_dl_metadata(tmp_path, require_resolution="daily")
    assert meta["schema_version"] == 2 and meta["target_kind"] == "absolute"
    assert meta["lookback"] == 7
    assert meta["cluster_feature_names"] == ["cluster_3"]
    assert meta["base_feature_names"] == ["wind_mag_max", "t2m_range"]
    assert MonthBlockSplit.from_dict(meta["split"]).val_units == ((2016, 2),)
    assert WindowSpec.from_dict(meta["window"]) == WindowSpec("daily", 7)


@pytest.mark.parametrize("legacy", [
    {"model_name": "cluster_dual_head_lstm", "lookback": 7, "extreme_threshold": 1.5},
    {"model_name": "cluster_tr_lstm", "lookback": 7},
    {"schema_version": 2, "model_name": "cluster_lstm"},  # sem target_kind
])
def test_legacy_artifacts_raise_retrain_error(tmp_path, legacy):
    joblib.dump(legacy, tmp_path / METADATA_FILENAME)
    with pytest.raises(LegacyLSTMArtifactError, match="retreine"):
        load_dl_metadata(tmp_path)


def test_model_with_inmet_derived_features_refused(tmp_path):
    joblib.dump(_meta(feature_names=["wind_mag_max", "lag1_gust_obs", "cluster_3"]),
                tmp_path / METADATA_FILENAME)
    with pytest.raises(ValueError, match="derivadas do INMET"):
        load_dl_metadata(tmp_path)


def test_hourly_model_refused_by_daily_consumer(tmp_path):
    joblib.dump(_meta("hourly"), tmp_path / METADATA_FILENAME)
    with pytest.raises(UnsupportedResolutionError):
        load_dl_metadata(tmp_path, require_resolution="daily")
    assert load_dl_metadata(tmp_path)["resolution"] == "hourly"


def test_missing_metadata_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dl_metadata(tmp_path)
