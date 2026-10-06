"""Teste de regressão pro split de `cluster_lstm.run()` em
`preprocess_lstm_data` + `train_lstm_and_finalize` (T1.5 do plano de
decomposição Kedro), incluindo o fix do predict 3x redundante (T2.2) —
`_predict_lstm_experts` roda uma única vez e é reusado por
`evaluate`/`_print_final_summary`/exportação de CSV.

Chama as duas funções separadamente (como o Kedro faria, um nó por função)
em vez de só `run()` — prova que a composição funciona e que
`resolved_exp_name` mantém as duas etapas no MESMO diretório de experimento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _lstm_smoke_config(output_dir: Path, raw_dir: Path, shp_dir: Path) -> dict:
    return {
        "version": 2,
        "experiment": {"name": "smoke_split", "seed": 42, "output_dir": str(output_dir)},
        "data": {
            "raw_dir": str(raw_dir), "shp_dir": str(shp_dir),
            "target_var": "daily_wind_gust_max",
            "resolution": "daily",
            "feature_groups": "original",
            "split": {"test_months": [1, 4, 7, 10], "val_fraction": 0.25},
        },
        "preprocessing": {"lookback": 3},
        "model": {
            "name": "cluster_lstm",
            "params": {"units": 8, "dropout": 0.1, "huber_delta": 1.0, "learning_rate": 0.01},
        },
        "training": {"epochs": 2, "batch_size": 64, "patience": 1, "min_samples": 2},
        "visualization": {"enabled": False},
    }


def test_preprocess_and_train_called_separately_match_run(
    synthetic_daily_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path,
):
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_split"
    config = _lstm_smoke_config(output_dir, synthetic_daily_raw_dir, synthetic_shp_dir)
    config_path = tmp_path / "smoke_split.yaml"
    config_path.write_text(yaml.dump(config))

    data, resolved_exp_name, cfg = cluster_lstm.preprocess_lstm_data(str(config_path))

    assert resolved_exp_name  # ArtifactManager já resolveu (auto-incrementou se preciso)
    assert sum(len(v) for v in data.x_train.values()) > 0

    cluster_lstm.train_lstm_and_finalize(data, resolved_exp_name, cfg, str(config_path))

    results_path = output_dir / resolved_exp_name / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty


def test_legacy_config_is_rejected_before_loading_data(tmp_path: Path):
    from src.pipelines import cluster_lstm

    legacy = {
        "experiment": {"name": "x", "output_dir": str(tmp_path)},
        "data": {"raw_dir": "nao/existe", "shp_dir": "nao/existe",
                 "train_slice": ["2008-01-01", "2018-12-31"]},
        "model": {"name": "cluster_dual_head_lstm", "params": {"extreme_threshold": 1.5}},
    }
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.dump(legacy))
    with pytest.raises(ValueError, match="train_slice"):
        cluster_lstm.preprocess_lstm_data(str(config_path))


def test_train_and_finalize_reuses_single_prediction(monkeypatch, tmp_path: Path):
    """Prova direta do fix T2.2: `trainer.predict` deve ser chamado no
    máximo 1 vez por `train_lstm_and_finalize`, não 3."""
    from src.pipeline.data.cluster_preprocessor import ClusterDataBatch
    from src.pipelines import cluster_lstm

    call_count = {"n": 0}

    class _FakeKerasModel:
        def save(self, path):
            Path(path).touch()

    class _FakeResult:
        histories = {}
        models = {1: {"global": _FakeKerasModel()}}
        test_losses = None

    class _FakeTrainer:
        def fit(self, data):
            return _FakeResult()

        def predict(self, **kwargs):
            call_count["n"] += 1
            return {}

    monkeypatch.setattr(
        "src.pipeline.training.cluster_trainer.ClusterTrainer", lambda **kw: _FakeTrainer(),
    )

    empty_x = {"DJF": np.zeros((0, 3, 1), dtype="float32")}
    data = ClusterDataBatch(
        x_train=empty_x, x_test=empty_x, y_test={"DJF": np.zeros((0, 1), dtype="float32")},
        cluster_ids=[1], feature_names=["f1"], window_length=3,
    )
    output_dir = tmp_path / "lstm_predict_once"
    cfg = {
        "experiment": {"output_dir": str(output_dir)},
        "data": {"feature_groups": "original", "target_var": "daily_wind_gust_max"},
        "preprocessing": {"lookback": 3},
        "model": {"name": "cluster_lstm", "params": {}},
        "training": {},
        "visualization": {"enabled": False},
    }

    fake_config_path = tmp_path / "config.yaml"
    fake_config_path.write_text("placeholder: true\n")

    cluster_lstm.train_lstm_and_finalize(data, "smoke", cfg, str(fake_config_path))

    assert call_count["n"] == 1, (
        f"trainer.predict foi chamado {call_count['n']}x — deveria ser 1 "
        "(ver T2.2: predição calculada uma vez e reusada)"
    )
    assert list(output_dir.rglob("dl_metadata.joblib")), "dl_metadata.joblib não foi salvo"
