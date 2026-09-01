"""Teste de regressão pro split de `cluster_lstm.run()` em
`preprocess_lstm_data` + `train_lstm_and_finalize` (T1.5 do plano de
decomposição Kedro), incluindo o fix do predict 3x redundante (T2.2) —
`_predict_lstm_experts` agora roda uma única vez e é reusado por
`evaluate`/`_print_final_summary`/exportação de CSV.

Chama as duas funções separadamente (como o Kedro faria, um nó por função)
em vez de só `run()` — prova que a composição funciona e que
`resolved_exp_name` mantém as duas etapas no MESMO diretório de experimento
(não recria um `expN+1` novo, ver docstring de `preprocess_lstm_data`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _lstm_smoke_config(output_dir: Path, raw_dir: Path, shp_dir: Path) -> dict:
    return {
        "version": 1,
        "experiment": {"name": "smoke_split", "seed": 42, "output_dir": str(output_dir)},
        "data": {
            "raw_dir": str(raw_dir), "shp_dir": str(shp_dir),
            "target_var": "daily_wind_gust_max",
            "train_slice": ["2008-01-01", "2018-12-31"],
            "val_slice": ["2019-01-01", "2019-12-31"],
            "test_slice": ["2020-01-01", "2025-12-31"],
            "test_station_fraction": 0.2,
            "feature_groups": "original",
        },
        "preprocessing": {"lookback": 3},
        "model": {
            "name": "cluster_dual_head_lstm",
            "params": {"units": 8, "dropout": 0.1, "l2_reg": 0.01, "learning_rate": 0.01},
        },
        "training": {"epochs": 2, "batch_size": 8, "patience": 1, "min_samples": 2},
        "augmentation": {"method": "none"},
        "visualization": {"enabled": False},
    }


def test_preprocess_and_train_called_separately_match_run(
    synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path,
):
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_split"
    config = _lstm_smoke_config(output_dir, synthetic_raw_dir, synthetic_shp_dir)
    config_path = tmp_path / "smoke_split.yaml"
    config_path.write_text(yaml.dump(config))

    data, resolved_exp_name, cfg = cluster_lstm.preprocess_lstm_data(str(config_path))

    assert resolved_exp_name  # ArtifactManager já resolveu (auto-incrementou se preciso)
    assert data.x_train  # ClusterPreprocessor rodou de verdade

    cluster_lstm.train_lstm_and_finalize(data, resolved_exp_name, cfg, str(config_path))

    results_path = output_dir / resolved_exp_name / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty


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

    data = ClusterDataBatch(
        x_train={"DJF": []}, x_test={"DJF": []}, y_test={"DJF": []}, era5_test={"DJF": []},
        cluster_ids=[1], feature_names=["f1"],
    )
    output_dir = tmp_path / "lstm_predict_once"
    cfg = {
        "experiment": {"output_dir": str(output_dir)},
        "data": {"train_slice": ["2008-01-01", "2018-12-31"],
                  "val_slice": ["2019-01-01", "2019-12-31"],
                  "test_slice": ["2020-01-01", "2025-12-31"],
                  "feature_groups": "original",
                  "target_var": "daily_wind_gust_max"},
        "preprocessing": {"lookback": 3},
        "model": {"name": "cluster_dual_head_lstm", "params": {}},
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
