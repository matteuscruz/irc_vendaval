"""Smoke tests de ponta a ponta para as 4 pipelines reais (cluster_lazy,
cluster_mlp, cluster_lstm): cada uma roda contra dados
sintéticos minúsculos (test/conftest.py::synthetic_raw_dir) e a asserção é
"rodou sem crashar E produziu artefato com o schema esperado" — não
resultado estatístico (os dados são sintéticos demais pra isso fazer
sentido). Servem pra pegar regressões de integração (import quebrado,
assinatura mudada, coluna renomeada) que testes unitários não cobrem.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.pipelines.metrics_schema import CORE_RESULTS_COLUMNS


def test_cluster_lazy_smoke(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_lazy

    output_dir = tmp_path / "lazy_clusters"
    cluster_lazy.run(
        raw_dir=str(synthetic_raw_dir),
        shp_dir=str(synthetic_shp_dir),
        output_dir=str(output_dir),
        exp_name="smoke",
        feature_groups="original",
        stratify_seasons=False,
    )

    results_path = output_dir / "smoke" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty
    assert set(CORE_RESULTS_COLUMNS).issubset(results.columns)
    assert (results["pipeline"] == "lazy").all()


def test_cluster_mlp_smoke(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_mlp

    output_dir = tmp_path / "mlp_clusters"
    cluster_mlp.run(
        raw_dir=str(synthetic_raw_dir),
        shp_dir=str(synthetic_shp_dir),
        output_dir=str(output_dir),
        exp_name="smoke",
        feature_groups="original",
        stratify_seasons=False,
        max_iter=5,
    )

    results_path = output_dir / "smoke" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty
    assert set(CORE_RESULTS_COLUMNS).issubset(results.columns)
    assert (results["pipeline"] == "mlp").all()




def _lstm_smoke_config(
    output_dir: Path, raw_dir: Path, shp_dir: Path,
    resolution: str = "daily",
) -> dict:
    """Config LSTM v2 mínima — dados diários contínuos (synthetic_daily_raw_dir);
    o fixture mensal zeraria as janelas depois da purga por bloco de mês."""
    cfg = {
        "version": 2,
        "experiment": {"name": "smoke", "seed": 42, "output_dir": str(output_dir)},
        "data": {
            "raw_dir": str(raw_dir),
            "shp_dir": str(shp_dir),
            "target_var": "daily_wind_gust_max",
            "resolution": resolution,
            "feature_groups": "original",
            "interp_method": "nearest",
            "split": {"scheme": "month_block", "test_months": [1, 4, 7, 10],
                      "val_fraction": 0.25, "purge": "strict"},
            "climatology": {"method": "harmonic", "n_harmonics": 3},
            "hourly": {"file": "test_cluster_3_hourly.nc", "day_offset_hours": 0},
        },
        "preprocessing": {"lookback": 3},
        "model": {
            "name": "cluster_lstm",
            "params": {"units": 8, "dropout": 0.1, "huber_delta": 1.0, "learning_rate": 0.01},
        },
        "training": {"epochs": 2, "batch_size": 64, "patience": 1, "min_samples": 2},
        "visualization": {"enabled": False},
    }
    if resolution == "hourly":
        cfg["data"].pop("feature_groups")   # a fonte horária traz as próprias features (data.hourly)
    return cfg


def _assert_lstm_v2_artifacts(output_dir: Path, resolution: str, lookback: int) -> None:
    from src.inference.dl_metadata import load_dl_metadata

    results_path = output_dir / "smoke" / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    assert not pd.read_csv(results_path).empty

    meta_paths = list(output_dir.rglob("dl_metadata.joblib"))
    assert len(meta_paths) == 1
    meta = load_dl_metadata(meta_paths[0].parent, require_resolution=resolution)
    assert meta["lookback"] == lookback and meta["target_kind"] == "absolute"
    assert list(meta_paths[0].parent.glob("best_model_c*.keras"))

    preds = pd.read_csv(next(output_dir.rglob("predictions_by_station.csv")), parse_dates=["time"])
    assert set(preds["time"].dt.month) <= {1, 4, 7, 10}
    assert preds["y_pred"].between(0, 80).all()


def test_cluster_lstm_smoke(synthetic_daily_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_experiments"
    config = _lstm_smoke_config(
        output_dir, synthetic_daily_raw_dir, synthetic_shp_dir
    )
    config_path = tmp_path / "smoke_lstm.yaml"
    config_path.write_text(yaml.dump(config))

    cluster_lstm.run(config=str(config_path))

    _assert_lstm_v2_artifacts(output_dir, "daily", 3)


def test_cluster_lstm_smoke_hourly(synthetic_hourly_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_experiments_hourly"
    config = _lstm_smoke_config(
        output_dir, synthetic_hourly_raw_dir, synthetic_shp_dir,
        resolution="hourly",
    )
    config_path = tmp_path / "smoke_lstm_hourly.yaml"
    config_path.write_text(yaml.dump(config))

    cluster_lstm.run(config=str(config_path))

    _assert_lstm_v2_artifacts(output_dir, "hourly", 24)




