"""Smoke tests de ponta a ponta para as 4 pipelines reais (cluster_lazy,
cluster_mlp, cluster_lstm, cluster_gan): cada uma roda contra dados
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


def test_cluster_gan_smoke(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_gan

    output_dir = tmp_path / "gan_clusters"
    cluster_gan.run(
        raw_dir=str(synthetic_raw_dir),
        shp_dir=str(synthetic_shp_dir),
        output_dir=str(output_dir),
        exp_name="smoke",
        epochs=3,
        n_per_cluster=10,
    )

    csv_path = output_dir / "smoke" / "synthetic_augment.csv"
    assert csv_path.exists(), f"synthetic_augment.csv não foi gerado em {csv_path}"
    synthetic = pd.read_csv(csv_path)
    assert not synthetic.empty
    assert "daily_wind_gust_max" in synthetic.columns
    assert "cluster_id" in synthetic.columns


def _lstm_smoke_config(output_dir: Path, raw_dir: Path, shp_dir: Path, augmentation: dict) -> dict:
    return {
        "version": 1,
        "experiment": {
            "name": "smoke",
            "seed": 42,
            "output_dir": str(output_dir),
        },
        "data": {
            "raw_dir": str(raw_dir),
            "shp_dir": str(shp_dir),
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
        "augmentation": augmentation,
        "visualization": {"enabled": False},
    }


def test_cluster_lstm_smoke(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_experiments"
    config = _lstm_smoke_config(
        output_dir, synthetic_raw_dir, synthetic_shp_dir, {"method": "none"}
    )
    config_path = tmp_path / "smoke_lstm.yaml"
    config_path.write_text(yaml.dump(config))

    cluster_lstm.run(config=str(config_path))

    results_path = output_dir / "smoke" / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty


def test_cluster_lstm_smoke_extreme_gan(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    """Fase 1 do plano de melhorias do GAN: ExGANAugmenter (gera X+y juntos,
    sem nearest-neighbor pós-hoc) acionado via augmentation.method do YAML,
    caminho até agora nunca exercitado pelo ablation do LSTM."""
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_experiments_extreme_gan"
    config = _lstm_smoke_config(
        output_dir, synthetic_raw_dir, synthetic_shp_dir,
        {
            "method": "extreme_gan",
            "extreme_percentile": 90.0,
            "multiplier": 1.0,
            "extreme_gan": {
                "latent_dim": 4, "hidden_units": 8, "epochs": 1, "batch_size": 8,
                "n_critic": 1, "k_shift": 1, "c_shift": 0.3,
            },
        },
    )
    config_path = tmp_path / "smoke_lstm_extreme_gan.yaml"
    config_path.write_text(yaml.dump(config))

    cluster_lstm.run(config=str(config_path))

    results_path = output_dir / "smoke" / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty


def test_cluster_lstm_smoke_extreme_diffusion(synthetic_raw_dir: Path, synthetic_shp_dir: Path, tmp_path: Path):
    """Fase 1 do plano de melhorias do GAN: ExtremeDiffusionAugmenter (DDPM+CFG,
    gera X+y juntos) acionado via augmentation.method do YAML."""
    from src.pipelines import cluster_lstm

    output_dir = tmp_path / "lstm_experiments_extreme_diffusion"
    config = _lstm_smoke_config(
        output_dir, synthetic_raw_dir, synthetic_shp_dir,
        {
            "method": "extreme_diffusion",
            "extreme_percentile": 90.0,
            "multiplier": 1.0,
            "extreme_diffusion": {
                "time_steps": 10, "sampling_steps": 3, "hidden_units": 8,
                "time_emb_dim": 4, "epochs": 1, "batch_size": 8,
            },
        },
    )
    config_path = tmp_path / "smoke_lstm_extreme_diffusion.yaml"
    config_path.write_text(yaml.dump(config))

    cluster_lstm.run(config=str(config_path))

    results_path = output_dir / "smoke" / "_partial" / "csv" / "results.csv"
    assert results_path.exists(), f"results.csv não foi gerado em {results_path}"
    results = pd.read_csv(results_path)
    assert not results.empty
