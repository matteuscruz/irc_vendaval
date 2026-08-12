"""Testes da avaliação (src/gan/evaluate.py)."""

import numpy as np

from src.gan.evaluate import evaluate_samples, frechet_distance_1d, tail_metrics


def test_frechet_zero_for_identical():
    x = np.random.RandomState(0).normal(5, 2, 1000)
    assert frechet_distance_1d(x, x) == 0.0


def test_frechet_positive_for_shifted():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(5, 1, 1000)
    assert frechet_distance_1d(a, b) > 1.0


def test_evaluate_samples_keys():
    rng = np.random.RandomState(1)
    real = rng.normal(10, 3, 2000)
    fake = rng.normal(10, 3, 2000)
    m = evaluate_samples(real, fake)
    for key in ("frechet_1d", "wasserstein", "ks_stat", "ks_pvalue",
                "delta_P90", "delta_P95", "delta_P99"):
        assert key in m
    # Distribuições próximas -> Wasserstein pequeno
    assert m["wasserstein"] < 1.0


def test_tail_metrics_delta_sign():
    real = np.linspace(0, 10, 1000)
    fake = np.linspace(0, 20, 1000)  # cauda mais alta
    m = tail_metrics(real, fake)
    assert m["delta_P90"] > 0
