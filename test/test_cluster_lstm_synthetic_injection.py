"""Teste de regressão pro fix de mutação in-place de `_inject_synthetic`
(T1.4 do plano de decomposição Kedro) — antes reatribuía
`data.x_train[season]`/`y_train[season]`/`era5_train[season]` diretamente
no objeto `ClusterDataBatch` recebido (mutando o input compartilhado);
agora retorna um `ClusterDataBatch` NOVO via `dataclasses.replace`, mesmo
precedente já usado por `BaseAugmenter._distribute_by_season`. Esse caminho
não tinha nenhuma cobertura de teste antes deste fix — os 3 smoke tests de
LSTM existentes (`test_pipeline_smoke.py`) só exercitam a injeção via
`augmentation.method` do YAML (ExGANAugmenter/ExtremeDiffusionAugmenter),
que já retornava objeto novo corretamente; nenhum exercita `synthetic_csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch
from src.pipelines.cluster_lstm import _inject_synthetic


def _build_minimal_batch() -> ClusterDataBatch:
    """1 trimestre (DJF), 1 cluster (id=1), 5 amostras reais — lookback=2,
    2 features base + 1 coluna one-hot de cluster (índice 2)."""
    n_real, lookback, n_base = 5, 2, 2
    rng = np.random.default_rng(0)

    x_djf = np.zeros((n_real, lookback, n_base + 1), dtype="float32")
    x_djf[:, :, :n_base] = rng.normal(size=(n_real, lookback, n_base))
    x_djf[:, -1, n_base] = 1.0  # one-hot do cluster 1, sempre ligado

    era5_djf = rng.uniform(5, 15, size=n_real).astype("float32")
    ratio_djf = rng.uniform(0.8, 1.2, size=(n_real, 1)).astype("float32")

    scaler_y = RobustScaler().fit(ratio_djf)
    y_djf = scaler_y.transform(ratio_djf)

    return ClusterDataBatch(
        x_train={"DJF": x_djf},
        y_train={"DJF": y_djf},
        era5_train={"DJF": era5_djf},
        scaler_y=scaler_y,
        feature_names=["feat_a", "feat_b", "cluster_1"],
        cluster_ids=[1],
    )


def _write_synthetic_csv(path: Path) -> None:
    pd.DataFrame({
        "daily_wind_gust_max": [20.0, 22.0, 25.0],
        "cluster_id": [1, 1, 1],
    }).to_csv(path, index=False)


def test_inject_synthetic_does_not_mutate_input_batch(tmp_path: Path):
    data = _build_minimal_batch()
    original_x_djf = data.x_train["DJF"]
    original_y_djf = data.y_train["DJF"]
    original_era5_djf = data.era5_train["DJF"]
    original_n_rows = original_x_djf.shape[0]

    synthetic_csv = tmp_path / "synthetic.csv"
    _write_synthetic_csv(synthetic_csv)

    result = _inject_synthetic(data, str(synthetic_csv))

    # O objeto/arrays originais não podem ter sido tocados.
    assert data.x_train["DJF"] is original_x_djf
    assert data.y_train["DJF"] is original_y_djf
    assert data.era5_train["DJF"] is original_era5_djf
    assert data.x_train["DJF"].shape[0] == original_n_rows

    # O retorno é um objeto NOVO, com as linhas sintéticas injetadas.
    assert result is not data
    assert isinstance(result, ClusterDataBatch)
    assert result.x_train["DJF"].shape[0] == original_n_rows + 3
    assert result.y_train["DJF"].shape[0] == original_n_rows + 3
    assert result.era5_train["DJF"].shape[0] == original_n_rows + 3


def test_inject_synthetic_leaves_untouched_seasons_as_is(tmp_path: Path):
    """Season sem estação sintética correspondente (nenhuma no CSV) fica
    exatamente igual (mesma referência) no objeto retornado."""
    data = _build_minimal_batch()
    data.x_train["JJA"] = np.zeros((2, 2, 3), dtype="float32")
    data.y_train["JJA"] = np.zeros((2, 1), dtype="float32")
    data.era5_train["JJA"] = np.zeros(2, dtype="float32")
    original_jja = data.x_train["JJA"]

    synthetic_csv = tmp_path / "synthetic.csv"
    _write_synthetic_csv(synthetic_csv)

    result = _inject_synthetic(data, str(synthetic_csv))

    assert result.x_train["JJA"] is original_jja
