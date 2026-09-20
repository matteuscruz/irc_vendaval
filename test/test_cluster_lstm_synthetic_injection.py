"""Teste de regressão de `_inject_synthetic` (src/pipelines/cluster_lstm.py):
retorna um `ClusterDataBatch` NOVO via `dataclasses.replace` (não muta o
input compartilhado — T1.4 do plano de decomposição Kedro) e, na LSTM v2,
opera com o alvo em m/s: o sintético herda o X da janela real de rajada mais
próxima e o seu y é a própria rajada sintética escalonada por scaler_y.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch
from src.pipeline.data.target import inverse_target
from src.pipelines.cluster_lstm import _inject_synthetic


def _build_minimal_batch() -> ClusterDataBatch:
    """1 trimestre (DJF), 1 cluster (id=1), 5 amostras reais — lookback=2,
    2 features base + 1 coluna one-hot de cluster (índice 2)."""
    n_real, lookback, n_base = 5, 2, 2
    rng = np.random.default_rng(0)

    x_djf = np.zeros((n_real, lookback, n_base + 1), dtype="float32")
    x_djf[:, :, :n_base] = rng.normal(size=(n_real, lookback, n_base))
    x_djf[:, -1, n_base] = 1.0  # one-hot do cluster 1, sempre ligado

    gust_djf = np.array([[6.0], [9.0], [12.0], [15.0], [24.0]], dtype="float32")
    scaler_y = RobustScaler().fit(gust_djf)

    return ClusterDataBatch(
        x_train={"DJF": x_djf},
        y_train={"DJF": scaler_y.transform(gust_djf).astype("float32")},
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
    original_n_rows = original_x_djf.shape[0]

    synthetic_csv = tmp_path / "synthetic.csv"
    _write_synthetic_csv(synthetic_csv)

    result = _inject_synthetic(data, str(synthetic_csv))

    assert data.x_train["DJF"] is original_x_djf
    assert data.y_train["DJF"] is original_y_djf
    assert data.x_train["DJF"].shape[0] == original_n_rows

    assert result is not data
    assert isinstance(result, ClusterDataBatch)
    assert result.x_train["DJF"].shape[0] == original_n_rows + 3
    assert result.y_train["DJF"].shape[0] == original_n_rows + 3


def test_injected_targets_are_the_synthetic_gusts_in_ms(tmp_path: Path):
    data = _build_minimal_batch()
    synthetic_csv = tmp_path / "synthetic.csv"
    _write_synthetic_csv(synthetic_csv)

    result = _inject_synthetic(data, str(synthetic_csv))

    y_new = inverse_target(result.scaler_y, result.y_train["DJF"][-3:])
    np.testing.assert_allclose(y_new, [20.0, 22.0, 25.0], rtol=1e-5)
    # X emprestado da janela real de rajada mais próxima (24 m/s → índice 4,
    # o searchsorted de 20/22 também cai no primeiro >= , ou seja, 24 m/s)
    np.testing.assert_array_equal(result.x_train["DJF"][-1], data.x_train["DJF"][4])


def test_inject_synthetic_leaves_untouched_seasons_as_is(tmp_path: Path):
    """Season sem estação sintética correspondente (nenhuma no CSV) fica
    exatamente igual (mesma referência) no objeto retornado."""
    data = _build_minimal_batch()
    data.x_train["JJA"] = np.zeros((2, 2, 3), dtype="float32")
    data.y_train["JJA"] = np.zeros((2, 1), dtype="float32")
    original_jja = data.x_train["JJA"]

    synthetic_csv = tmp_path / "synthetic.csv"
    _write_synthetic_csv(synthetic_csv)

    result = _inject_synthetic(data, str(synthetic_csv))

    assert result.x_train["JJA"] is original_jja
