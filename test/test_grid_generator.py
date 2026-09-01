"""Testes estreitos pras funções extraídas de `grid_generator.run()`
(T1.2/T2.1 do plano de decomposição Kedro) — `select_quarterly_winners`,
`predict_all_combos`, `stitch_predictions`. Cada uma é pura lógica pandas
(exceto `predict_all_combos`, que orquestra `_predict_combo` — mockado
aqui, já que `_predict_combo` em si não muda) extraída sem mudança de
comportamento do corpo de `run()` — estes testes garantem isso.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.dataset.creation.grid_generator import (
    predict_all_combos,
    select_quarterly_winners,
    stitch_predictions,
)


def test_select_quarterly_winners_filters_season_and_fitted_model():
    winner_table = pd.DataFrame([
        {"cluster_id": 1, "season": "ALL", "pipeline": "lazy", "arm": "basin",
         "has_fitted_model": True},
        {"cluster_id": 1, "season": "DJF", "pipeline": "lazy", "arm": "basin",
         "has_fitted_model": True},
        {"cluster_id": 2, "season": "DJF", "pipeline": "mlp", "arm": "original",
         "has_fitted_model": False},
    ])

    quarterly = select_quarterly_winners(winner_table)

    assert len(quarterly) == 1
    assert quarterly.iloc[0]["cluster_id"] == 1
    assert quarterly.iloc[0]["season"] == "DJF"


def test_select_quarterly_winners_raises_when_empty():
    winner_table = pd.DataFrame([
        {"cluster_id": 1, "season": "ALL", "pipeline": "lazy", "arm": "basin",
         "has_fitted_model": True},
    ])
    with pytest.raises(RuntimeError):
        select_quarterly_winners(winner_table)


def test_predict_all_combos_orchestrates_predict_combo(monkeypatch):
    quarterly = pd.DataFrame([
        {"cluster_id": 1, "season": "DJF", "pipeline": "lazy", "arm": "basin"},
    ])

    class _FakeCombo:
        pipeline = "lazy"
        arm = "basin"

    combos = [_FakeCombo()]

    def _fake_predict_combo(combo, raw_dir, shp_dir, time_slice):
        return pd.DataFrame({
            "time": pd.to_datetime(["2021-01-15"]),
            "estacao": ["A1"],
            "cluster_id": [1],
            "latitude": [-20.0],
            "longitude": [-45.0],
            "rajada_corrigida": [12.0],
            "wind_mag_max": [10.0],
        })

    monkeypatch.setattr(
        "src.dataset.creation.grid_generator._predict_combo", _fake_predict_combo,
    )

    all_preds = predict_all_combos(quarterly, combos, "raw", "shp", ("2021-01-01", "2021-01-31"))

    assert len(all_preds) == 1
    assert all_preds.iloc[0]["pipeline"] == "lazy"
    assert all_preds.iloc[0]["arm"] == "basin"
    assert all_preds.iloc[0]["season"] == "DJF"


def test_predict_all_combos_raises_when_all_combos_empty(monkeypatch):
    quarterly = pd.DataFrame([
        {"cluster_id": 1, "season": "DJF", "pipeline": "lazy", "arm": "basin"},
    ])

    class _FakeCombo:
        pipeline = "lazy"
        arm = "basin"

    monkeypatch.setattr(
        "src.dataset.creation.grid_generator._predict_combo",
        lambda combo, raw_dir, shp_dir, time_slice: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError):
        predict_all_combos(quarterly, [_FakeCombo()], "raw", "shp", ("2021-01-01", "2021-01-31"))


def test_stitch_predictions_picks_matching_cluster_season_and_computes_residual():
    quarterly = pd.DataFrame([
        {"cluster_id": 1, "season": "DJF", "pipeline": "lazy", "arm": "basin"},
    ])
    all_preds = pd.DataFrame([
        # Bate com o vencedor — deve entrar.
        {"pipeline": "lazy", "arm": "basin", "cluster_id": 1, "season": "DJF",
         "estacao": "A1", "rajada_corrigida": 12.0, "wind_mag_max": 10.0},
        # Mesmo cluster/season, mas combo diferente (não venceu) — deve ficar de fora.
        {"pipeline": "mlp", "arm": "original", "cluster_id": 1, "season": "DJF",
         "estacao": "A1", "rajada_corrigida": 99.0, "wind_mag_max": 10.0},
        # Combo vencedor, mas trimestre diferente — deve ficar de fora.
        {"pipeline": "lazy", "arm": "basin", "cluster_id": 1, "season": "JJA",
         "estacao": "A1", "rajada_corrigida": 50.0, "wind_mag_max": 10.0},
    ])

    stitched = stitch_predictions(quarterly, all_preds)

    assert len(stitched) == 1
    assert stitched.iloc[0]["rajada_corrigida"] == 12.0
    assert stitched.iloc[0]["residual"] == pytest.approx(12.0 - 10.0)


def test_stitch_predictions_raises_when_nothing_matches():
    quarterly = pd.DataFrame([
        {"cluster_id": 1, "season": "DJF", "pipeline": "lazy", "arm": "basin"},
    ])
    all_preds = pd.DataFrame([
        {"pipeline": "mlp", "arm": "original", "cluster_id": 1, "season": "DJF",
         "estacao": "A1", "rajada_corrigida": 12.0, "wind_mag_max": 10.0},
    ])
    with pytest.raises(RuntimeError):
        stitch_predictions(quarterly, all_preds)
