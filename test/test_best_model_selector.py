"""Testes para src/dataset/creation/best_model_selector.py.

Constrói uma árvore de artefatos sintética em disco (tmp_path), no mesmo
layout de artifacts/{lazy_modal/lazy_clusters,mlp_modal/mlp_clusters,
modal/experiments}/{arm}/, pra não depender dos artefatos reais do repo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset.creation.best_model_selector import (
    Combo,
    PIPELINE_ROOTS,
    apply_fitted_model_fallback,
    build_winner_table,
    derive_quarterly_from_predictions,
    discover_combos,
    load_results,
    resolve_all_season,
    select_winner_table,
    select_winners,
)
from src.pipelines.common import compute_metrics


def _write_results_csv(combo_dir: Path, rows: list[dict]) -> None:
    combo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(combo_dir / "results.csv", index=False)


def _write_fitted_models(combo_dir: Path, cluster_ids: list[int], suffix: str) -> None:
    models_dir = combo_dir / "fitted_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for cid in cluster_ids:
        (models_dir / f"best_model_c{cid}.{suffix}").write_bytes(b"")
    if suffix == "keras":
        (models_dir / "dl_metadata.joblib").write_bytes(b"")


def _write_station_predictions(combo_dir: Path, df: pd.DataFrame) -> None:
    preds_dir = combo_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(preds_dir / "predictions_by_station.csv", index=False)


def _base_result_row(pipeline: str, arm: str, cluster_id: int, season: str, split: str, **overrides) -> dict:
    row = {
        "pipeline": pipeline, "experiment": arm, "cluster_id": cluster_id,
        "season": season, "split": split, "n_samples": 100,
        "R2": 0.5, "RMSE": 2.0, "Bias": 0.1, "Bias_P90": 0.1, "RMSE_P90": 3.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def artifacts_root(tmp_path):
    return tmp_path


class TestDiscoverCombos:
    def test_finds_all_present_combos(self, artifacts_root):
        for pipeline in ("lazy", "mlp", "lstm"):
            for arm in ("original", "synthetic"):
                combo_dir = artifacts_root / PIPELINE_ROOTS[pipeline] / arm
                _write_results_csv(combo_dir, [_base_result_row(pipeline, arm, 1, "ALL", "test")])

        combos = discover_combos(artifacts_root)
        found = {(c.pipeline, c.arm) for c in combos}
        assert found == {
            ("lazy", "original"), ("lazy", "synthetic"),
            ("mlp", "original"), ("mlp", "synthetic"),
            ("lstm", "original"), ("lstm", "synthetic"),
        }

    def test_missing_arm_is_skipped_not_raised(self, artifacts_root, capsys):
        combos = discover_combos(artifacts_root)
        assert combos == []
        captured = capsys.readouterr()
        assert "AVISO" in captured.out

    def test_available_clusters_parses_joblib_and_keras(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [_base_result_row("lazy", "original", 1, "ALL", "test")])
        _write_fitted_models(combo_dir, [1, 2, 5], suffix="joblib")

        combos = discover_combos(artifacts_root)
        assert combos[0].available_clusters() == {1, 2, 5}


class TestMetricDirection:
    def test_bias_prefers_closest_to_zero(self, artifacts_root):
        combo_dir_a = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        combo_dir_b = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir_a, [
            _base_result_row("lazy", "original", 1, "ALL", "test", Bias=-0.5),
        ])
        _write_results_csv(combo_dir_b, [
            _base_result_row("mlp", "original", 1, "ALL", "test", Bias=0.05),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)

        winners = select_winners(results, metric="Bias", season=None)
        assert winners[1] == ("mlp", "original")

    def test_rmse_prefers_lower(self, artifacts_root):
        combo_dir_a = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        combo_dir_b = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir_a, [
            _base_result_row("lazy", "original", 1, "ALL", "test", RMSE=5.0),
        ])
        _write_results_csv(combo_dir_b, [
            _base_result_row("mlp", "original", 1, "ALL", "test", RMSE=1.5),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)

        winners = select_winners(results, metric="RMSE", season=None)
        assert winners[1] == ("mlp", "original")

    def test_r2_prefers_higher(self, artifacts_root):
        combo_dir_a = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        combo_dir_b = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir_a, [
            _base_result_row("lazy", "original", 1, "ALL", "test", R2=0.2),
        ])
        _write_results_csv(combo_dir_b, [
            _base_result_row("mlp", "original", 1, "ALL", "test", R2=0.6),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)

        winners = select_winners(results, metric="R2", season=None)
        assert winners[1] == ("mlp", "original")


class TestDeriveQuarterlyFromPredictions:
    def test_matches_compute_metrics_directly(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir, [_base_result_row("mlp", "original", 1, "ALL", "test")])

        rng = np.random.default_rng(0)
        n = 40
        y_true = rng.uniform(5, 20, n)
        y_pred = y_true + rng.normal(0, 1, n)
        times = pd.date_range("2024-01-01", periods=n, freq="D")  # todo DJF/parte de MAM
        preds = pd.DataFrame({
            "estacao": "A001", "time": times, "latitude": -25.0, "longitude": -50.0,
            "cluster_id": 1, "split": "test", "y_true": y_true, "y_pred": y_pred,
        })
        _write_station_predictions(combo_dir, preds)

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        derived = derive_quarterly_from_predictions(combos[0], results)

        assert not derived.empty
        djf_mask = times.month.isin([12, 1, 2])
        expected = compute_metrics(y_true[djf_mask], y_pred[djf_mask])
        djf_row = derived[derived["season"] == "DJF"].iloc[0]
        assert djf_row["R2"] == pytest.approx(expected["R2"])
        assert djf_row["RMSE"] == pytest.approx(expected["RMSE"])
        assert djf_row["n_samples"] == int(djf_mask.sum())

    def test_skips_combo_with_native_seasons(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lazy", "original", 1, "DJF", "test"),
            _base_result_row("lazy", "original", 1, "MAM", "test"),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        derived = derive_quarterly_from_predictions(combos[0], results)
        assert derived.empty

    def test_skips_predictions_missing_split_column(self, artifacts_root):
        """Bug real encontrado numa run remota do corrected_grid no Modal:
        predictions_by_station.csv de um schema mais antigo, sem coluna
        'split', derrubava load_all_results_with_derived inteiro com
        KeyError('split'). Deve pular esse combo com aviso, não crashar."""
        combo_dir = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir, [_base_result_row("mlp", "original", 1, "ALL", "test")])

        preds = pd.DataFrame({
            "estacao": ["A1", "A1"],
            "time": ["2021-01-15", "2021-04-15"],
            "cluster_id": [1, 1],
            "y_true": [10.0, 8.0],
            "y_pred": [9.5, 7.5],
        })  # sem coluna "split" — schema antigo.
        _write_station_predictions(combo_dir, preds)

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        derived = derive_quarterly_from_predictions(combos[0], results)

        assert derived.empty


class TestResolveAllSeason:
    def test_derives_all_for_lstm_weighted_by_n_samples(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lstm"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lstm", "original", 1, "DJF", "test", n_samples=100, R2=0.4),
            _base_result_row("lstm", "original", 1, "MAM", "test", n_samples=100, R2=0.6),
            _base_result_row("lstm", "original", 1, "JJA", "test", n_samples=100, R2=0.5),
            _base_result_row("lstm", "original", 1, "SON", "test", n_samples=100, R2=0.5),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        resolved = resolve_all_season(results)

        all_row = resolved[resolved["season"] == "ALL"].iloc[0]
        assert all_row["n_samples"] == 400
        assert all_row["R2"] == pytest.approx(0.5)

    def test_native_all_not_overridden(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lazy", "original", 1, "ALL", "test", n_samples=999, R2=0.9),
            _base_result_row("lazy", "original", 1, "DJF", "test", n_samples=10, R2=0.1),
        ])
        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        resolved = resolve_all_season(results)

        all_rows = resolved[resolved["season"] == "ALL"]
        assert len(all_rows) == 1
        assert all_rows.iloc[0]["n_samples"] == 999


class TestFallback:
    def test_falls_back_when_winner_has_no_fitted_model(self, artifacts_root):
        winner_dir = artifacts_root / PIPELINE_ROOTS["lstm"] / "newfeatures"
        runnerup_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(winner_dir, [
            _base_result_row("lstm", "newfeatures", 1, "DJF", "test", R2=0.9),
        ])
        _write_results_csv(runnerup_dir, [
            _base_result_row("lazy", "original", 1, "DJF", "test", R2=0.3),
        ])
        # lstm/newfeatures não tem modelo salvo pro cluster 1 (gap de
        # --restrict-coverage); lazy/original tem.
        _write_fitted_models(runnerup_dir, [1], suffix="joblib")

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        table = select_winner_table(results, metric="R2", seasons=["DJF"])
        table = apply_fitted_model_fallback(table, combos, results, "R2")

        row = table[(table["cluster_id"] == 1) & (table["season"] == "DJF")].iloc[0]
        assert row["pipeline"] == "lazy"
        assert row["arm"] == "original"
        assert row["fallback_from"] == "lstm/newfeatures"
        assert row["has_fitted_model"]

    def test_no_fallback_when_winner_has_model(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lazy", "original", 1, "DJF", "test", R2=0.5),
        ])
        _write_fitted_models(combo_dir, [1], suffix="joblib")

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        table = select_winner_table(results, metric="R2", seasons=["DJF"])
        table = apply_fitted_model_fallback(table, combos, results, "R2")

        row = table[(table["cluster_id"] == 1) & (table["season"] == "DJF")].iloc[0]
        assert row["fallback_from"] is None
        assert row["has_fitted_model"]

    def test_no_candidate_with_model_reports_missing(self, artifacts_root):
        """Nenhum combo tem modelo salvo pro cluster/trimestre — deve
        marcar has_fitted_model=False em vez de crashar."""
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lazy", "original", 9, "JJA", "test", R2=0.4),
        ])
        # Sem _write_fitted_models — nenhum modelo salvo.

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        table = select_winner_table(results, metric="R2", seasons=["JJA"])
        table = apply_fitted_model_fallback(table, combos, results, "R2")

        row = table[(table["cluster_id"] == 9) & (table["season"] == "JJA")].iloc[0]
        assert not row["has_fitted_model"]


class TestLazyDuplicateRows:
    def test_load_station_predictions_dedups(self, artifacts_root):
        from src.dataset.creation.best_model_selector import load_station_predictions

        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [_base_result_row("lazy", "original", 1, "DJF", "test")])
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        one_copy = pd.DataFrame({
            "estacao": "A001", "time": times, "latitude": -25.0, "longitude": -50.0,
            "cluster_id": 1, "split": "test", "y_true": 10.0, "y_pred": 9.5,
        })
        duplicated = pd.concat([one_copy, one_copy], ignore_index=True)
        _write_station_predictions(combo_dir, duplicated)

        combos = discover_combos(artifacts_root)
        loaded = load_station_predictions(combos[0])
        assert len(loaded) == 5


# ── Fase 2 do plano de correção ERA5 2000-2024: modelos lazy por trimestre ──
#
# Cobre a lógica season-aware adicionada a Combo.available_cluster_seasons()
# e apply_fitted_model_fallback() — antes, `available_clusters()` só sabia
# dizer "tem algum modelo salvo pro cluster X", sem distinguir se era o
# modelo certo pro trimestre ou só o pooled (season=None).

def _write_fitted_model_seasons(combo_dir: Path, entries: list[tuple[int, str | None]]) -> None:
    """entries: lista de (cluster_id, season) — season=None grava sem
    sufixo (pooled), senão grava best_model_c{cid}_{season}.joblib."""
    models_dir = combo_dir / "fitted_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for cid, season in entries:
        suffix = f"_{season}" if season else ""
        (models_dir / f"best_model_c{cid}{suffix}.joblib").write_bytes(b"")


class TestAvailableClusterSeasons:
    def test_parses_pooled_and_per_season_filenames(self, artifacts_root):
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_fitted_model_seasons(combo_dir, [(1, None), (2, "DJF"), (2, "JJA")])
        combo = Combo(pipeline="lazy", arm="original", dir=combo_dir, feature_groups="original")

        assert combo.available_cluster_seasons() == {(1, None), (2, "DJF"), (2, "JJA")}
        assert combo.available_clusters() == {1, 2}

    def test_empty_when_no_models_dir(self, artifacts_root):
        combo = Combo(pipeline="lazy", arm="original", dir=artifacts_root / "nope",
                       feature_groups="original")
        assert combo.available_cluster_seasons() == set()
        assert combo.available_clusters() == set()


class TestSeasonAwareFallback:
    def test_accepts_winner_with_season_specific_model(self, artifacts_root):
        """Vencedor (lazy, original) pro cluster 1 trimestre DJF tem modelo
        específico de DJF salvo (além do pooled) — deve ser aceito direto,
        sem fallback."""
        combo_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("lazy", "original", 1, "DJF", "test", R2=0.8),
        ])
        _write_fitted_model_seasons(combo_dir, [(1, None), (1, "DJF")])

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        table = select_winner_table(results, metric="R2", seasons=["DJF"])
        table = apply_fitted_model_fallback(table, combos, results, "R2")

        row = table[(table["cluster_id"] == 1) & (table["season"] == "DJF")].iloc[0]
        assert row["has_fitted_model"]
        assert row["pipeline"] == "lazy"
        assert row["fallback_from"] is None

    def test_falls_back_to_pooled_when_no_season_specific_model(self, artifacts_root):
        """Combo só tem o modelo pooled (mlp/lstm hoje, ou lazy antes do
        fix) — ainda deve contar como tendo modelo (fallback implícito pro
        pooled), sem precisar descer pro próximo candidato do ranking."""
        combo_dir = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(combo_dir, [
            _base_result_row("mlp", "original", 1, "DJF", "test", R2=0.5),
        ])
        _write_fitted_model_seasons(combo_dir, [(1, None)])

        combos = discover_combos(artifacts_root)
        results = load_results(combos)
        table = select_winner_table(results, metric="R2", seasons=["DJF"])
        table = apply_fitted_model_fallback(table, combos, results, "R2")

        row = table[(table["cluster_id"] == 1) & (table["season"] == "DJF")].iloc[0]
        assert row["has_fitted_model"]
        assert row["fallback_from"] is None


class TestBuildWinnerTableEndToEnd:
    def test_lazy_per_season_champion_wins_over_mlp_all_only(self, artifacts_root):
        mlp_dir = artifacts_root / PIPELINE_ROOTS["mlp"] / "original"
        _write_results_csv(mlp_dir, [
            _base_result_row("mlp", "original", 1, "ALL", "test", n_samples=40, R2=0.5),
        ])
        _write_fitted_model_seasons(mlp_dir, [(1, None)])

        lazy_dir = artifacts_root / PIPELINE_ROOTS["lazy"] / "original"
        _write_results_csv(lazy_dir, [
            _base_result_row("lazy", "original", 1, "ALL", "test", n_samples=40, R2=0.4),
            _base_result_row("lazy", "original", 1, "DJF", "test", n_samples=10, R2=0.9),
        ])
        _write_fitted_model_seasons(lazy_dir, [(1, None), (1, "DJF")])

        table, combos = build_winner_table(str(artifacts_root), metric="R2")

        assert {(c.pipeline, c.arm) for c in combos} == {("mlp", "original"), ("lazy", "original")}
        djf_row = table[(table["cluster_id"] == 1) & (table["season"] == "DJF")]
        assert len(djf_row) == 1
        assert djf_row.iloc[0]["pipeline"] == "lazy"
        assert djf_row.iloc[0]["has_fitted_model"]
