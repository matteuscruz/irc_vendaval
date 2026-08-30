"""Teste unitário puro-pandas de scripts/compare_ablation_all.py::build_ranking.

Fase 1.5 do plano de melhorias do GAN (eixo TSTR): rankear braços de ablation
por RMSE_P90/Bias_P90 em vez de só R2, que mascara desempenho na cauda. Sem
dependência de treino/dados reais — alimenta um summary_df sintético e
confere a ordem esperada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_ablation_all import build_ranking


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pipeline": "lstm", "arm": "original", "n_samples": 100, "R2": 0.29, "RMSE": 2.5, "Bias": 0.2, "Bias_P90": -2.86, "RMSE_P90": 4.74},
            {"pipeline": "lstm", "arm": "synthetic", "n_samples": 100, "R2": -0.10, "RMSE": 3.2, "Bias": 1.0, "Bias_P90": -1.65, "RMSE_P90": 4.82},
            {"pipeline": "mlp", "arm": "original", "n_samples": 100, "R2": 0.25, "RMSE": 2.6, "Bias": 0.4, "Bias_P90": -2.29, "RMSE_P90": 4.53},
            {"pipeline": "mlp", "arm": "synthetic", "n_samples": 100, "R2": 0.31, "RMSE": 2.5, "Bias": 0.25, "Bias_P90": -2.50, "RMSE_P90": 4.61},
        ]
    )


def test_rank_by_r2_higher_is_better():
    ranked = build_ranking(_summary_df(), "R2")
    lstm = ranked[ranked["pipeline"] == "lstm"].set_index("arm")
    assert lstm.loc["original", "rank"] == 1
    assert lstm.loc["synthetic", "rank"] == 2

    mlp = ranked[ranked["pipeline"] == "mlp"].set_index("arm")
    assert mlp.loc["synthetic", "rank"] == 1
    assert mlp.loc["original", "rank"] == 2


def test_rank_by_rmse_p90_lower_is_better():
    ranked = build_ranking(_summary_df(), "RMSE_P90")
    lstm = ranked[ranked["pipeline"] == "lstm"].set_index("arm")
    # original (4.74) tem RMSE_P90 menor que synthetic (4.82) -> rank 1.
    assert lstm.loc["original", "rank"] == 1
    assert lstm.loc["synthetic", "rank"] == 2


def test_rank_by_bias_p90_uses_absolute_value():
    ranked = build_ranking(_summary_df(), "Bias_P90")
    lstm = ranked[ranked["pipeline"] == "lstm"].set_index("arm")
    # |synthetic Bias_P90|=1.65 < |original Bias_P90|=2.86 -> synthetic rank 1,
    # mesmo synthetic tendo R2/RMSE_P90 piores — é exatamente o caso real que
    # motivou o eixo TSTR (uma métrica isolada não conta a história toda).
    assert lstm.loc["synthetic", "rank"] == 1
    assert lstm.loc["original", "rank"] == 2


def test_ranking_preserves_all_rows_and_columns():
    summary = _summary_df()
    ranked = build_ranking(summary, "R2")
    assert len(ranked) == len(summary)
    assert set(summary.columns) <= set(ranked.columns)
    assert "rank" in ranked.columns
