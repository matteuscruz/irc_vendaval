"""Conjunto de avaliação comum do seletor de vencedores
(best_model_selector.derive_common_eval_metrics / eval_set="common").

lazy reporta teste no ano inteiro e a LSTM v2 só em Jan/Abr/Jul/Out (sem os
primeiros dias de cada mês, purgados pela janela) — as métricas nativas não
são comparáveis; as do conjunto comum sim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.creation.best_model_selector import (
    COMMON_SPLIT,
    PIPELINE_ROOTS,
    build_winner_table,
    derive_common_eval_metrics,
    discover_combos,
)

DATES = pd.date_range("2019-01-01", "2024-12-31", freq="D")
STATIONS = ["A001", "A002"]


def _write_combo(root: Path, pipeline: str, arm: str, preds: pd.DataFrame,
                 results_rows: list[dict], suffix: str) -> None:
    d = root / PIPELINE_ROOTS[pipeline] / arm
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_csv(d / "predictions" / "predictions_by_station.csv", index=False)
    pd.DataFrame(results_rows).to_csv(d / "results.csv", index=False)
    (d / "fitted_models").mkdir(exist_ok=True)
    (d / "fitted_models" / f"best_model_c1.{suffix}").write_bytes(b"")


def _preds(dates, noise: float, seed: int, split_fn) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([STATIONS, dates], names=["estacao", "time"]).to_frame(index=False)
    truth = 10 + 3 * np.sin(np.arange(len(idx)) / 5.0)
    idx["cluster_id"] = 1
    idx["y_true"] = truth
    idx["y_pred"] = truth + rng.normal(0, noise, len(idx))
    idx["split"] = split_fn(idx["time"])
    return idx


def _row(pipeline, arm, season, r2):
    return {
        "pipeline": pipeline, "experiment": arm, "cluster_id": 1, "season": season,
        "split": "test", "n_samples": 100, "R2": r2, "RMSE": 1.0,
        "Bias": 0.0, "Bias_P90": 0.0, "RMSE_P90": 1.0,
    }


def _build(root: Path) -> None:
    # lazy: teste 2020+ no ano inteiro, predição ruim; results.csv nativo diz R2=0.99
    lazy = _preds(DATES, noise=3.0, seed=0,
                  split_fn=lambda t: np.where(t.dt.year >= 2020, "test", "val"))
    _write_combo(root, "lazy", "original", lazy,
                 [_row("lazy", "original", s, 0.99) for s in ("ALL", "DJF", "MAM", "JJA", "SON")],
                 "joblib")
    # lstm v2: teste só Jan/Abr/Jul/Out de todos os anos, dias 1-6 purgados; predição boa
    lstm_dates = DATES[DATES.month.isin([1, 4, 7, 10]) & (DATES.day > 6)]
    lstm = _preds(lstm_dates, noise=0.3, seed=1, split_fn=lambda t: np.full(len(t), "test"))
    _write_combo(root, "lstm", "original", lstm,
                 [_row("lstm", "original", s, 0.10) for s in ("DJF", "MAM", "JJA", "SON")],
                 "keras")


def test_common_metrics_same_slice_for_every_combo(tmp_path):
    _build(tmp_path)
    common = derive_common_eval_metrics(discover_combos(tmp_path))

    assert set(common["split"]) == {COMMON_SPLIT}
    assert set(common["season"]) == {"ALL", "DJF", "MAM", "JJA", "SON"}
    n = common.pivot_table(index="season", columns="pipeline", values="n_samples")
    # mesmas amostras para os dois: só 2020-2024, meses 1/4/7/10, sem 2019
    # (split "val" do lazy) e sem os dias 1-6 que a LSTM não tem
    assert (n["lazy"] == n["lstm"]).all()
    days_per_month = {1: 25, 4: 24, 7: 25, 10: 25}
    assert n.loc["ALL", "lstm"] == len(STATIONS) * 5 * sum(days_per_month.values())
    assert n.loc["DJF", "lstm"] == len(STATIONS) * 5 * days_per_month[1]


def test_winner_uses_common_set_by_default_and_native_on_request(tmp_path):
    _build(tmp_path)
    table, _ = build_winner_table(tmp_path, metric="R2")
    quarterly = table[table["season"] != "ALL"]
    assert (quarterly["pipeline"] == "lstm").all()
    assert (table["eval_set"] == "common").all()

    native, _ = build_winner_table(tmp_path, metric="R2", eval_set="native")
    assert (native[native["season"] != "ALL"]["pipeline"] == "lazy").all()
    assert (native["eval_set"] == "native").all()
