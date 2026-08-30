"""Experimento isolado: pipeline diária vs. horária, restrito ao Cluster 3.

Treina o mesmo modelo já identificado como melhor para o Cluster 3
(`cluster_lstm` dual-head, arm "basin") uma vez com os dados diários de
produção e outra vez com o dataset ERA5 horário novo
(`dataset/raw/test_cluster_3_hourly.nc`), com os mesmos hiperparâmetros nos
dois braços — a única diferença controlada é a fonte/resolução dos dados.

Não modifica nenhum arquivo de `src/` — só importa e reaproveita
`ClusterPreprocessor`, `ClusterTrainer`, `ClusterMetricsEvaluator` como estão.

Uso
---
    python -m experiments.cluster3_hourly_vs_daily.run_experiment --arm both
    python -m experiments.cluster3_hourly_vs_daily.run_experiment --arm hourly --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.data.cluster_preprocessor import ClusterPreprocessor, SEASONS
from src.pipeline.training.cluster_trainer import ClusterTrainer
from src.pipeline.validation.cluster_metrics import (
    ClusterMetricsEvaluator,
    get_cluster_season_station_arrays,
)

from experiments.cluster3_hourly_vs_daily.hourly_windower import (
    TARGET_CLUSTER,
    build_hourly_cluster3_batch,
)
from experiments.cluster3_hourly_vs_daily.hourly_flat_features import (
    build_hourly_cluster3_flat_table,
    flat_feature_columns,
)

# Hiperparâmetros do modelo já identificado como melhor para o Cluster 3
# (arm "basin", config/experiment_cluster_lstm_modal.yaml) — idênticos nos
# dois braços de propósito.
LSTM_PARAMS = dict(
    units=64, dropout=0.4, l2_reg=0.01, learning_rate=0.001,
    weight_normal=0.7, weight_extreme=0.3,
    extreme_weight=20.0, extreme_threshold=1.5,
)
TRAIN_PARAMS = dict(epochs=50, batch_size=64, patience=5, min_samples=10)
SMOKE_TRAIN_PARAMS = dict(epochs=2, batch_size=8, patience=1, min_samples=5)


def _cluster3_masked_stats(data) -> dict:
    """Amostras/bytes efetivamente usados no treino do Cluster 3 (após a
    máscara one-hot que ClusterTrainer aplica) — base justa de comparação de
    custo entre os dois braços, já que o braço diário monta janelas de todos
    os 14 clusters antes de mascarar."""
    col_idx = data.feature_names.index(f"cluster_{TARGET_CLUSTER}")
    n_samples: dict[str, int] = {}
    total_bytes = 0
    for split in ("x_train", "x_val", "x_test"):
        n = 0
        for arr in getattr(data, split).values():
            if len(arr) == 0:
                continue
            mask = arr[:, -1, col_idx] == 1
            n += int(mask.sum())
            total_bytes += arr[mask].nbytes
        n_samples[split] = n
    return {"n_samples": n_samples, "input_tensor_bytes": total_bytes}


def _save_predictions(data, result, trainer, output_dir: Path) -> None:
    preds = trainer.predict(
        x_dict=data.x_test, models=result.models, era5_dict=data.era5_test,
        scaler_y=data.scaler_y, feature_names=data.feature_names,
        cluster_ids=data.cluster_ids,
    )
    rows = []
    for season in SEASONS:
        for cluster_id in data.cluster_ids:
            arrays = get_cluster_season_station_arrays(data, preds, season, cluster_id)
            if arrays is None:
                continue
            yt, yp, est, lat, lon, time_ = arrays
            for i in range(len(yt)):
                rows.append({
                    "season": season, "cluster_id": cluster_id, "estacao": est[i],
                    "latitude": lat[i], "longitude": lon[i], "time": time_[i],
                    "y_true": float(yt[i]), "y_pred": float(yp[i]),
                })
    pd.DataFrame(rows).to_csv(output_dir / "predictions_by_station.csv", index=False)


def _run_arm(arm: str, data, output_dir: Path, train_params: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = ClusterTrainer(**LSTM_PARAMS, **train_params)

    t0 = time.perf_counter()
    result = trainer.fit(data)
    elapsed = time.perf_counter() - t0

    metrics = ClusterMetricsEvaluator().evaluate(data, result, trainer)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _save_predictions(data, result, trainer, output_dir)

    non_empty = next((v for v in data.x_train.values() if len(v)), None)
    lookback = int(non_empty.shape[1]) if non_empty is not None else None

    timing = {
        "arm": arm,
        "elapsed_seconds": elapsed,
        "n_features": len(data.feature_names),
        "lookback": lookback,
        "train_params": train_params,
        "epochs_ran": {
            str(cid): len(hist.get("global", {}).get("loss", []))
            for cid, hist in result.histories.items()
        },
        **_cluster3_masked_stats(data),
    }
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=2, default=str))
    print(f"[{arm}] concluído em {elapsed:.1f}s — metrics/timing salvos em {output_dir}")


def run_daily_arm(
    output_dir: str,
    smoke: bool = False,
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
) -> None:
    pre = ClusterPreprocessor(
        raw_dir=raw_dir, shp_dir=shp_dir,
        feature_groups="original,era5_basin", lookback=7,
    )
    data = pre.run()
    data.cluster_ids = [TARGET_CLUSTER]
    train_params = SMOKE_TRAIN_PARAMS if smoke else TRAIN_PARAMS
    _run_arm("daily", data, Path(output_dir), train_params)


def run_hourly_arm(
    output_dir: str,
    smoke: bool = False,
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
) -> None:
    kwargs: dict = dict(raw_dir=raw_dir, shp_dir=shp_dir)
    if smoke:
        kwargs.update(stations_limit=3, max_days_per_station=60)
    data = build_hourly_cluster3_batch(**kwargs)
    train_params = SMOKE_TRAIN_PARAMS if smoke else TRAIN_PARAMS
    _run_arm("hourly", data, Path(output_dir), train_params)


def run_daily_lazy_arm(
    output_dir: str,
    smoke: bool = False,
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
) -> None:
    """Roda `cluster_lazy` (LazyPredict, ~40 modelos sklearn) restrito ao
    Cluster 3, mesmo arm de features do braço diário do LSTM
    (`feature_groups="original,era5_basin"`), pra comparação justa entre
    famílias de modelo. `n_neighbor_clusters=0` — sem contaminação de
    estações de outros clusters, mesma disciplina de isolamento do resto do
    experimento."""
    from src.pipelines import cluster_lazy

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lazy_root = str(out / "_cluster_lazy_run")
    exp_name = "lazy_daily"

    t0 = time.perf_counter()
    cluster_lazy.run(
        raw_dir=raw_dir, shp_dir=shp_dir, output_dir=lazy_root,
        exp_name=exp_name, feature_groups="original,era5_basin",
        stratify_seasons=True, cluster_id=TARGET_CLUSTER,
        n_neighbor_clusters=0,
    )
    elapsed = time.perf_counter() - t0

    exp_dir = Path(lazy_root) / exp_name
    unified_dir = exp_dir / "_partial" / "unified_results"
    pred_dir = exp_dir / "_partial" / "unified_predictions"

    rows = []
    for season_label, slug_season in [("ALL", ""), *[(s, f"_{s}") for s in SEASONS]]:
        f = unified_dir / f"unified_c{TARGET_CLUSTER}{slug_season}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        test_row = df[df["split"] == "test"]
        if not len(test_row):
            continue
        row = test_row.iloc[0].to_dict()
        row["season"] = season_label

        pred_f = pred_dir / f"unified_pred_c{TARGET_CLUSTER}{slug_season}.csv"
        if pred_f.exists():
            preds_df = pd.read_csv(pred_f)
            preds_df = preds_df[preds_df["split"] == "test"]
            if len(preds_df) > 1:
                row["Corr"] = round(
                    float(np.corrcoef(preds_df["y_true"], preds_df["y_pred"])[0, 1]), 3
                )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    from src.pipelines.cluster_lazy import _FAST_REGRESSORS
    from src.pipelines.common import resolve_feature_groups

    timing = {
        "arm": "daily_lazy",
        "elapsed_seconds": elapsed,
        "n_features": len(resolve_feature_groups("original,era5_basin")) + 1,  # + gust_P50
        "train_params": {"n_regressors_screened": len(_FAST_REGRESSORS)},
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2, default=str))
    print(f"[daily_lazy] concluído em {elapsed:.1f}s — metrics/timing salvos em {out}")


def run_hourly_lazy_arm(
    output_dir: str,
    smoke: bool = False,
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
) -> None:
    """LazyPredict com as 12 variáveis horárias agregadas (mean/max/min/std
    sobre os 7 dias anteriores) em vez das features diárias — mesmo
    princípio isolado do braço hourly do LSTM, adaptado pra features flat."""
    from lazypredict.Supervised import LazyRegressor

    from src.pipelines.cluster_lazy import _FAST_REGRESSORS
    from src.pipelines.common import RANDOM_STATE, TARGET_VAR, compute_metrics, preprocess_df

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    kwargs: dict = dict(raw_dir=raw_dir, shp_dir=shp_dir)
    if smoke:
        kwargs.update(stations_limit=3, max_days_per_station=60)

    t0 = time.perf_counter()
    df = build_hourly_cluster3_flat_table(**kwargs)
    train_features = flat_feature_columns()

    rows = []
    for season in [None, *SEASONS]:
        df_season = df if season is None else df[df["season"] == season]
        df_tr = df_season[df_season["split"] == "train"]
        df_vl = df_season[df_season["split"] == "val"]
        df_te = df_season[df_season["split"] == "test"]
        if df_tr.empty or df_vl.empty:
            continue

        x_train = df_tr[train_features].reset_index(drop=True)
        y_train = df_tr[TARGET_VAR].reset_index(drop=True)
        x_val = df_vl[train_features].reset_index(drop=True)
        y_val = df_vl[TARGET_VAR].reset_index(drop=True)

        x_train_pp, x_val_pp = preprocess_df(x_train, x_val)
        reg = LazyRegressor(
            verbose=0, ignore_warnings=True, predictions=True,
            random_state=RANDOM_STATE, regressors=_FAST_REGRESSORS,
        )
        scores, _ = reg.fit(x_train_pp, x_val_pp, y_train, y_val)
        scores = scores.reset_index().rename(columns={"index": "Model"})
        season_label = season or "ALL"
        scores.to_csv(out / f"lazy_scores_{season_label}.csv", index=False)

        if df_te.empty:
            continue
        best_model = scores.iloc[scores["R-Squared"].argmax()]["Model"]
        x_test = df_te[train_features].reset_index(drop=True)
        y_test = df_te[TARGET_VAR].reset_index(drop=True)
        _, x_test_pp = preprocess_df(x_train, x_test)

        fitted = reg.provide_models(x_train_pp, x_val_pp, y_train, y_val)
        if best_model not in fitted:
            continue
        y_pred_test = np.asarray(fitted[best_model].predict(x_test_pp), dtype=float)
        core = compute_metrics(y_test.to_numpy(float), y_pred_test)
        corr = float(np.corrcoef(y_test, y_pred_test)[0, 1]) if len(y_test) > 1 else float("nan")
        rows.append({
            "cluster_id": TARGET_CLUSTER, "season": season_label,
            "n_samples": len(y_test), "R2": round(core["R2"], 4),
            "RMSE": round(core["RMSE"], 3), "Bias": round(core["Bias"], 3),
            "Bias_P90": round(core["Bias_P90"], 3), "RMSE_P90": round(core["RMSE_P90"], 3),
            "Corr": round(corr, 3), "model": best_model,
        })

    elapsed = time.perf_counter() - t0

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    timing = {
        "arm": "hourly_lazy",
        "elapsed_seconds": elapsed,
        "n_features": len(train_features),
        "n_samples": {
            "x_train": int((df["split"] == "train").sum()),
            "x_val": int((df["split"] == "val").sum()),
            "x_test": int((df["split"] == "test").sum()),
        },
        "train_params": {"n_regressors_screened": len(_FAST_REGRESSORS)},
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2, default=str))
    print(f"[hourly_lazy] concluído em {elapsed:.1f}s — metrics/timing salvos em {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=["daily", "hourly", "daily_lazy", "hourly_lazy", "both", "all"],
        default="both",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/experiments/cluster3_hourly_vs_daily"
    )
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--shp-dir", default="dataset/shp")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Config minúscula p/ validar localmente, sem gastar Modal nem "
             "estressar a GPU local (poucas estações/dias, poucas épocas).",
    )
    args = parser.parse_args()

    if args.arm in ("daily", "both", "all"):
        run_daily_arm(
            f"{args.output_dir}/daily", smoke=args.smoke,
            raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        )
    if args.arm in ("hourly", "both", "all"):
        run_hourly_arm(
            f"{args.output_dir}/hourly", smoke=args.smoke,
            raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        )
    if args.arm in ("daily_lazy", "all"):
        run_daily_lazy_arm(
            f"{args.output_dir}/daily_lazy", smoke=args.smoke,
            raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        )
    if args.arm in ("hourly_lazy", "all"):
        run_hourly_lazy_arm(
            f"{args.output_dir}/hourly_lazy", smoke=args.smoke,
            raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        )


if __name__ == "__main__":
    main()
