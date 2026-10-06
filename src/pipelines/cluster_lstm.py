"""Pipeline LSTM por cluster (YAML-driven, schema v2).

Modelo: Input(T, F) → LSTM(96) → Dropout(0.3) → Dense(1), Huber + Adam, alvo =
rajada máxima diária em m/s. Dados diários (`ClusterPreprocessor`) ou horários
(`build_hourly_batch`) via `data.resolution`; split por blocos de mês (teste =
Jan/Abr/Jul/Out de todos os anos) — ver src/pipeline/data/splits.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.pipelines.metrics_schema import (
    build_predictions_frame, build_station_predictions_frame, CORE_PREDICTIONS_COLUMNS,
)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _print_final_summary(results_df, data, preds):
    """Resumo enriquecido após avaliação: global, por season, extremos.
    `preds` (m/s) já vem calculado pelo chamador (`_predict_lstm_experts`)."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    from src.pipeline.data.target import inverse_target

    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)

    if results_df.empty or "n_samples" not in results_df.columns:
        print(
            "\n[AVISO] Nenhuma métrica válida — resultados vazios. "
            "Verifique se os modelos treinados cobrem os clusters/seasons do teste."
        )
        print("=" * 60)
        return

    w = results_df["n_samples"]
    if w.sum() == 0:
        print("\n[AVISO] Nenhuma amostra válida para métricas ponderadas.")
        print("=" * 60)
        return

    rmse_g = float(np.average(results_df["RMSE"], weights=w))
    corr_g = float(np.average(results_df["Corr"].fillna(0), weights=w))
    print(
        f"\nGlobal (ponderado):  "
        f"RMSE={rmse_g:.3f}  Corr={corr_g:.3f}  N={int(w.sum())}"
    )

    print("\nPor Estacao Climatica:")

    def _wagg(g):
        wts = g["n_samples"]
        if wts.sum() == 0:
            return pd.Series({"N": 0, "RMSE": float("nan"), "Corr": float("nan")})
        return pd.Series({
            "N": int(wts.sum()),
            "RMSE": round(float(np.average(g["RMSE"], weights=wts)), 3),
            "Corr": round(float(np.average(g["Corr"].fillna(0), weights=wts)), 3),
        })

    season_agg = results_df.groupby("season").apply(_wagg).reset_index()
    print(season_agg.to_string(index=False))

    ranked = results_df.sort_values("RMSE")
    cols = ["cluster_id", "season", "n_samples", "R2", "RMSE", "Bias_P90", "Corr"]
    print("\nTop-3 melhores (menor RMSE):")
    print(ranked.head(3)[cols].to_string(index=False))
    print("\nTop-3 piores (maior RMSE):")
    print(ranked.tail(3)[cols].to_string(index=False))

    y_true_all, y_pred_all = [], []
    for season in data.x_test:
        if season not in preds or len(data.y_test.get(season, [])) == 0:
            continue
        y_t = inverse_target(data.scaler_y, data.y_test[season])
        y_p = preds[season]
        n = min(len(y_t), len(y_p))
        y_true_all.append(y_t[:n])
        y_pred_all.append(y_p[:n])

    if y_true_all:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        valid = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        y_true_all = y_true_all[valid]
        y_pred_all = y_pred_all[valid]
        if len(y_true_all) > 0:
            thr_ext = float(np.percentile(y_true_all, 90))
            mask_ext = y_true_all > thr_ext
            if mask_ext.sum() > 0:
                rmse_ext = float(np.sqrt(mean_squared_error(
                    y_true_all[mask_ext], y_pred_all[mask_ext]
                )))
                print(
                    f"\nRMSE extremos (y > P90={thr_ext:.1f} m/s): "
                    f"{rmse_ext:.3f}  N_ext={int(mask_ext.sum())}"
                )

    print("=" * 60)


def preprocess_lstm_data(
    config: str,
    feature_groups: str | None = None,
    exp_name_override: str | None = None,
):
    """Carrega/valida o YAML, fixa a semente, constrói o `ClusterDataBatch`
    (fonte diária ou horária, ver `build_lstm_batch`).

    Retorna `(data, resolved_exp_name, cfg)` — quem for treinar deve
    reconstruir o `ArtifactManager` com esse nome explícito, nunca `None` de
    novo (senão resolve pra um `expN+1` novo).
    """
    from src.pipeline.data.lstm_sources import build_lstm_batch, validate_lstm_config
    from src.utils.artifact_manager import ArtifactManager
    from src.utils.seeding import set_global_seed

    cfg = validate_lstm_config(_load_config(config))

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    if feature_groups is not None:
        data_cfg["feature_groups"] = feature_groups

    set_global_seed(int(exp_cfg.get("seed", 42)), deterministic=bool(exp_cfg.get("deterministic", False)))

    exp_name = exp_name_override or exp_cfg.get("name")
    manager = ArtifactManager(exp_cfg["output_dir"], exp_name)
    print(f"\n=== Experimento: {exp_name} ===")
    print(f"Saída: {manager.root}\n")

    data = build_lstm_batch(cfg)

    return data, manager.exp_dir.name, cfg


def _predict_lstm_experts(data, result, trainer) -> dict:
    """Prediz (m/s) uma única vez; reusado por avaliação, resumo e CSVs."""
    return trainer.predict(
        x_dict=data.x_test,
        models=result.models,
        scaler_y=data.scaler_y,
        feature_names=data.feature_names,
        cluster_ids=data.cluster_ids,
    )


def _window_spec(data):
    from src.pipeline.data.windowing import WindowSpec

    offset = int((data.hourly or {}).get("day_offset_hours", 0))
    return WindowSpec(data.resolution, int(data.window_length), offset)


def train_lstm_and_finalize(
    data, resolved_exp_name: str, cfg: dict, config: str,
    ablation_group: str | None = None,
) -> None:
    """Treina, avalia, serializa modelos + `dl_metadata.joblib` (schema v2) e
    exporta predições/gráficos no MESMO diretório resolvido por
    `preprocess_lstm_data`."""
    from src.inference.dl_metadata import METADATA_FILENAME, MODEL_NAME, build_dl_metadata

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}
    viz_cfg = cfg.get("visualization", {})
    feature_groups = data_cfg.get("feature_groups")
    seed = int(exp_cfg.get("seed", 42))

    from src.utils.artifact_manager import ArtifactManager
    manager = ArtifactManager(exp_cfg["output_dir"], exp_name=resolved_exp_name)
    output_dir = manager.root

    model_params = dict(model_cfg.get("params") or {})

    from src.pipeline.training import cluster_trainer
    trainer = cluster_trainer.ClusterTrainer(
        units=model_params.get("units", 96),
        dropout=model_params.get("dropout", 0.3),
        huber_delta=model_params.get("huber_delta", 1.0),
        learning_rate=model_params.get("learning_rate", 0.001),
        epochs=train_cfg.get("epochs", 50),
        batch_size=train_cfg.get("batch_size", 64),
        patience=train_cfg.get("patience", 5),
        min_samples=train_cfg.get("min_samples", 10),
        seed=seed,
    )

    print("\n[training] Iniciando treinamento (1 LSTM por cluster)...")
    result = trainer.fit(data)

    histories_path = manager.get_root_path("histories.json")
    json_histories = {
        str(cluster_id): {
            key: {k: [float(v) for v in vals] for k, vals in hist.items()}
            for key, hist in season_dict.items()
        }
        for cluster_id, season_dict in result.histories.items()
    }
    histories_path.write_text(json.dumps(json_histories, indent=2))
    print(f"\nHistórico salvo: {histories_path}")

    from src.pipeline.validation.cluster_metrics import ClusterMetricsEvaluator

    print("\n[eval] Avaliando no conjunto de teste (Jan/Abr/Jul/Out)...")
    preds = _predict_lstm_experts(data, result, trainer)
    results_df = ClusterMetricsEvaluator().evaluate(data, result, preds)
    print(results_df.to_string(index=False))

    results_path = manager.get_partial_path("csv", "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados salvos: {results_path}")

    unified_results_df = results_df.copy()
    unified_results_df["pipeline"] = "lstm"
    unified_results_df["experiment"] = manager.exp_dir.name
    unified_results_df["split"] = "test"
    unified_results_path = manager.write_results(unified_results_df)
    print(f"Resultados unificados salvos: {unified_results_path}")

    best_per_cluster = (
        unified_results_df.loc[
            unified_results_df.groupby("cluster_id")["R2"].idxmax()
        ].to_dict("records")
        if not unified_results_df.empty else []
    )
    meta = {
        "model_name": MODEL_NAME,
        "model_params": model_params,
        "resolution": data.resolution,
        "window": _window_spec(data).to_dict(),
        "split": data.split.to_dict(),
        "interp_method": data.interp_method,
        "climatology": data.climatology,
        "hourly": data.hourly,
        "seed": seed,
        "feature_groups": feature_groups,
        "active_features": data.feature_names,
        "ablation_group": ablation_group,
        "best_per_cluster": best_per_cluster,
    }
    meta_path = manager.write_run_meta(meta)
    print(f"Metadados salvos: {meta_path}")
    if not unified_results_df.empty:
        index_path = manager.append_experiments_index(unified_results_df)
        print(f"Índice atualizado: {index_path}")

    _print_final_summary(results_df, data, preds)

    # ── Serializar modelos + metadados v2 para inferência ────────────────
    import joblib
    models_dir = manager.get_model_dir()
    joblib.dump(
        build_dl_metadata(
            scaler_x=data.scaler_x,
            scaler_y=data.scaler_y,
            feature_names=data.feature_names,
            window=_window_spec(data),
            split=data.split,
            interp_method=data.interp_method,
            climatology=data.climatology,
            hourly=data.hourly,
            model_params=model_params,
            seed=seed,
            target_var=data_cfg.get("target_var", "daily_wind_gust_max"),
        ),
        models_dir / METADATA_FILENAME,
    )

    n_saved = 0
    for cid, season_models in result.models.items():
        if not season_models:
            continue  # cluster sem dados suficientes — nada a salvar
        model_to_save = season_models.get("global")
        if model_to_save is None:
            model_to_save = next(iter(season_models.values()))
        model_to_save.save(models_dir / f"best_model_c{cid}.keras")
        n_saved += 1
    print(f"\n[training] {n_saved} modelos salvos em {models_dir}")

    import pandas as pd
    from src.pipeline.validation.cluster_metrics import (
        get_cluster_season_arrays, get_cluster_season_station_arrays,
    )

    pred_frames = []
    station_pred_frames = []
    for season in data.x_test:
        for cluster_id in data.cluster_ids:
            arrays = get_cluster_season_arrays(data, preds, season, cluster_id)
            if arrays is None:
                continue
            yt, yp = arrays
            pred_frames.append(build_predictions_frame(
                pipeline="lstm", experiment=manager.exp_dir.name,
                cluster_id=cluster_id, season=season, split="test",
                y_true=yt, y_pred=yp,
            ))

            station_arrays = get_cluster_season_station_arrays(data, preds, season, cluster_id)
            if station_arrays is not None:
                yt_s, yp_s, estacao_s, lat_s, lon_s, time_s = station_arrays
                station_pred_frames.append(build_station_predictions_frame(
                    estacao=estacao_s, latitude=lat_s, longitude=lon_s, time=time_s,
                    cluster_id=cluster_id, split="test",
                    y_true=yt_s, y_pred=yp_s,
                ))

    predictions_path = manager.get_prediction_path("predictions.csv")
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(predictions_path, index=False)
    else:
        pd.DataFrame(columns=CORE_PREDICTIONS_COLUMNS).to_csv(predictions_path, index=False)
    print(f"Predições salvas: {predictions_path}")

    station_predictions_path = manager.get_prediction_path("predictions_by_station.csv")
    if station_pred_frames:
        pd.concat(station_pred_frames, ignore_index=True).to_csv(
            station_predictions_path, index=False
        )
        print(f"Predições por estação salvas: {station_predictions_path}")

    if viz_cfg.get("enabled", True):
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization.cluster_plots import (
            plot_cluster_diagnostics,
            plot_learning_curves,
        )

        print("\n[viz] Gerando gráficos...")
        plot_cluster_diagnostics(
            data, result, trainer, save_dir=str(manager.get_plot_dir("diagnostics"))
        )
        plot_learning_curves(
            result.histories,
            test_losses=getattr(result, "test_losses", None),
            save_dir=str(manager.get_plot_dir("learning_curves")),
        )

    manager.get_root_path("config.yaml").write_text(Path(config).read_text())
    print(f"\n=== Pipeline concluído. Artefatos em: {output_dir} ===")


def run(
    config: str,
    feature_groups: str | None = None,
    ablation_group: str | None = None,
    exp_name_override: str | None = None,
    **_,
) -> None:
    """Executa a LSTM por cluster via YAML — composição de
    `preprocess_lstm_data` + `train_lstm_and_finalize`."""
    data, resolved_exp_name, cfg = preprocess_lstm_data(
        config, feature_groups, exp_name_override,
    )
    train_lstm_and_finalize(
        data, resolved_exp_name, cfg, config, ablation_group,
    )
