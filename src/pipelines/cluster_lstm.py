"""Pipeline LSTM dual-head por cluster (YAML-driven)."""
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


def _print_final_summary(results_df, data, result, trainer, aug_cfg=None):
    """Resumo enriquecido após avaliação: global, por season, extremos."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from src.pipeline.training.cluster_tr_trainer import ClusterTRTrainer

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
        print(
            "\n[AVISO] Nenhuma amostra válida para métricas ponderadas."
        )
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
            return pd.Series({
                "N": 0,
                "RMSE": float("nan"),
                "Corr": float("nan"),
            })
        return pd.Series({
            "N": int(wts.sum()),
            "RMSE": round(float(np.average(g["RMSE"], weights=wts)), 3),
            "Corr": round(
                float(np.average(g["Corr"].fillna(0), weights=wts)), 3
            ),
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
    if isinstance(trainer, ClusterTRTrainer):
        preds = trainer.predict(
            x_dict=data.x_test,
            xs_dict=data.x_static_test,
            models=result.models,
            era5_dict=data.era5_test,
            scaler_y=data.scaler_y,
            feature_names=data.feature_names,
            cluster_ids=data.cluster_ids,
        )
    else:
        preds = trainer.predict(
            x_dict=data.x_test,
            models=result.models,
            era5_dict=data.era5_test,
            scaler_y=data.scaler_y,
            feature_names=data.feature_names,
            cluster_ids=data.cluster_ids,
        )
    for season in data.x_test:
        if season not in preds:
            continue
        era5_safe = np.clip(np.array(data.era5_test[season], dtype="float32"), 0.1, None)
        y_t = (
            data.scaler_y.inverse_transform(
                data.y_test[season].astype("float32")
            ).flatten()
            * era5_safe
        )
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
                rmse_ext = float(
                    np.sqrt(mean_squared_error(
                        y_true_all[mask_ext], y_pred_all[mask_ext]
                    ))
                )
                print(
                    f"\nRMSE extremos (y > P90={thr_ext:.1f} m/s): "
                    f"{rmse_ext:.3f}  N_ext={int(mask_ext.sum())}"
                )

    if aug_cfg and aug_cfg.get("method") not in (None, "none"):
        y_train = np.concatenate([v.flatten() for v in data.y_train.values()])
        p = aug_cfg.get("extreme_percentile", 90)
        thr_aug = float(np.percentile(y_train, p))
        n_ext_aug = int(np.sum(y_train > thr_aug))
        print(
            f"\nAugmentacao: method={aug_cfg['method']}  "
            f"extremos_treino={n_ext_aug}  limiar={thr_aug:.3f}"
        )

    print("=" * 60)


def _inject_synthetic(data, synthetic_csv: str) -> None:
    """
    Injeta sintéticos (cluster_gan.py ou legado) em data.x_train/y_train/era5_train.

    data.x_train/y_train são indexados por SEASON (DJF/MAM/JJA/SON), não por
    cluster_id — o cluster é codificado como coluna one-hot embutida no
    último timestep de cada sequência (feature_names = avail_features +
    cluster_cols). Por isso, para cada cluster sintético precisamos: (1)
    localizar o índice da coluna one-hot desse cluster, (2) mascarar as
    sequências reais de cada season que pertencem a esse cluster, (3) fazer
    nearest-neighbor em y ABSOLUTO (reconstituído via scaler_y.inverse_transform
    * valor ERA5 da própria amostra) — não no y escalado, que não é
    comparável a rajadas em m/s.

    Para cada alvo sintético (y absoluto) de um cluster, busca a sequência
    real de treino desse cluster/season com y absoluto mais próximo e reusa
    (X, era5) dessa amostra, recalculando a razão/escala do alvo sintético
    a partir do valor ERA5 emprestado — preserva features realistas e
    ancoragem ERA5 consistente com o que o resto do pipeline espera.
    """
    import numpy as np
    import pandas as pd
    from src.pipeline.data.cluster_preprocessor import SEASONS

    synth = pd.read_csv(synthetic_csv)
    target_col = (
        "daily_wind_gust_max" if "daily_wind_gust_max" in synth.columns
        else "rajada_sintetica"
    )
    if target_col not in synth.columns:
        raise ValueError(
            f"{synthetic_csv} precisa ter coluna 'daily_wind_gust_max' "
            "(cluster_gan.py) ou 'rajada_sintetica' (legado)."
        )
    if "cluster_id" not in synth.columns:
        raise ValueError(f"{synthetic_csv} precisa ter coluna 'cluster_id'.")

    n_cluster_cols = len(data.cluster_ids)
    n_base = len(data.feature_names) - n_cluster_cols
    col_idx_by_cluster = {str(cid): n_base + i for i, cid in enumerate(data.cluster_ids)}
    has_season_col = "season" in synth.columns
    season_list = list(SEASONS)
    rng = np.random.default_rng(42)

    added = 0
    for cid_raw, grp in synth.groupby("cluster_id"):
        col_idx = col_idx_by_cluster.get(str(cid_raw))
        if col_idx is None:
            continue
        y_synth_abs = grp[target_col].to_numpy(float)
        season_synth = grp["season"].to_numpy() if has_season_col else None

        counts = {
            s: int(np.sum(data.x_train[s][:, -1, col_idx] == 1))
            for s in season_list if s in data.x_train and len(data.x_train[s])
        }
        total = sum(counts.values())
        if total == 0:
            continue

        if has_season_col:
            season_assign = season_synth
        else:
            avail_seasons = [s for s in counts if counts[s] > 0]
            probs = [counts[s] / total for s in avail_seasons]
            season_assign = rng.choice(avail_seasons, size=len(y_synth_abs), p=probs)

        for season in season_list:
            if season not in data.x_train or len(data.x_train[season]) == 0:
                continue
            y_s = y_synth_abs[season_assign == season]
            if len(y_s) == 0:
                continue

            mask_real = data.x_train[season][:, -1, col_idx] == 1
            if not np.any(mask_real):
                continue

            x_real = data.x_train[season][mask_real]
            y_real_scaled = data.y_train[season][mask_real]
            era5_real = data.era5_train[season][mask_real]
            era5_real_safe = np.clip(era5_real, 0.1, None)
            y_real_abs = (
                data.scaler_y.inverse_transform(y_real_scaled).flatten() * era5_real_safe
            )

            order = np.argsort(y_real_abs)
            idx = np.searchsorted(y_real_abs[order], y_s)
            idx = np.clip(idx, 0, len(y_real_abs) - 1)

            x_new = x_real[order[idx]]
            era5_new = era5_real[order[idx]]
            ratio_new = (y_s / np.clip(era5_new, 0.1, None)).reshape(-1, 1).astype("float32")
            y_new_scaled = data.scaler_y.transform(ratio_new)

            data.x_train[season] = np.concatenate([data.x_train[season], x_new], axis=0)
            data.y_train[season] = np.concatenate([data.y_train[season], y_new_scaled], axis=0)
            data.era5_train[season] = np.concatenate([data.era5_train[season], era5_new], axis=0)
            # NOTA: data.meta_train[season] (estacao/lat/lon/time por janela)
            # não é estendido aqui — fica desalinhado com x_train/y_train/
            # era5_train após a injeção. Inofensivo hoje (só meta_test é lido,
            # pra montar predictions_by_station.csv a partir do teste, que
            # nunca recebe sintéticos); só usar meta_train no futuro depois
            # de estendê-lo aqui também.
            added += len(y_s)

    print(
        f"[synthetic] +{added} pares (X_real, y_sintético)"
        f" injetados de {synthetic_csv}"
    )


def run(
    config: str,
    augmentation_method: str | None = None,
    synthetic_csv: str | None = None,
    feature_groups: str | None = None,
    restrict_coverage: bool = False,
    ablation_group: str | None = None,
    exp_name_override: str | None = None,
    **_,
) -> None:
    """Executa o pipeline dual-head LSTM por cluster via YAML config."""
    cfg = _load_config(config)

    if augmentation_method and "augmentation" in cfg:
        cfg["augmentation"]["method"] = augmentation_method

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    viz_cfg = cfg.get("visualization", {})

    # CLI sobrepõe o YAML (mesmo padrão de augmentation_method acima) —
    # feature_groups=None mantém o que estiver em data.feature_groups
    # (ou "all" se ausente, via resolve_feature_groups).
    if feature_groups is not None:
        data_cfg["feature_groups"] = feature_groups
    feature_groups = data_cfg.get("feature_groups")

    exp_name = exp_name_override or exp_cfg.get("name")

    from src.utils.artifact_manager import ArtifactManager
    manager = ArtifactManager(exp_cfg["output_dir"], exp_name)
    output_dir = manager.root

    print(f"\n=== Experimento: {exp_name} ===")
    print(f"Saída: {output_dir}\n")

    from src.pipeline.data.cluster_preprocessor import ClusterPreprocessor

    preprocessor = ClusterPreprocessor(
        raw_dir=data_cfg["raw_dir"],
        shp_dir=data_cfg["shp_dir"],
        target_var=data_cfg["target_var"],
        train_slice=tuple(data_cfg["train_slice"]),
        val_slice=tuple(data_cfg["val_slice"]),
        test_slice=tuple(data_cfg["test_slice"]),
        test_station_fraction=data_cfg["test_station_fraction"],
        lookback=prep_cfg["lookback"],
        seed=exp_cfg.get("seed", 42),
        feature_groups=feature_groups,
        restrict_coverage=restrict_coverage,
    )
    data = preprocessor.run()

    if synthetic_csv:
        _inject_synthetic(data, synthetic_csv)

    model_params = model_cfg.get("params", {})
    model_name = model_cfg.get("name", "cluster_dual_head_lstm")

    if model_name == "cluster_tr_lstm":
        from src.pipeline.training.cluster_tr_trainer import ClusterTRTrainer

        trainer = ClusterTRTrainer(
            units=model_params.get("units", 64),
            static_hidden=model_params.get("static_hidden", 32),
            dropout=model_params.get("dropout", 0.4),
            dropout_static=model_params.get("dropout_static", 0.05),
            recurrent_dropout=model_params.get("recurrent_dropout", 0.0),
            l2_reg=model_params.get("l2_reg", 0.01),
            learning_rate=model_params.get("learning_rate", 0.001),
            huber_delta=model_params.get("huber_delta", 1.5),
            weight_normal=model_params.get("weight_normal", 0.7),
            weight_extreme=model_params.get("weight_extreme", 0.3),
            extreme_weight=model_params.get("extreme_weight", 20.0),
            extreme_threshold=model_params.get("extreme_threshold", 1.5),
            epochs=train_cfg.get("epochs", 50),
            batch_size=train_cfg.get("batch_size", 64),
            patience=train_cfg.get("patience", 5),
            min_samples=train_cfg.get("min_samples", 10),
        )
    else:
        from src.pipeline.training.cluster_trainer import ClusterTrainer

        trainer = ClusterTrainer(
            units=model_params.get("units", 64),
            dropout=model_params.get("dropout", 0.4),
            l2_reg=model_params.get("l2_reg", 0.01),
            learning_rate=model_params.get("learning_rate", 0.001),
            weight_normal=model_params.get("weight_normal", 0.7),
            weight_extreme=model_params.get("weight_extreme", 0.3),
            extreme_weight=model_params.get("extreme_weight", 20.0),
            extreme_threshold=model_params.get("extreme_threshold", 1.5),
            epochs=train_cfg.get("epochs", 50),
            batch_size=train_cfg.get("batch_size", 64),
            patience=train_cfg.get("patience", 5),
            min_samples=train_cfg.get("min_samples", 10),
        )

    aug_cfg = cfg.get("augmentation")
    if aug_cfg:
        from src.pipeline.augmentation.factory import augmenter_factory

        # Checkpoint dentro do diretório do próprio experimento — sobrevive
        # a timeout/crash do processo (ver src/pipeline/augmentation/
        # checkpoint.py): um retry do MESMO exp_name retoma o treino do GAN/
        # diffusion em vez de recomeçar do zero, desde que o volume Modal
        # tenha sido commitado com o checkpoint parcial antes do timeout.
        checkpoint_dir = str(output_dir / "_gan_checkpoint")
        augmenter = augmenter_factory(aug_cfg, checkpoint_dir=checkpoint_dir)
        if augmenter is not None:
            print(
                "\n[augmentation] Gerando amostras sinteticas de extremos..."
            )
            data = augmenter.fit_augment(data)

    print("\n[training] Iniciando treinamento dos especialistas...")
    result = trainer.fit(data)

    histories_path = manager.get_root_path("histories.json")
    json_histories: dict = {}
    for cluster_id, season_dict in result.histories.items():
        json_histories[str(cluster_id)] = {
            key: {k: [float(v) for v in vals] for k, vals in hist.items()}
            for key, hist in season_dict.items()
        }
    histories_path.write_text(json.dumps(json_histories, indent=2))
    print(f"\nHistórico salvo: {histories_path}")

    from src.pipeline.validation.cluster_metrics import ClusterMetricsEvaluator

    print("\n[eval] Avaliando no conjunto de teste...")
    evaluator = ClusterMetricsEvaluator()
    results_df = evaluator.evaluate(data, result, trainer)
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
        "model_name": model_name,
        "model_params": model_params,
        "train_slice": list(data_cfg["train_slice"]),
        "val_slice": list(data_cfg["val_slice"]),
        "test_slice": list(data_cfg["test_slice"]),
        "augmentation": aug_cfg,
        "synthetic_csv": synthetic_csv,
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

    _print_final_summary(results_df, data, result, trainer, aug_cfg)

    # ── Serializar Modelos para Inferência Espacial ───────────────────────
    import joblib
    models_dir = manager.get_model_dir()
    
    metadata = {
        "scaler_x": data.scaler_x,
        "scaler_y": data.scaler_y,
        "imputer_x": data.imputer_x,
        "feature_names": data.feature_names,
        "lookback": prep_cfg["lookback"],
        "target_var": data_cfg["target_var"],
        "model_name": model_name,
        # Usado por DLSpatialCorrector pra decidir cabeça normal vs. extrema
        # na inferência espacial — antes hardcoded lá, agora vem do mesmo
        # valor efetivamente usado no treino/predict (ClusterTrainer.predict).
        "extreme_threshold": model_params.get("extreme_threshold", 1.5),
    }
    joblib.dump(metadata, models_dir / "dl_metadata.joblib")
    
    n_saved = 0
    for cid, season_models in result.models.items():
        if not season_models:
            # nenhuma temporada teve dados suficientes pra esse cluster
            # (ex.: cobertura ERA5-18Z/BT55 insuficiente) — nada a salvar.
            continue
        # A inferência espacial usa a predição global (season="global")
        # Se não houver global, tenta DJF, etc, mas o correto é treinar um global
        model_to_save = season_models.get("global") or next(iter(season_models.values()))
        model_path = models_dir / f"best_model_c{cid}.keras"
        model_to_save.save(model_path)
        n_saved += 1
    print(f"\n[training] {n_saved} modelos salvos em {models_dir}")

    import pandas as pd
    from src.pipeline.validation.cluster_metrics import (
        get_cluster_season_arrays, get_cluster_season_station_arrays,
    )

    if model_name == "cluster_tr_lstm":
        preds = trainer.predict(
            x_dict=data.x_test,
            xs_dict=data.x_static_test,
            models=result.models,
            era5_dict=data.era5_test,
            scaler_y=data.scaler_y,
            feature_names=data.feature_names,
            cluster_ids=data.cluster_ids,
        )
    else:
        preds = trainer.predict(
            x_dict=data.x_test,
            models=result.models,
            era5_dict=data.era5_test,
            scaler_y=data.scaler_y,
            feature_names=data.feature_names,
            cluster_ids=data.cluster_ids,
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

    # predictions_by_station.csv — mesma granularidade que cluster_mlp.py já
    # produz (estacao/lat/lon reais, não só o cluster_id agregado acima).
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
