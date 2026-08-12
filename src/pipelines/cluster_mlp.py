"""Pipeline MLPRegressor por cluster com extreme-weighting."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

from src.data.netcdf_loader import NetCDFLoader
from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_climatology
from src.pipelines.common import (
    TARGET_VAR, ERA5_GUST_PROXY, BASE_FEATURES, RANDOM_STATE,
    TRAIN_SLICE, VAL_SLICE, TEST_SLICE,
    build_flat_dataframe, make_split,
    parse_cluster_merge, apply_cluster_merge,
    preprocess_df, compute_metrics, resolve_feature_groups,
    restrict_to_feature_coverage,
)
from src.pipelines.metrics_schema import build_results_row, build_predictions_frame
from src.pipeline.validation.station_holdout import iter_holdout_folds
from src.utils.artifact_manager import ArtifactManager

warnings.filterwarnings("ignore")


# ── Metadados de experimento ──────────────────────────────────────────────────

def _write_experiment_meta(
    manager,
    cluster_merge,
    merge_groups,
    hidden_layers,
    alpha,
    extreme_power,
    results_df,
    unified_results_df,
    synthetic_csv=None,
    max_iter: int = 500,
    feature_groups: str = "original,era5_18z,bt55",
    active_features: list[str] | None = None,
    ablation_group: str | None = None,
) -> None:
    index_cols = [
        c for c in [
            "cluster_id", "n_stations",
            "MLP_R2", "MLP_RMSE", "MLP_Bias_P90", "ERA5_Bias_P90",
        ]
        if c in results_df.columns
    ]
    meta = {
        "cluster_merge": cluster_merge,
        "merge_groups": merge_groups,
        "hidden_layers": list(hidden_layers),
        "alpha": alpha,
        "extreme_power": extreme_power,
        "max_iter": max_iter,
        "synthetic_csv": synthetic_csv,
        "features": BASE_FEATURES,
        "feature_groups": feature_groups,
        "active_features": active_features,
        "ablation_group": ablation_group,
        "train_slice": list(TRAIN_SLICE),
        "val_slice": list(VAL_SLICE),
        "test_slice": list(TEST_SLICE),
        "per_cluster": results_df[index_cols].to_dict("records"),
    }
    meta_path = manager.write_run_meta(meta)
    print(f"[mlp_clusters] Metadados salvos: {meta_path}")

    extra_cols = [c for c in index_cols if c != "cluster_id"]
    extra_df = results_df[["cluster_id"] + extra_cols].copy()
    extra_df["cluster_id"] = extra_df["cluster_id"].astype(str)
    index_rows = unified_results_df.merge(extra_df, on="cluster_id", how="left")
    index_rows.insert(2, "cluster_merge", cluster_merge or "")
    index_path = manager.append_experiments_index(index_rows)
    print(f"[mlp_clusters] Índice atualizado: {index_path}")


def _plot_train_distribution(data: list[dict], manager, plt) -> None:
    if not data:
        return
    n = len(data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False)
    for i, d in enumerate(data):
        ax = axes[i // ncols][i % ncols]
        y_real = d["y_real"]
        y_synth = d["y_synth"]
        all_vals = np.concatenate([y_real, y_synth] if y_synth.size else [y_real])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)
        ax.hist(y_real, bins=bins, density=True, alpha=0.6, color="#1f77b4",
                label=f"Real (n={y_real.size})")
        if y_synth.size:
            ax.hist(y_synth, bins=bins, density=True, alpha=0.6, color="#d62728",
                    label=f"Sintético (n={y_synth.size})")
            p90 = np.quantile(y_real, 0.90)
            ax.axvline(p90, color="black", linestyle="--", linewidth=1,
                       label=f"P90 real ({p90:.1f})")
        ax.set_title(f"Cluster {d['cluster_id']}", fontsize=11)
        ax.set_xlabel("Rajada máx. (m/s)")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    plt.suptitle("Distribuição do alvo no treino — real vs sintético", fontsize=13, y=1.02)
    plt.tight_layout()
    path = manager.get_plot_path("summary", "train_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[mlp_clusters] Distribuição de treino salva: {path}")


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    output_dir: str = "artifacts/mlp_clusters",
    hidden_layers: tuple = (128, 64),
    alpha: float = 0.001,
    extreme_power: float = 2.0,
    max_iter: int = 500,
    cluster_merge = None,
    synthetic_csv = None,
    exp_name = None,
    feature_groups: str = "original,era5_18z,bt55",
    restrict_coverage: bool = False,
    ablation_group: str | None = None,
    validation_mode: str = "temporal",
    spatial_n_folds: int = 5,
    spatial_seed: int = 42,
    **_,
) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    import shap

    manager = ArtifactManager(output_dir, exp_name)
    out = manager.root
    cluster_plots_dir = manager.get_plot_dir("clusters")
    print(f"[mlp_clusters] Experimento: {out}")

    active_features = resolve_feature_groups(feature_groups)
    if "wind_mag_max" not in active_features:
        raise ValueError(
            "feature_groups precisa incluir 'original' — 'wind_mag_max' é "
            "usado incondicionalmente no plot de fator de correção ERA5."
        )
    print(f"[mlp_clusters] Grupos de features: {feature_groups} ({len(active_features)} features)")

    synth_df = None
    if synthetic_csv:
        synth_df = pd.read_csv(synthetic_csv)
        print(f"[mlp_clusters] Augment: {len(synth_df)} linhas sintéticas de {synthetic_csv}")

    print("[mlp_clusters] Carregando NetCDF...")
    ds_inmet, ds_era5 = NetCDFLoader(raw_dir).load_extended()
    print(f"  Estações: {len(ds_inmet.estacao.values)} | ERA5 features: {list(ds_era5.data_vars)}")

    print("[mlp_clusters] Atribuindo clusters...")
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    print(station_clusters.to_string(index=False))

    print("[mlp_clusters] Calculando climatologia ERA5...")
    ds_clim = get_climatology(ds_era5, ERA5_GUST_PROXY, slice(*TRAIN_SLICE))

    print("[mlp_clusters] Construindo DataFrame de features...")
    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim)
    print(f"  Shape total: {df.shape}")
    if restrict_coverage:
        df = restrict_to_feature_coverage(df, feature_groups)

    merge_groups = parse_cluster_merge(cluster_merge)
    if merge_groups:
        df = apply_cluster_merge(df, merge_groups)
        station_clusters = apply_cluster_merge(station_clusters, merge_groups)
        labels = sorted("-".join(str(m) for m in sorted(g)) for g in merge_groups)
        print(f"[mlp_clusters] Agregando clusters: {', '.join(labels)}")

    stations_latlon = (
        ds_inmet[["latitude", "longitude"]].to_dataframe()
        .groupby("estacao").first().reset_index()
    )
    stations_meta = station_clusters.merge(stations_latlon, on="estacao", how="left")
    stations_meta.to_csv(manager.get_partial_path("csv", "stations_metadata.csv"), index=False)
    print(f"  Estações: {len(stations_meta)}")

    all_rows, all_importances, all_preds, fig_list = [], [], [], []
    unified_rows, unified_pred_rows = [], []
    shap_means, train_dist = {}, []
    spatial_holdout_preds = []

    if validation_mode not in ("temporal", "spatial-kfold", "loocv"):
        raise ValueError(
            f"validation_mode inválido: {validation_mode!r} "
            "(use 'temporal', 'spatial-kfold' ou 'loocv')"
        )

    for cid, group in df.groupby("cluster_id"):
        try:
            n_stations = group["estacao"].nunique()
            print(f"── Cluster {cid} | {n_stations} estação(ões) ──")

            df_tr = make_split(group, TRAIN_SLICE)
            df_vl = make_split(group, VAL_SLICE)
            df_te = make_split(group, TEST_SLICE)

            if df_tr.empty or df_vl.empty:
                print(f"   Sem dados — pulando cluster {cid}.")
                continue

            # Filtra apenas features disponíveis no DataFrame (ERA5-18UTC/BT55 podem
            # estar ausentes se load_extended falhou silenciosamente)
            avail_features = [f for f in active_features if f in df_tr.columns and df_tr[f].notna().any()]
            missing = [f for f in active_features if f not in df_tr.columns]
            if missing:
                print(f"   [AVISO] {len(missing)} features ausentes (p.ex. _18z/bt55): {missing[:5]}{'...' if len(missing) > 5 else ''}")

            x_train = df_tr[avail_features].reset_index(drop=True)
            y_train_abs = df_tr[TARGET_VAR].reset_index(drop=True).values
            era5_train = df_tr[ERA5_GUST_PROXY].reset_index(drop=True).values
            x_val = df_vl[avail_features]
            y_val = df_vl[TARGET_VAR].reset_index(drop=True).values
            era5_val = df_vl[ERA5_GUST_PROXY].reset_index(drop=True).values
            x_test = df_te[avail_features] if not df_te.empty else None
            y_test = df_te[TARGET_VAR].reset_index(drop=True).values if not df_te.empty else None
            era5_test = df_te[ERA5_GUST_PROXY].reset_index(drop=True).values if not df_te.empty else None

            y_real_train = np.asarray(y_train_abs, dtype=float).copy()
            y_synth_train = np.array([], dtype=float)

            if synth_df is not None:
                s = synth_df[synth_df["cluster_id"] == cid]
                if len(s):
                    y_synth_train = s[TARGET_VAR].to_numpy(float)
                    x_train = pd.concat(
                        [x_train, s.reindex(columns=avail_features).reset_index(drop=True)],
                        ignore_index=True,
                    )
                    y_train_abs = np.concatenate([y_train_abs, s[TARGET_VAR].to_numpy(float)])
                    era5_train = np.concatenate([era5_train, s[ERA5_GUST_PROXY].to_numpy(float)])
                    print(f"   +{len(s)} linhas sintéticas (augment)")

            train_dist.append({"cluster_id": cid, "y_real": y_real_train, "y_synth": y_synth_train})

            era5_train_safe = np.clip(era5_train, 0.1, None)
            y_train_ratio = y_train_abs / era5_train_safe

            x_train_sc, x_val_sc = preprocess_df(x_train, x_val)
            print(f"   Treino: {len(x_train_sc)} | Val: {len(x_val_sc)} | Features: {x_train_sc.shape[1]}")

            w = (y_train_abs / y_train_abs.max()) ** extreme_power
            w = w / w.mean()
            rng = np.random.default_rng(RANDOM_STATE)
            repeat_counts = np.round(w).astype(int).clip(1)
            idx_os = np.repeat(np.arange(len(x_train_sc)), repeat_counts)
            idx_os = rng.permutation(idx_os)
            x_train_os = x_train_sc.values[idx_os]
            y_train_os = y_train_ratio[idx_os]

            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=alpha,
                max_iter=max_iter,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                learning_rate="adaptive",
                n_iter_no_change=30,
            )
            mlp.fit(x_train_os, y_train_os)

            # ── Serializar o modelo MLP para inferência espacial ────────────────
            import joblib
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import RobustScaler
            models_dir = out / "fitted_models"
            models_dir.mkdir(exist_ok=True)
            _imputer = SimpleImputer(strategy="mean").fit(x_train)
            _scaler = RobustScaler().fit(_imputer.transform(x_train))
            artifact = {
                "model": mlp,
                "model_name": "MLPRegressor",
                "imputer": _imputer,
                "scaler": _scaler,
                "features": list(avail_features),
                "cluster_id": cid,
                # MLP prevê a razão sobre o ERA5 (ratio * wind_mag_max), não
                # o valor absoluto — usado por SpatialCorrector pra decidir
                # a reconstrução correta (dispatch explícito em vez de
                # inferir só pelo nome do modelo).
                "target_kind": "ratio",
            }
            joblib.dump(artifact, models_dir / f"best_model_c{cid}.joblib")

            era5_train_sc_safe = np.clip(era5_train_safe[:len(x_train_sc)], 0.1, None)
            ratio_pred_train = mlp.predict(x_train_sc)
            y_pred_train = ratio_pred_train * era5_train_sc_safe
            y_true_train = y_train_ratio[:len(x_train_sc)] * era5_train_sc_safe

            era5_val_safe = np.clip(era5_val, 0.1, None)
            ratio_pred = mlp.predict(x_val_sc)
            y_pred = ratio_pred * era5_val_safe

            y_pred_test = metrics_test = metrics_era5_test = None
            if x_test is not None:
                _, x_test_sc = preprocess_df(x_train, x_test)
                era5_test_safe = np.clip(era5_test, 0.1, None)
                y_pred_test = mlp.predict(x_test_sc) * era5_test_safe
                metrics_test = compute_metrics(y_test, y_pred_test)
                metrics_era5_test = compute_metrics(y_test, era5_test)

            preds_chunk = df_vl[["estacao", "time", "latitude", "longitude"]].reset_index(drop=True).copy()
            preds_chunk["cluster_id"] = cid
            preds_chunk["split"] = "val"
            preds_chunk["y_true"] = y_val
            preds_chunk["y_pred"] = y_pred
            preds_chunk["era5_wind_mag_max"] = era5_val
            preds_chunk["ratio_pred"] = ratio_pred
            preds_chunk["ratio_true"] = y_val / np.clip(era5_val, 0.1, None)
            all_preds.append(preds_chunk)

            if y_pred_test is not None:
                preds_chunk_test = df_te[["estacao", "time", "latitude", "longitude"]].reset_index(drop=True).copy()
                preds_chunk_test["cluster_id"] = cid
                preds_chunk_test["split"] = "test"
                preds_chunk_test["y_true"] = y_test
                preds_chunk_test["y_pred"] = y_pred_test
                preds_chunk_test["era5_wind_mag_max"] = era5_test
                preds_chunk_test["ratio_pred"] = y_pred_test / np.clip(era5_test, 0.1, None)
                preds_chunk_test["ratio_true"] = y_test / np.clip(era5_test, 0.1, None)
                all_preds.append(preds_chunk_test)

            metrics_mlp = compute_metrics(y_val, y_pred)
            metrics_era5 = compute_metrics(y_val, era5_val)

            row = {
                "cluster_id": cid,
                "n_stations": n_stations,
                "n_train": len(df_tr),
                "n_val": len(df_vl),
                "n_test": len(df_te),
                **{f"MLP_{k}": v for k, v in metrics_mlp.items()},
                **{f"ERA5_{k}": v for k, v in metrics_era5.items()},
                **(
                    {f"MLP_test_{k}": v for k, v in metrics_test.items()}
                    if metrics_test else {}
                ),
                **(
                    {f"ERA5_test_{k}": v for k, v in metrics_era5_test.items()}
                    if metrics_era5_test else {}
                ),
            }
            all_rows.append(row)

            unified_rows.append(build_results_row(
                pipeline="mlp", experiment=manager.exp_dir.name, cluster_id=cid,
                season="ALL", split="val", y_true=y_val, y_pred=y_pred,
                n_samples=len(y_val), extra={"n_stations": n_stations},
            ))
            unified_pred_rows.append(build_predictions_frame(
                pipeline="mlp", experiment=manager.exp_dir.name, cluster_id=cid,
                season="ALL", split="val", y_true=y_val, y_pred=y_pred,
            ))
            if metrics_test is not None:
                unified_rows.append(build_results_row(
                    pipeline="mlp", experiment=manager.exp_dir.name, cluster_id=cid,
                    season="ALL", split="test", y_true=y_test, y_pred=y_pred_test,
                    n_samples=len(y_test), extra={"n_stations": n_stations},
                ))
                unified_pred_rows.append(build_predictions_frame(
                    pipeline="mlp", experiment=manager.exp_dir.name, cluster_id=cid,
                    season="ALL", split="test", y_true=y_test, y_pred=y_pred_test,
                ))

            print(f"   MLP  val  → R²={metrics_mlp['R2']:.3f}  RMSE={metrics_mlp['RMSE']:.3f}  Bias@P90={metrics_mlp['Bias_P90']:.3f}")
            if metrics_test:
                print(f"   MLP  test → R²={metrics_test['R2']:.3f}  RMSE={metrics_test['RMSE']:.3f}  Bias@P90={metrics_test['Bias_P90']:.3f}")
            print(f"   ERA5 → R²={metrics_era5['R2']:.3f}  RMSE={metrics_era5['RMSE']:.3f}  Bias@P90={metrics_era5['Bias_P90']:.3f}")

            # ── Validação cruzada espacial (station-holdout), opcional ──────────
            # Aditivo: só roda se validation_mode != "temporal". Gera linhas
            # split="spatial_holdout" em results.csv/predictions.csv, ignoradas
            # por quem filtra split=="test" (scripts/_ablation_common.py), então
            # não altera o comportamento default nem os braços de ablation.
            if validation_mode != "temporal":
                if validation_mode == "loocv" and n_stations > 50:
                    print(
                        f"   [AVISO] cluster {cid} tem {n_stations} estações — "
                        "LOOCV completo pode ser caro; considere --validation-mode spatial-kfold."
                    )
                cluster_holdout_preds = []
                for fold in iter_holdout_folds(
                    group, mode=validation_mode,
                    n_folds=spatial_n_folds, seed=spatial_seed,
                ):
                    fold_avail = [
                        f for f in avail_features
                        if f in fold.df_train.columns and fold.df_train[f].notna().any()
                    ]
                    x_tr_f = fold.df_train[fold_avail].reset_index(drop=True)
                    y_tr_f_abs = fold.df_train[TARGET_VAR].reset_index(drop=True).values
                    era5_tr_f = fold.df_train[ERA5_GUST_PROXY].reset_index(drop=True).values
                    x_ev_f = fold.df_eval[fold_avail].reset_index(drop=True)
                    y_ev_f = fold.df_eval[TARGET_VAR].reset_index(drop=True).values
                    era5_ev_f = fold.df_eval[ERA5_GUST_PROXY].reset_index(drop=True).values

                    if len(x_tr_f) < 20 or len(x_ev_f) == 0:
                        continue

                    era5_tr_f_safe = np.clip(era5_tr_f, 0.1, None)
                    y_tr_f_ratio = y_tr_f_abs / era5_tr_f_safe
                    x_tr_f_sc, x_ev_f_sc = preprocess_df(x_tr_f, x_ev_f)

                    w_f = (y_tr_f_abs / y_tr_f_abs.max()) ** extreme_power
                    w_f = w_f / w_f.mean()
                    rng_f = np.random.default_rng(RANDOM_STATE)
                    repeat_counts_f = np.round(w_f).astype(int).clip(1)
                    idx_os_f = rng_f.permutation(np.repeat(np.arange(len(x_tr_f_sc)), repeat_counts_f))
                    x_tr_f_os = x_tr_f_sc.values[idx_os_f]
                    y_tr_f_os = y_tr_f_ratio[idx_os_f]

                    mlp_f = MLPRegressor(
                        hidden_layer_sizes=hidden_layers, activation="relu", solver="adam",
                        alpha=alpha, max_iter=max_iter, random_state=RANDOM_STATE,
                        early_stopping=True, validation_fraction=0.1,
                        learning_rate="adaptive", n_iter_no_change=30,
                    )
                    mlp_f.fit(x_tr_f_os, y_tr_f_os)

                    era5_ev_f_safe = np.clip(era5_ev_f, 0.1, None)
                    y_pred_f = mlp_f.predict(x_ev_f_sc) * era5_ev_f_safe

                    unified_rows.append(build_results_row(
                        pipeline="mlp", experiment=manager.exp_dir.name, cluster_id=cid,
                        season="ALL", split="spatial_holdout", y_true=y_ev_f, y_pred=y_pred_f,
                        n_samples=len(y_ev_f),
                        extra={
                            "fold_id": fold.fold_id,
                            "validation_mode": validation_mode,
                            "n_stations_held_out": len(fold.held_out_stations),
                        },
                    ))
                    holdout_chunk = fold.df_eval[["estacao", "time"]].reset_index(drop=True).copy()
                    holdout_chunk["cluster_id"] = cid
                    holdout_chunk["fold_id"] = fold.fold_id
                    holdout_chunk["validation_mode"] = validation_mode
                    holdout_chunk["y_true"] = y_ev_f
                    holdout_chunk["y_pred"] = y_pred_f
                    cluster_holdout_preds.append(holdout_chunk)

                if cluster_holdout_preds:
                    _sh_df = pd.concat(cluster_holdout_preds, ignore_index=True)
                    spatial_holdout_preds.append(_sh_df)
                    _sh_metrics = compute_metrics(_sh_df["y_true"].values, _sh_df["y_pred"].values)
                    msg = (
                        f"   MLP  spatial-holdout ({validation_mode}) → "
                        f"R²={_sh_metrics['R2']:.3f}  RMSE={_sh_metrics['RMSE']:.3f}  "
                        f"Bias@P90={_sh_metrics['Bias_P90']:.3f}"
                    )
                    if metrics_test:
                        msg += f"  (vs. temporal test RMSE={metrics_test['RMSE']:.3f})"
                    print(msg)

            p90 = np.percentile(y_val, 90)
            mask_ext = y_val >= p90

            wmax_col = avail_features.index("wind_mag_max")
            x_median = x_train_sc.median().values
            wmax_sc_range = np.linspace(x_val_sc.iloc[:, wmax_col].min(), x_val_sc.iloc[:, wmax_col].max(), 200)
            x_sweep = np.tile(x_median, (200, 1))
            x_sweep[:, wmax_col] = wmax_sc_range
            ratio_sweep = mlp.predict(x_sweep)

            wmax_raw_range = np.linspace(x_val["wind_mag_max"].min(), x_val["wind_mag_max"].max(), 200)
            y_val_ratio_plot = y_val / np.clip(era5_val, 0.1, None)
            era5_val_for_plot = era5_val

            # Loss curve
            fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
            ax_loss.plot(mlp.loss_curve_, label="Treino", color="steelblue")
            if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
                val_epochs = np.linspace(0, len(mlp.loss_curve_) - 1, len(mlp.validation_scores_))
                ax_loss_r = ax_loss.twinx()
                ax_loss_r.plot(val_epochs, mlp.validation_scores_, label="Val R²", color="crimson", ls="--")
                ax_loss_r.set_ylabel("Val R²", color="crimson", fontsize=9)
                ax_loss_r.tick_params(axis="y", labelcolor="crimson")
                ax_loss_r.legend(loc="lower right", fontsize=8)
            ax_loss.set_xlabel("Época")
            ax_loss.set_ylabel("Loss (treino)", color="steelblue", fontsize=9)
            ax_loss.tick_params(axis="y", labelcolor="steelblue")
            ax_loss.set_title(f"Cluster {cid} — Curva de perda ({mlp.n_iter_} épocas, hidden={hidden_layers})", fontsize=10)
            ax_loss.legend(loc="upper right", fontsize=8)
            fig_loss.tight_layout()
            loss_path = cluster_plots_dir / f"loss_curve_{cid}.png"
            fig_loss.savefig(loss_path, dpi=150, bbox_inches="tight")
            fig_list.insert(0, fig_loss)
            plt.close(fig_loss)

            # Scatter obs×pred
            p90_train = np.percentile(y_true_train, 90)
            mask_tr = y_true_train >= p90_train
            metrics_train = compute_metrics(y_true_train, y_pred_train)

            has_test = y_pred_test is not None
            ncols_sc = 3 if has_test else 2
            fig_obs, axes_obs = plt.subplots(1, ncols_sc, figsize=(5.5 * ncols_sc, 5))
            fig_obs.suptitle(f"Cluster {cid} ({n_stations} est.) — Observado × Predito", fontsize=11)

            def _scatter_panel(ax, y_true, y_pred_p, mask, p90_v, r2, bias, label):
                ax.scatter(y_true[~mask], y_pred_p[~mask], alpha=0.3, s=10, color="steelblue", label="< P90")
                ax.scatter(y_true[mask], y_pred_p[mask], alpha=0.6, s=16, color="crimson", label=f"≥ P90 ({p90_v:.1f} m/s)")
                lo = min(y_true.min(), y_pred_p.min())
                hi = max(y_true.max(), y_pred_p.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1")
                ax.axvline(p90_v, color="gray", ls="--", lw=0.8)
                ax.set_xlabel("Observado (m/s)")
                ax.set_ylabel("Predito (m/s)")
                ax.set_title(f"{label}\nR²={r2:.3f}  Bias@P90={bias:.2f}", fontsize=10)
                ax.legend(fontsize=8)
                ax.set_aspect("equal", adjustable="box")

            _scatter_panel(axes_obs[0], y_true_train, y_pred_train, mask_tr, p90_train, metrics_train["R2"], metrics_train["Bias_P90"], "Treino")
            _scatter_panel(axes_obs[1], y_val, y_pred, mask_ext, p90, metrics_mlp["R2"], metrics_mlp["Bias_P90"], "Validação")
            if has_test:
                p90_te = np.percentile(y_test, 90)
                mask_te = y_test >= p90_te
                _scatter_panel(axes_obs[2], y_test, y_pred_test, mask_te, p90_te, metrics_test["R2"], metrics_test["Bias_P90"], "Teste")
            fig_obs.tight_layout()
            obs_path = cluster_plots_dir / f"scatter_obs_pred_{cid}.png"
            fig_obs.savefig(obs_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_obs)
            plt.close(fig_obs)

            # Correction ratio
            fig_resp, ax2 = plt.subplots(figsize=(5, 5))
            ax2.scatter(era5_val_for_plot[~mask_ext], y_val_ratio_plot[~mask_ext], alpha=0.25, s=10, color="steelblue", label="obs < P90")
            ax2.scatter(era5_val_for_plot[mask_ext], y_val_ratio_plot[mask_ext], alpha=0.5, s=16, color="crimson", label="obs ≥ P90")
            ax2.plot(wmax_raw_range, ratio_sweep, color="darkorange", lw=2.5, label="razão predita pelo modelo")
            ax2.axhline(1.0, color="black", ls="--", lw=0.8, label="razão=1 (sem correção)")
            ax2.axvline(x_val["wind_mag_max"].quantile(0.9), color="gray", ls=":", lw=0.8, label="ERA5 P90")
            ax2.set_xlabel("ERA5 wind_mag_max (m/s)")
            ax2.set_ylabel("Razão INMET / ERA5 wind_mag_max")
            ax2.set_title(f"Cluster {cid} — Fator de correção aprendido", fontsize=10)
            ax2.legend(fontsize=8)
            fig_resp.tight_layout()
            resp_path = cluster_plots_dir / f"correction_ratio_{cid}.png"
            fig_resp.savefig(resp_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_resp)
            plt.close(fig_resp)

            # Permutation importance
            y_val_ratio = y_val / np.clip(era5_val, 0.1, None)
            perm = permutation_importance(mlp, x_val_sc, y_val_ratio, n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1)
            imp_df = pd.DataFrame({
                "feature": avail_features,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
                "cluster_id": cid,
            }).sort_values("importance", ascending=False)
            all_rows[-1]["features_ranked"] = "|".join(imp_df["feature"].tolist())

            imp_df_sorted = imp_df.sort_values("importance")
            fig_imp, ax_imp = plt.subplots(figsize=(7, max(4, len(avail_features) * 0.35)))
            ax_imp.barh(imp_df_sorted["feature"], imp_df_sorted["importance"], xerr=imp_df_sorted["std"], color="steelblue", ecolor="gray", capsize=3)
            ax_imp.axvline(0, color="black", lw=0.8)
            ax_imp.set_xlabel("Queda em R² (permutação)")
            ax_imp.set_title(f"Cluster {cid} — Importância por Permutação", fontsize=11)
            fig_imp.tight_layout()
            imp_path = cluster_plots_dir / f"feature_importance_{cid}.png"
            fig_imp.savefig(imp_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_imp)
            plt.close(fig_imp)
            all_importances.append(imp_df)

            # SHAP
            background = x_train_sc.sample(min(50, len(x_train_sc)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(mlp.predict, background)
            x_val_shap = x_val_sc.sample(min(150, len(x_val_sc)), random_state=RANDOM_STATE)
            shap_vals = explainer.shap_values(x_val_shap, silent=True)
            shap_means[cid] = pd.Series(np.abs(shap_vals).mean(axis=0), index=avail_features)

            shap.summary_plot(shap_vals, x_val_shap, plot_type="dot", show=False)
            fig_bee = plt.gcf()
            fig_bee.suptitle(f"Cluster {cid} — SHAP Beeswarm", fontsize=11, y=1.01)
            bee_path = cluster_plots_dir / f"shap_beeswarm_{cid}.png"
            fig_bee.savefig(bee_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_bee)
            plt.close(fig_bee)

            shap_bar_vals = pd.Series(np.abs(shap_vals).mean(axis=0), index=avail_features).sort_values()
            fig_shap_bar, ax_shap_bar = plt.subplots(figsize=(7, max(4, len(avail_features) * 0.35)))
            ax_shap_bar.barh(shap_bar_vals.index, shap_bar_vals.values, color="darkorange")
            ax_shap_bar.set_xlabel("|SHAP| médio")
            ax_shap_bar.set_title(f"Cluster {cid} — SHAP Importância Global", fontsize=11)
            fig_shap_bar.tight_layout()
            shap_bar_path = cluster_plots_dir / f"shap_bar_{cid}.png"
            fig_shap_bar.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_shap_bar)
            plt.close(fig_shap_bar)

            top_feature = avail_features[np.abs(shap_vals).mean(axis=0).argmax()]
            shap.dependence_plot(top_feature, shap_vals, x_val_shap, show=False)
            fig_dep = plt.gcf()
            fig_dep.suptitle(f"Cluster {cid} — SHAP Dependence: {top_feature}", fontsize=11, y=1.01)
            dep_path = cluster_plots_dir / f"shap_dependence_{cid}.png"
            fig_dep.savefig(dep_path, dpi=150, bbox_inches="tight")
            fig_list.append(fig_dep)
            plt.close(fig_dep)
        except Exception as e:
            print(f"[mlp_clusters] ERRO no cluster {cid} — pulando cluster. Detalhe: {e!r}")
            import traceback
            traceback.print_exc()
            continue

    if all_preds:
        preds_all = pd.concat(all_preds, ignore_index=True)
        preds_all.to_csv(manager.get_prediction_path("predictions_by_station.csv"), index=False)

    if spatial_holdout_preds:
        holdout_all = pd.concat(spatial_holdout_preds, ignore_index=True)
        holdout_all.to_csv(manager.get_prediction_path("spatial_holdout_predictions.csv"), index=False)
        print(
            f"[mlp_clusters] Predições de station-holdout salvas: "
            f"{manager.get_prediction_path('spatial_holdout_predictions.csv')}"
        )

    results_df = pd.DataFrame(all_rows)
    csv_path = manager.get_partial_path("csv", "mlp_cluster_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[mlp_clusters] Resultados salvos: {csv_path}")

    unified_results_df = pd.DataFrame(unified_rows)
    unified_results_path = manager.write_results(unified_results_df)
    print(f"[mlp_clusters] Resultados unificados salvos: {unified_results_path}")
    if unified_pred_rows:
        unified_pred_path = manager.get_prediction_path("predictions.csv")
        pd.concat(unified_pred_rows, ignore_index=True).to_csv(unified_pred_path, index=False)
        print(f"[mlp_clusters] Predições unificadas salvas: {unified_pred_path}")

    _plot_train_distribution(train_dist, manager, plt)

    metrics_to_plot = ["RMSE", "RMSE_P90", "Bias_P90"]
    clusters = results_df["cluster_id"].tolist()
    x = np.arange(len(clusters))
    width = 0.35
    for metric in metrics_to_plot:
        fig_m, ax_m = plt.subplots(figsize=(max(5, len(clusters) * 0.9), 4))
        ax_m.bar(x - width / 2, results_df[f"MLP_{metric}"].values, width, label="MLP", color="steelblue")
        ax_m.bar(x + width / 2, results_df[f"ERA5_{metric}"].values, width, label="ERA5", color="tomato")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels([f"C{c}" for c in clusters])
        ax_m.set_title(f"MLP vs ERA5 — {metric}", fontsize=12)
        ax_m.axhline(0, color="black", lw=0.5)
        ax_m.legend(fontsize=9)
        fig_m.tight_layout()
        cmp_path = manager.get_plot_path("summary", f"metrics_{metric.lower()}.png")
        fig_m.savefig(cmp_path, dpi=150, bbox_inches="tight")
        fig_list.append(fig_m)
        plt.close(fig_m)

    if all_importances:
        imp_all = pd.concat(all_importances, ignore_index=True)
        pivot_imp = imp_all.pivot_table(index="feature", columns="cluster_id", values="importance")
        pivot_imp = pivot_imp.loc[pivot_imp.mean(axis=1).sort_values(ascending=False).index]
        fig_imp_h, ax_h = plt.subplots(figsize=(max(6, pivot_imp.shape[1] * 1.5), len(BASE_FEATURES) * 0.4 + 2))
        sns.heatmap(pivot_imp, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.4, ax=ax_h)
        ax_h.set_title("Permutation Importance — Queda em R² (Feature × Cluster)", fontsize=12, pad=10)
        ax_h.set_xlabel("Cluster ID")
        ax_h.set_ylabel("")
        fig_imp_h.tight_layout()
        imp_heat_path = manager.get_plot_path("summary", "feature_importance_summary.png")
        fig_imp_h.savefig(imp_heat_path, dpi=150, bbox_inches="tight")
        fig_list.append(fig_imp_h)
        plt.close(fig_imp_h)
        imp_all.to_csv(manager.get_partial_path("csv", "feature_importance.csv"), index=False)

    if shap_means:
        shap_pivot = pd.DataFrame(shap_means)
        _shap_features = shap_pivot.index.tolist()
        shap_pivot = shap_pivot.loc[shap_pivot.mean(axis=1).sort_values(ascending=False).index]
        fig_shap_h, ax_sh = plt.subplots(figsize=(max(6, shap_pivot.shape[1] * 1.5), len(_shap_features) * 0.4 + 2))
        sns.heatmap(shap_pivot, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.4, ax=ax_sh)
        ax_sh.set_title("SHAP — |Valor SHAP| Médio (Feature × Cluster)", fontsize=12, pad=10)
        ax_sh.set_xlabel("Cluster ID")
        ax_sh.set_ylabel("")
        fig_shap_h.tight_layout()
        shap_heat_path = manager.get_plot_path("summary", "shap_summary.png")
        fig_shap_h.savefig(shap_heat_path, dpi=150, bbox_inches="tight")
        fig_list.append(fig_shap_h)
        plt.close(fig_shap_h)

    # PDF report
    fig_cover = plt.figure(figsize=(11.7, 8.3))
    fig_cover.text(0.5, 0.91, "IRC Vendaval — MLP Bias Correction Report", ha="center", fontsize=20, fontweight="bold")
    fig_cover.text(0.5, 0.85, f"Treino: {TRAIN_SLICE[0]} → {TRAIN_SLICE[1]}   |   Validação: {VAL_SLICE[0]} → {VAL_SLICE[1]}", ha="center", fontsize=12, color="gray")
    tbl_cols = ["Cluster", "Estações", "R²", "RMSE", "Bias", "Bias@P90", "ERA5 Bias@P90"]
    tbl_data = results_df[["cluster_id", "n_stations", "MLP_R2", "MLP_RMSE", "MLP_Bias", "MLP_Bias_P90", "ERA5_Bias_P90"]].round(3).values.tolist()
    ax_tbl = fig_cover.add_axes([0.04, 0.08, 0.92, 0.68])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=tbl_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    pdf_path = manager.get_plot_path("summary", "cluster_report.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)
        for fig in fig_list:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.infodict()["Title"] = "IRC Vendaval — MLP Bias Correction"
        pdf.infodict()["Author"] = "cluster_mlp"
    print(f"[mlp_clusters] Relatório PDF salvo: {pdf_path}")

    _write_experiment_meta(
        manager=manager, cluster_merge=cluster_merge,
        merge_groups=merge_groups, hidden_layers=hidden_layers,
        alpha=alpha, extreme_power=extreme_power, results_df=results_df,
        unified_results_df=unified_results_df,
        synthetic_csv=synthetic_csv, max_iter=max_iter,
        feature_groups=feature_groups, active_features=active_features,
        ablation_group=ablation_group,
    )

    return results_df
