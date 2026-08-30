"""Pipeline LazyPredict por cluster."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lazypredict.Supervised import LazyRegressor, REGRESSORS

# SVR/NuSVR/KernelRidge são O(n²–n³) — inviáveis com datasets grandes (vizinhos)
_SLOW_MODELS = {"SVR", "NuSVR", "KernelRidge", "GaussianProcessRegressor"}
_FAST_REGRESSORS = [c for n, c in REGRESSORS if n not in _SLOW_MODELS]

from src.data.netcdf_loader import NetCDFLoader
from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_climatology
from sklearn.metrics import r2_score as _r2_score

from src.pipelines.common import (
    BASE_FEATURES, TARGET_VAR, RANDOM_STATE,
    TRAIN_SLICE, VAL_SLICE, TEST_SLICE,
    build_flat_dataframe, make_split,
    parse_cluster_merge, apply_cluster_merge,
    preprocess_df, month_to_season, resolve_feature_groups,
    compute_metrics, restrict_to_feature_coverage,
)
from src.pipelines.metrics_schema import (
    CORE_RESULTS_COLUMNS, build_results_row, build_predictions_frame,
    build_station_predictions_frame,
)
from src.pipeline.validation.station_holdout import iter_holdout_folds
from src.utils.artifact_manager import ArtifactManager

warnings.filterwarnings("ignore")


# ── Vizinhança entre clusters ────────────────────────────────────────────────

def _compute_cluster_neighbors(df: pd.DataFrame, n_neighbors: int) -> dict:
    """Retorna {cid: [neighbor_cid, ...]} usando distância euclidiana entre centroides."""
    if n_neighbors <= 0:
        return {}
    centroids = df.groupby("cluster_id")[["latitude", "longitude"]].mean()
    cids = centroids.index.tolist()
    coords = centroids.to_numpy()
    result = {}
    for i, cid in enumerate(cids):
        dists = sorted(
            (float(np.sqrt(((coords[i] - coords[j]) ** 2).sum())), other_cid)
            for j, other_cid in enumerate(cids) if j != i
        )
        result[cid] = [c for _, c in dists[:n_neighbors]]
    return result


# ── Metadados de experimento ──────────────────────────────────────────────────

def _write_experiment_meta(
    manager, cluster_merge, merge_groups, cluster_summary, results_df, unified_results_df,
    synthetic_csv=None, feature_groups: str = "original,era5_18z,bt55",
    active_features: list[str] | None = None, ablation_group: str | None = None,
):
    best = results_df.loc[results_df.groupby("cluster_id")["R-Squared"].idxmax()]
    best_cols = [c for c in ["cluster_id", "n_stations", "Model", "R-Squared", "RMSE"] if c in best.columns]
    best = best[best_cols]

    meta = {
        "cluster_merge": cluster_merge,
        "merge_groups": merge_groups,
        "synthetic_csv": synthetic_csv,
        "features": BASE_FEATURES,
        "feature_groups": feature_groups,
        "active_features": active_features,
        "ablation_group": ablation_group,
        "train_slice": list(TRAIN_SLICE),
        "val_slice": list(VAL_SLICE),
        "clusters": {
            str(cid): {"n_stations": int(row["n_stations"]), "n_samples": int(row["n_samples"])}
            for cid, row in cluster_summary.iterrows()
        },
        "best_per_cluster": best.to_dict("records"),
    }
    meta_path = manager.write_run_meta(meta)
    print(f"[lazy_clusters] Metadados salvos: {meta_path}")

    if not unified_results_df.empty:
        index_rows = unified_results_df.copy()
        index_rows.insert(2, "cluster_merge", cluster_merge or "")
        index_path = manager.append_experiments_index(index_rows)
        print(f"[lazy_clusters] Índice atualizado: {index_path}")


# ── Plots por cluster ─────────────────────────────────────────────────────────

def _evaluate_rolling_windows(
    times: pd.Series,
    y_true: np.ndarray,
    preds_df: pd.DataFrame,
    model_cols: list,
    window: str = "monthly",
) -> pd.DataFrame:
    """R² por janela de deploy para os modelos solicitados."""
    df = pd.DataFrame({"time": pd.to_datetime(times.values), "y_true": y_true})
    for col in model_cols:
        df[col] = preds_df[col].values

    if window == "monthly":
        df["period"] = df["time"].dt.to_period("M").astype(str)
    else:  # biweekly
        df["period"] = df["time"].apply(
            lambda t: f"{t.year}-{t.month:02d}-{'Q1' if t.day <= 15 else 'Q2'}"
        )

    rows = []
    for period, grp in df.groupby("period", sort=True):
        if len(grp) < 5:
            continue
        row = {"period": period, "n_samples": len(grp)}
        for col in model_cols:
            row[col] = round(float(_r2_score(grp["y_true"], grp[col])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_rolling_r2(rolling_df, model_cols, slug, manager, plt):
    if rolling_df.empty:
        return
    periods = rolling_df["period"].tolist()
    x = list(range(len(periods)))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(max(8, len(periods) * 0.75), 4))
    for i, col in enumerate(model_cols):
        if col not in rolling_df.columns:
            continue
        r2s = rolling_df[col].tolist()
        mean_r2 = rolling_df[col].mean()
        ax.plot(x, r2s, marker="o", ms=5, lw=1.5,
                color=palette[i % len(palette)],
                label=f"{col}  (μ={mean_r2:.3f})")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(top=1.02)
    ax.set_title(f"{slug} — R² por janela de deploy ({rolling_df['period'].iloc[0][:7]} …)")
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    fig.savefig(manager.get_plot_path("rolling_r2", f"rolling_r2_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   R² por janela salvo: rolling_r2/rolling_r2_{slug}.png")


def _plot_seasonal_scatter(out_df, scores, slug, manager, plt):
    """Scatter 2×2 por trimestre climático (DJF/MAM/JJA/SON) para o melhor modelo."""
    from src.pipelines.common import SEASONS as _S

    if "season" not in out_df.columns:
        return
    best_model = scores.iloc[scores["R-Squared"].argmax()]["Model"]
    if best_model not in out_df.columns:
        return

    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    y_true = out_df["y_true"]
    y_pred = out_df[best_model]
    g_max = max(y_true.max(), y_pred.max()) if not out_df.empty else 80

    for i, s_name in enumerate(seasons):
        ax = axes[i]
        tr = out_df[(out_df["split"] == "train") & (out_df["season"] == s_name)]
        vl = out_df[(out_df["split"] == "val") & (out_df["season"] == s_name)]

        has_source = "source" in out_df.columns

        if not tr.empty:
            if has_source:
                tr_real = tr[tr["source"] == "real"]
                tr_synth = tr[tr["source"] == "synth"]
                ax.scatter(tr_real["y_true"], tr_real[best_model], alpha=0.3, s=15, color="blue", label="Treino Real")
                if not tr_synth.empty:
                    ax.scatter(tr_synth["y_true"], tr_synth[best_model], alpha=0.3, s=15, color="orange", label="Treino Synth")
            else:
                ax.scatter(tr["y_true"], tr[best_model], alpha=0.3, s=15, color="blue", label="Treino")

        if not vl.empty:
            ax.scatter(vl["y_true"], vl[best_model], alpha=0.6, s=20, color="red", label="Validação")

        ax.plot([0, g_max], [0, g_max], "k--", lw=1)
        ax.set_title(s_name)
        ax.set_xlabel("Observado")
        ax.set_ylabel("Predito")
        ax.set_xlim(0, g_max)
        ax.set_ylim(0, g_max)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"{slug} — {best_model} | Scatter por trimestre", fontsize=11)
    fig.tight_layout()
    fig.savefig(manager.get_plot_path("seasonal_scatter", f"seasonal_scatter_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Scatter sazonal salvo: seasonal_scatter_{slug}.png")


def _plot_cluster_train_dist(d, manager, plt):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(d["y_real"], fill=True, label="Real", ax=ax)
    if len(d["y_synth"]) > 0:
        sns.kdeplot(d["y_synth"], fill=True, label="Sintético (Augment)", ax=ax)
    ax.set_title(f"Distribuição Rajadas Treino - Cluster {d['cluster_id']}")
    ax.set_xlabel("Rajada Máxima (m/s)")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(manager.get_plot_path("train_distribution", f"train_distribution_c{d['cluster_id']}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_top5(scores, slug, manager, plt):
    top5 = scores.head(5).copy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(top5["Model"][::-1], top5["R-Squared"][::-1], color="steelblue")
    ax.set_title(f"Top 5 Modelos - {slug}")
    ax.set_xlabel("R²")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(manager.get_plot_path("top5", f"top5_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_scatter(cp, manager, plt):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cp["y_true"], cp["y_pred"], alpha=0.5, color="blue", label="Validação")
    g_max = max(max(cp["y_true"]), max(cp["y_pred"])) if len(cp["y_true"]) else 80
    ax.plot([0, g_max], [0, g_max], "r--")
    ax.set_title(f"Cluster {cp['cluster_id']} ({cp['n_stations']} estações) - {cp['model']}")
    ax.set_xlabel("Rajada Real (m/s)")
    ax.set_ylabel("Rajada Predita (m/s)")
    ax.set_xlim(0, g_max)
    ax.set_ylim(0, g_max)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(manager.get_plot_path("scatter", f"scatter_c{cp['cluster_id']}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── PDF com todos os modelos ──────────────────────────────────────────────────

def _plot_synth_comparison(scores_aug, scores_base, slug, manager, plt):
    import seaborn as sns
    merged = pd.merge(
        scores_base[["Model", "R-Squared"]],
        scores_aug[["Model", "R-Squared"]],
        on="Model", suffixes=("_base", "_aug")
    ).dropna()
    if merged.empty:
        return

    merged["diff"] = merged["R-Squared_aug"] - merged["R-Squared_base"]
    merged = merged.sort_values("diff", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=merged.head(15), x="diff", y="Model", ax=axes[0], palette="vlag")
    axes[0].set_title(f"Ganho/Perda R² com Sintéticos (Top 15) - {slug}")
    axes[0].set_xlabel("Δ R² (Augment - Base)")
    axes[0].axvline(0, color="k", lw=1)

    axes[1].scatter(merged["R-Squared_base"], merged["R-Squared_aug"], alpha=0.7)
    g_min = min(merged["R-Squared_base"].min(), merged["R-Squared_aug"].min(), 0)
    g_max = max(merged["R-Squared_base"].max(), merged["R-Squared_aug"].max(), 1)
    axes[1].plot([g_min, g_max], [g_min, g_max], "r--")
    axes[1].set_xlabel("R² (Sem Sintéticos)")
    axes[1].set_ylabel("R² (Com Sintéticos)")
    axes[1].set_title("Acima da diagonal = melhora")

    fig.tight_layout()
    path = manager.get_plot_path("synth_comparison", f"synth_comparison_{slug}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Comparação sintéticos salva: {path.name}")


def _plot_all_models_pdf(out_df, scores, slug, manager, plt):
    """Gera um PDF único com o scatter plot de todos os modelos do cluster."""
    from matplotlib.backends.backend_pdf import PdfPages

    model_order = scores.sort_values("R-Squared", ascending=False)["Model"].tolist()
    meta_cols = ["split", "y_true", "season", "cluster_id", "source"]
    model_cols = [c for c in out_df.columns if c not in meta_cols]
    
    if not model_cols:
        return
        
    tr = out_df[out_df["split"] == "train"]
    vl = out_df[out_df["split"] == "val"]
    has_source = "source" in out_df.columns

    pdf_path = manager.get_plot_path("scatter_all", f"scatter_all_{slug}.pdf")
    with PdfPages(pdf_path) as pdf:
        for model in model_order:
            if model not in out_df.columns:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            r2 = scores.loc[scores["Model"] == model, "R-Squared"]
            r2_val = float(r2.iloc[0]) if len(r2) else float("nan")

            for ax, split_df, label in [
                (axes[0], tr, "Treino"),
                (axes[1], vl, "Validação"),
            ]:
                if split_df.empty or model not in split_df.columns:
                    ax.set_visible(False)
                    continue
                yt = split_df["y_true"].values
                yp = split_df[model].values
                lim = (min(yt.min(), yp.min()), max(yt.max(), yp.max()))

                if has_source and label == "Treino":
                    is_synth = split_df["source"].values == "synthetic"
                    ax.scatter(
                        yt[~is_synth], yp[~is_synth],
                        alpha=0.35, s=8, color="steelblue", label="Real",
                    )
                    if is_synth.any():
                        ax.scatter(
                            yt[is_synth], yp[is_synth],
                            alpha=0.5, s=10, color="darkorange",
                            marker="^", label="Sintético",
                        )
                    ax.legend(fontsize=7, markerscale=1.5)
                else:
                    ax.scatter(yt, yp, alpha=0.35, s=8, color="steelblue")

                ax.plot(lim, lim, "k--", lw=1)
                ax.set_xlabel("Observado (m/s)")
                ax.set_ylabel("Predito (m/s)")
                ax.set_title(label)

            fig.suptitle(
                f"{slug} — {model}  (R²val={r2_val:.3f})",
                fontsize=11,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"   PDF salvo: {pdf_path.name}")


# ── Processamento por cluster ─────────────────────────────────────────────────

def _filter_synth(synth_df_cluster, y_real, synth_n_above, synth_n_below, extreme_percentile):
    """Filtra e amostra sintéticos por grupo (extremo / normal) para um cluster."""
    if synth_df_cluster is None or len(synth_df_cluster) == 0:
        return None

    threshold = float(np.quantile(y_real, extreme_percentile))

    if "is_extreme" in synth_df_cluster.columns:
        above = synth_df_cluster[synth_df_cluster["is_extreme"]].copy()
        below = synth_df_cluster[~synth_df_cluster["is_extreme"]].copy()
    else:
        above = synth_df_cluster[synth_df_cluster[TARGET_VAR] >= threshold].copy()
        below = synth_df_cluster[synth_df_cluster[TARGET_VAR] < threshold].copy()

    parts = []
    if synth_n_above != 0 and len(above):
        n = synth_n_above if synth_n_above is not None else len(above)
        parts.append(above.sample(n=min(n, len(above)), replace=n > len(above), random_state=42))
    if synth_n_below and len(below):
        parts.append(below.sample(n=min(synth_n_below, len(below)), replace=synth_n_below > len(below), random_state=42))

    if not parts:
        return None
    result = pd.concat(parts, ignore_index=True)
    n_above_used = int((result[TARGET_VAR] >= threshold).sum())
    n_below_used = len(result) - n_above_used
    print(f"   Sintéticos selecionados: {n_above_used} extremos + {n_below_used} normais (threshold={threshold:.1f} m/s)")
    return result


def _process_one_cluster(cid, group, synth_df, manager, plt,
                         synth_n_above=None, synth_n_below=0,
                         extreme_percentile=0.90, season=None,
                         neighbor_data=None, eval_window="monthly",
                         active_features=None,
                         validation_mode="temporal", spatial_n_folds=5, spatial_seed=42):
    label = f"Cluster {cid}" + (f" / {season}" if season else "")
    n_stations = group["estacao"].nunique()
    df_tr = make_split(group, TRAIN_SLICE)
    df_vl = make_split(group, VAL_SLICE)
    df_te = make_split(group, TEST_SLICE)

    # Incorpora dados reais de clusters vizinhos ao treino (validação/teste
    # permanecem apenas no cluster alvo para medir performance sem contaminação)
    if neighbor_data is not None and not neighbor_data.empty:
        df_tr_neigh = make_split(neighbor_data, TRAIN_SLICE)
        if not df_tr_neigh.empty:
            n_neigh = len(df_tr_neigh)
            df_tr = pd.concat([df_tr, df_tr_neigh], ignore_index=True)
            print(f"   +{n_neigh} amostras reais de clusters vizinhos no treino")

    if season is not None:
        tr_season = month_to_season(df_tr["time"].dt.month)
        vl_season = month_to_season(df_vl["time"].dt.month)
        te_season = month_to_season(df_te["time"].dt.month)
        df_tr = df_tr[tr_season == season]
        df_vl = df_vl[vl_season == season]
        df_te = df_te[te_season == season]

    if df_tr.empty or df_vl.empty:
        print(f"   {label}: sem dados suficientes — pulando.")
        return

    # Teste (2024) nunca influencia a seleção do modelo (só val faz isso) —
    # é só calculado depois, com o vencedor já escolhido. Falta de teste num
    # cluster/season específico não pula o cluster, só desliga esse bloco.
    has_test = not df_te.empty
    if not has_test:
        print(f"   {label}: sem dados de teste (2024) — val segue valendo pra seleção, teste omitido.")

    # Climatologia por estação
    gust_p50_station = df_tr.groupby("estacao")[TARGET_VAR].median()
    gust_p50_cluster = float(df_tr[TARGET_VAR].median())
    df_tr = df_tr.copy()
    df_vl = df_vl.copy()
    df_tr["gust_P50"] = df_tr["estacao"].map(gust_p50_station)
    df_vl["gust_P50"] = df_vl["estacao"].map(gust_p50_station).fillna(gust_p50_cluster)
    if has_test:
        df_te = df_te.copy()
        df_te["gust_P50"] = df_te["estacao"].map(gust_p50_station).fillna(gust_p50_cluster)

    _features = active_features if active_features is not None else BASE_FEATURES
    train_features = [f for f in _features + ["gust_P50"] if f in df_tr.columns and df_tr[f].notna().any()]

    x_train = df_tr[train_features].reset_index(drop=True)
    y_train = df_tr[TARGET_VAR].reset_index(drop=True)
    x_val = df_vl[train_features]
    y_val = df_vl[TARGET_VAR].reset_index(drop=True)
    times_train = df_tr["time"].reset_index(drop=True)
    times_val = df_vl["time"].reset_index(drop=True)

    x_test = df_te[train_features] if has_test else None
    y_test = df_te[TARGET_VAR].reset_index(drop=True) if has_test else None

    y_real_train = y_train.to_numpy(float)
    y_synth_train = np.array([], dtype=float)
    has_synth = False
    synth_seasons_list = []

    x_train_orig = x_train.copy()
    y_train_orig = y_train.copy()

    if synth_df is not None:
        # Inclui sintéticos do cluster alvo e dos vizinhos
        neighbor_cids = (
            [str(c) for c in neighbor_data["cluster_id"].unique()]
            if neighbor_data is not None and not neighbor_data.empty
            else []
        )
        all_cids = {str(cid)} | set(neighbor_cids)
        s_raw = synth_df[synth_df["cluster_id"].astype(str).isin(all_cids)]
        if season is not None and "season" in s_raw.columns:
            s_raw = s_raw[s_raw["season"] == season]
        s = _filter_synth(s_raw if len(s_raw) else None, y_real_train,
                          synth_n_above, synth_n_below, extreme_percentile)
        if s is not None and len(s):
            has_synth = True
            y_synth_train = s[TARGET_VAR].to_numpy(float)
            s = s.copy()
            s["gust_P50"] = gust_p50_cluster
            x_synth = s.reindex(columns=train_features).reset_index(drop=True)
            x_train = pd.concat([x_train, x_synth], ignore_index=True)
            y_train = pd.concat(
                [y_train, s[TARGET_VAR].reset_index(drop=True)], ignore_index=True
            )
            synth_seasons_list = (
                s["season"].reset_index(drop=True).tolist()
                if "season" in s.columns
                else [None] * len(s)
            )
            print(f"   +{len(s)} linhas sintéticas no treino")

    train_season_labels = (
        month_to_season(times_train.dt.month).tolist() + synth_seasons_list
    )

    _plot_cluster_train_dist(
        {"cluster_id": cid, "y_real": y_real_train, "y_synth": y_synth_train},
        manager, plt,
    )

    # Cópia crua (real+sintético, pré-scaling) — usada só pra pré-processar
    # x_test de forma consistente com o x_train que o modelo realmente vê.
    x_train_full_raw = x_train.copy()

    x_train, x_val = preprocess_df(x_train, x_val)
    print(f"   Treino: {len(x_train)} | Val: {len(x_val)} | Features: {x_train.shape[1]}")

    x_test_pp = None
    if has_test:
        _, x_test_pp = preprocess_df(x_train_full_raw, x_test)

    # Baseline sem sintéticos (só se houver augment)
    scores_base = None
    if has_synth:
        x_tr_orig_pp, _ = preprocess_df(x_train_orig.copy(), x_val.copy())
        print("   [baseline] Rodando LazyPredict sem sintéticos...")
        reg_base = LazyRegressor(verbose=0, ignore_warnings=True, predictions=False,
                                 random_state=RANDOM_STATE, regressors=_FAST_REGRESSORS)
        scores_base, _ = reg_base.fit(x_tr_orig_pp, x_val, y_train_orig, y_val)
        scores_base = scores_base.reset_index().rename(columns={"index": "Model"})

    print(f"   {label}: rodando LazyPredict ({len(_FAST_REGRESSORS)} modelos)...", flush=True)
    reg = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True,
                        random_state=RANDOM_STATE, regressors=_FAST_REGRESSORS)
    scores, preds = reg.fit(x_train, x_val, y_train, y_val)

    slug = f"c{cid}" + (f"_{season}" if season else "")

    scores = scores.reset_index().rename(columns={"index": "Model"})
    scores.insert(0, "cluster_id", cid)
    if season:
        scores.insert(1, "season", season)
    scores.insert(2 if season else 1, "n_stations", n_stations)

    # ── Avaliação por janela de deploy ────────────────────────────────────
    top5_models = scores.nlargest(5, "R-Squared")["Model"].tolist()
    top5_in_preds = [m for m in top5_models if m in preds.columns]
    rolling_df = _evaluate_rolling_windows(
        times_val, y_val.to_numpy(), preds, top5_in_preds, eval_window
    )
    if not rolling_df.empty and top5_in_preds:
        best_col = top5_in_preds[0]
        r2_w = rolling_df[best_col]
        print(
            f"   Deploy R² ({eval_window}) — {best_col}: "
            f"μ={r2_w.mean():.3f}  σ={r2_w.std():.3f}  mín={r2_w.min():.3f}"
        )
        for model in top5_in_preds:
            if model in rolling_df.columns:
                mask = scores["Model"] == model
                scores.loc[mask, "R2_deploy_mean"] = round(rolling_df[model].mean(), 4)
                scores.loc[mask, "R2_deploy_std"] = round(rolling_df[model].std(), 4)
                scores.loc[mask, "R2_deploy_min"] = round(rolling_df[model].min(), 4)

    scores.to_csv(manager.get_partial_path("clusters", f"cluster_{slug}.csv"), index=False)
    
    pd.DataFrame([{"cluster_id": cid, "season": season, "n_stations": n_stations,
                   "n_samples": int(len(group))}]).to_csv(
        manager.get_partial_path("meta", f"meta_{slug}.csv"), index=False
    )

    val_df = preds.copy()
    val_df.insert(0, "y_true", y_val.values)
    val_df.insert(0, "split", "val")

    n_real_train = len(y_real_train)

    fitted = {}
    try:
        fitted = reg.provide_models(x_train, x_val, y_train, y_val)
        train_rows = {
            name: model.predict(x_train)
            for name, model in fitted.items()
            if name in preds.columns
        }
        train_df = pd.DataFrame(train_rows, index=range(len(x_train)))
        train_df.insert(0, "y_true", y_train.values)
        source_labels = ["real"] * n_real_train + ["synthetic"] * (len(x_train) - n_real_train)
        train_df.insert(0, "source", source_labels)
        train_df.insert(0, "season", train_season_labels)
        train_df.insert(0, "split", "train")
        val_df.insert(0, "source", "real")
        val_df.insert(0, "season", month_to_season(times_val.dt.month).values)
        out_df = pd.concat([train_df, val_df], ignore_index=True)
    except Exception as e:
        print(f"   [AVISO] Train predictions indisponíveis: {e}")
        val_df.insert(0, "source", "real")
        val_df.insert(0, "season", month_to_season(times_val.dt.month).values)
        out_df = val_df

    out_df.to_csv(manager.get_prediction_path(f"predictions_{slug}.csv"), index=False)
    print(f"   Predições salvas: predictions_{slug}.csv")

    rolling_dir = manager.get_plot_dir("rolling_r2")
    rolling_df.to_csv(rolling_dir / f"rolling_r2_{slug}.csv", index=False)
    _plot_rolling_r2(rolling_df, top5_in_preds, slug, manager, plt)

    metric_cols = [c for c in ["R-Squared", "RMSE", "MAE"] if c in scores.columns]
    top3 = scores.nlargest(3, "R-Squared")[["Model"] + metric_cols]
    print(f"   Top-3:\n{top3.to_string(index=False)}\n")

    best_model = scores.iloc[scores["R-Squared"].argmax()]["Model"]

    # ── Reavaliação em teste (2024) — só do vencedor escolhido por val ─────
    # Teste nunca participa da seleção (scores/best_model já vêm só de val);
    # aqui só medimos como o vencedor generaliza pra um período genuinamente
    # não visto durante a escolha do modelo.
    y_pred_test = None
    if has_test:
        if best_model in fitted:
            y_pred_test = fitted[best_model].predict(x_test_pp)
            r2_test = float(_r2_score(y_test, y_pred_test))
            print(f"   Teste (2024) — {best_model}: R²={r2_test:.3f}  n={len(y_test)}")
        else:
            print(f"   [AVISO] {label}: '{best_model}' indisponível em `fitted` — teste pulado.")

    # ── Linha/predições unificadas (schema comum entre pipelines) ──────────
    unified_rows = [build_results_row(
        pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
        season=season or "ALL", split="val",
        y_true=y_val.values, y_pred=preds[best_model].values,
        n_samples=len(y_val), extra={"model": best_model, "n_stations": n_stations},
    )]
    unified_pred_frames = [build_predictions_frame(
        pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
        season=season or "ALL", split="val",
        y_true=y_val.values, y_pred=preds[best_model].values,
    )]
    # predictions_by_station.csv — mesma granularidade que cluster_mlp.py já
    # produz (estacao/lat/lon reais, não só o cluster_id agregado acima).
    # df_vl mantém a mesma ordem de linhas usada pra construir x_val/y_val,
    # então reset_index(drop=True) alinha posicionalmente com y_val/preds.
    df_vl_aligned = df_vl.reset_index(drop=True)
    unified_station_pred_frames = [build_station_predictions_frame(
        estacao=df_vl_aligned["estacao"].values,
        latitude=df_vl_aligned["latitude"].values,
        longitude=df_vl_aligned["longitude"].values,
        time=df_vl_aligned["time"].values,
        cluster_id=cid, split="val",
        y_true=y_val.values, y_pred=preds[best_model].values,
    )]

    if y_pred_test is not None:
        unified_rows.append(build_results_row(
            pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
            season=season or "ALL", split="test",
            y_true=y_test.values, y_pred=y_pred_test,
            n_samples=len(y_test), extra={"model": best_model, "n_stations": n_stations},
        ))
        unified_pred_frames.append(build_predictions_frame(
            pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
            season=season or "ALL", split="test",
            y_true=y_test.values, y_pred=y_pred_test,
        ))
        df_te_aligned = df_te.reset_index(drop=True)
        unified_station_pred_frames.append(build_station_predictions_frame(
            estacao=df_te_aligned["estacao"].values,
            latitude=df_te_aligned["latitude"].values,
            longitude=df_te_aligned["longitude"].values,
            time=df_te_aligned["time"].values,
            cluster_id=cid, split="test",
            y_true=y_test.values, y_pred=y_pred_test,
        ))

    # ── Validação cruzada espacial (station-holdout), opcional ─────────────
    # Aditiva: só roda quando validation_mode != "temporal", só na passada
    # "todas as estações" (season is None, evita multiplicar custo por
    # trimestre) e reusa o ESTIMADOR CAMPEÃO já escolhido pelo split temporal
    # em vez de repetir o screening completo do LazyPredict por fold (custo
    # proibitivo: k folds × ~40 modelos). Isso significa que o k-fold espacial
    # valida o modelo já selecionado, não re-seleciona o modelo — limitação
    # assumida, documentada no relatório de saída.
    if validation_mode != "temporal" and season is None and best_model in fitted:
        from sklearn.base import clone

        if validation_mode == "loocv" and n_stations > 50:
            print(
                f"   [AVISO] cluster {cid} tem {n_stations} estações — "
                "LOOCV completo pode ser caro; considere --validation-mode spatial-kfold."
            )

        holdout_chunks = []
        for fold in iter_holdout_folds(
            group, mode=validation_mode, n_folds=spatial_n_folds, seed=spatial_seed,
            clim_value_col=TARGET_VAR,
        ):
            # gust_P50: mesma lógica de fallback já usada no split temporal —
            # média por estação no treino do fold, cluster-mean se a estação
            # held-out não aparecer no treino (sempre o caso aqui). Calculado
            # ANTES de `fold_features` — senão a coluna ainda não existiria em
            # `fold.df_train` e ficaria de fora do conjunto de features do fold.
            gp50_fold = fold.df_train.groupby("estacao")[TARGET_VAR].median()
            gp50_fold_cluster = float(fold.df_train[TARGET_VAR].median())
            df_tr_f = fold.df_train.copy()
            df_ev_f = fold.df_eval.copy()
            df_tr_f["gust_P50"] = df_tr_f["estacao"].map(gp50_fold)
            df_ev_f["gust_P50"] = df_ev_f["estacao"].map(gp50_fold).fillna(gp50_fold_cluster)

            # Reindexa para o schema COMPLETO de `train_features` (não um
            # subconjunto filtrado por fold) — o estimador campeão clonado
            # (`fitted[best_model]`) é um Pipeline do LazyPredict com um
            # ColumnTransformer que espera exatamente os nomes de coluna
            # vistos no fit original (temporal); um subconjunto diferente
            # levanta "A given column is not a column of the dataframe".
            # Colunas ausentes/100%-NaN neste fold (ex.: features ERA5-18UTC/
            # BT55 fora da cobertura do Paraná) viram NaN e são preenchidas
            # pelo imputer em `preprocess_df` (keep_empty_features=True).
            x_tr_f = df_tr_f.reindex(columns=train_features).reset_index(drop=True)
            y_tr_f = df_tr_f[TARGET_VAR].reset_index(drop=True)
            x_ev_f = df_ev_f.reindex(columns=train_features).reset_index(drop=True)
            y_ev_f = df_ev_f[TARGET_VAR].reset_index(drop=True)

            if len(x_tr_f) < 20 or len(x_ev_f) == 0:
                continue

            x_tr_f_pp, x_ev_f_pp = preprocess_df(x_tr_f, x_ev_f)
            try:
                champion = clone(fitted[best_model])
                champion.fit(x_tr_f_pp, y_tr_f)
                y_pred_f = champion.predict(x_ev_f_pp)
            except Exception as e:
                print(f"   [AVISO] station-holdout fold {fold.fold_id}: falha ao treinar '{best_model}' ({e!r}) — pulando fold.")
                continue

            unified_rows.append(build_results_row(
                pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
                season="ALL", split="spatial_holdout",
                y_true=y_ev_f.values, y_pred=y_pred_f, n_samples=len(y_ev_f),
                extra={
                    "model": best_model, "fold_id": fold.fold_id,
                    "validation_mode": validation_mode,
                    "n_stations_held_out": len(fold.held_out_stations),
                },
            ))
            unified_pred_frames.append(build_predictions_frame(
                pipeline="lazy", experiment=manager.exp_dir.name, cluster_id=cid,
                season="ALL", split="spatial_holdout",
                y_true=y_ev_f.values, y_pred=y_pred_f,
                extra_cols={"fold_id": fold.fold_id},
            ))

            chunk = df_ev_f[["estacao", "time"]].reset_index(drop=True).copy()
            chunk["cluster_id"] = cid
            chunk["fold_id"] = fold.fold_id
            chunk["validation_mode"] = validation_mode
            chunk["model"] = best_model
            chunk["y_true"] = y_ev_f.values
            chunk["y_pred"] = y_pred_f
            holdout_chunks.append(chunk)

        if holdout_chunks:
            holdout_df = pd.concat(holdout_chunks, ignore_index=True)
            holdout_df.to_csv(
                manager.get_prediction_path(f"spatial_holdout_predictions_{slug}.csv"), index=False
            )
            sh_metrics = compute_metrics(holdout_df["y_true"].values, holdout_df["y_pred"].values)
            print(
                f"   {best_model}  spatial-holdout ({validation_mode}) → "
                f"R²={sh_metrics['R2']:.3f}  RMSE={sh_metrics['RMSE']:.3f}  Bias@P90={sh_metrics['Bias_P90']:.3f}"
            )

    pd.DataFrame(unified_rows).to_csv(
        manager.get_partial_path("unified_results", f"unified_{slug}.csv"), index=False
    )
    pd.concat(unified_pred_frames, ignore_index=True).to_csv(
        manager.get_partial_path("unified_predictions", f"unified_pred_{slug}.csv"), index=False
    )
    pd.concat(unified_station_pred_frames, ignore_index=True).to_csv(
        manager.get_partial_path("unified_station_predictions", f"unified_station_pred_{slug}.csv"),
        index=False,
    )

    _plot_cluster_top5(scores, slug, manager, plt)
    _plot_cluster_scatter(
        {"cluster_id": slug, "n_stations": n_stations, "model": best_model,
         "y_true": y_val.values, "y_pred": preds[best_model].values},
        manager, plt,
    )

    # ── Serializar o melhor modelo para inferência espacial ────────────────
    # Salva: modelo fitado + imputer + scaler + lista de features. Persiste
    # tanto o modelo pooled (season=None) quanto os campeões por trimestre —
    # antes só o pooled era salvo, então o vencedor "por trimestre" da tabela
    # de ablation nunca chegava a ser usado na inferência espacial.
    if fitted and best_model in fitted:
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler

        # Reconstruir imputer/scaler (mesma lógica de preprocess_df)
        # x_train_orig contém os dados REAIS pré-preprocess (sem sintéticos, sem scaling)
        _imputer = SimpleImputer(strategy="mean").fit(x_train_orig)
        _scaler = RobustScaler().fit(_imputer.transform(x_train_orig))

        artifact = {
            "model": fitted[best_model],
            "model_name": best_model,
            "imputer": _imputer,
            "scaler": _scaler,
            "features": list(train_features),
            "cluster_id": cid,
            "season": season,
            "r2": float(scores.loc[scores["Model"] == best_model, "R-Squared"].iloc[0]),
            # LazyPredict prevê o valor absoluto (m/s), não uma razão sobre
            # o ERA5 — usado por SpatialCorrector pra decidir a reconstrução
            # correta (antes o dispatch era só `model_name == "MLPRegressor"`,
            # o que trataria errado um campeão LazyPredict que por acaso
            # também fosse MLPRegressor).
            "target_kind": "absolute",
        }
        joblib_path = manager.get_model_path(f"best_model_{slug}.joblib")
        joblib.dump(artifact, joblib_path)
        print(f"   Modelo salvo: {joblib_path.name} ({best_model}, R²={artifact['r2']:.4f})")

    _plot_seasonal_scatter(out_df, scores, slug, manager, plt)
    _plot_all_models_pdf(out_df, scores, slug, manager, plt)
    if scores_base is not None:
        _plot_synth_comparison(scores, scores_base, slug, manager, plt)
    print(f"[lazy_clusters] {label} concluído e salvo.")


def _build_aggregate(
    manager, cluster_merge, synthetic_csv, plt, sns,
    feature_groups: str = "original,era5_18z,bt55",
    active_features: list[str] | None = None, ablation_group: str | None = None,
):
    partial_dir = manager.get_partial_dir("clusters")
    partials = sorted(partial_dir.glob("cluster_*.csv"))
    if not partials:
        print("[lazy_clusters] Nada a agregar (sem parciais).")
        return

    results_df = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
    # Inclui 'season' no dedup em runs estratificadas: senão o mesmo
    # (cluster, Model) de DJF/MAM/JJA/SON/global colapsa em uma linha só.
    dedup_keys = ["cluster_id", "Model"]
    if "season" in results_df.columns:
        dedup_keys.append("season")
    results_df = results_df.drop_duplicates(subset=dedup_keys, keep="last")
    csv_path = manager.get_root_path("lazy_cluster_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[lazy_clusters] Resultados agregados de {len(partials)} cluster(s): {csv_path}")

    unified_dir = manager.get_partial_dir("unified_results")
    unified_partials = sorted(unified_dir.glob("unified_*.csv"))
    if unified_partials:
        unified_results_df = pd.concat(
            [pd.read_csv(p) for p in unified_partials], ignore_index=True
        )
        # split="spatial_holdout" tem várias linhas por (cluster_id, season,
        # split) — uma por fold — então "fold_id" entra na chave de dedup
        # quando presente, senão os folds colapsariam numa linha só.
        dedup_unified_keys = ["cluster_id", "season", "split"]
        if "fold_id" in unified_results_df.columns:
            dedup_unified_keys.append("fold_id")
        unified_results_df = unified_results_df.drop_duplicates(subset=dedup_unified_keys, keep="last")
        manager.write_results(unified_results_df)

        pred_dir = manager.get_partial_dir("unified_predictions")
        pred_partials = sorted(pred_dir.glob("unified_pred_*.csv"))
        if pred_partials:
            unified_preds_df = pd.concat(
                [pd.read_csv(p) for p in pred_partials], ignore_index=True
            )
            unified_preds_df.to_csv(manager.get_prediction_path("predictions.csv"), index=False)

        station_pred_dir = manager.get_partial_dir("unified_station_predictions")
        station_pred_partials = sorted(station_pred_dir.glob("unified_station_pred_*.csv"))
        if station_pred_partials:
            station_preds_df = pd.concat(
                [pd.read_csv(p) for p in station_pred_partials], ignore_index=True
            )
            station_preds_df.to_csv(
                manager.get_prediction_path("predictions_by_station.csv"), index=False
            )

    else:
        unified_results_df = pd.DataFrame(columns=CORE_RESULTS_COLUMNS)

    spatial_holdout_partials = sorted(
        manager.get_prediction_dir().glob("spatial_holdout_predictions_*.csv")
    )
    if spatial_holdout_partials:
        spatial_holdout_df = pd.concat(
            [pd.read_csv(p) for p in spatial_holdout_partials], ignore_index=True
        )
        spatial_holdout_df.to_csv(manager.get_prediction_path("spatial_holdout_predictions.csv"), index=False)
        print(f"[lazy_clusters] Predições de station-holdout agregadas: {len(spatial_holdout_partials)} cluster(s).")

    pivot = results_df.pivot_table(index="Model", columns="cluster_id", values="R-Squared")
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)
    _, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 2), min(30, pivot.shape[0] * 0.35 + 2)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.4, ax=ax)
    ax.set_title("R² por Modelo × Cluster", fontsize=14, pad=12)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Modelo")
    plt.tight_layout()
    heatmap_path = manager.get_plot_path("r2_heatmap", "r2_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[lazy_clusters] Heatmap agregado salvo.")

    meta_dir = manager.get_partial_dir("meta")
    meta_partials = sorted(meta_dir.glob("meta_*.csv"))
    if meta_partials:
        cluster_summary = (
            pd.concat([pd.read_csv(p) for p in meta_partials], ignore_index=True)
            .drop_duplicates("cluster_id", keep="last")
            .set_index("cluster_id")[["n_stations", "n_samples"]]
        )
    else:
        cluster_summary = results_df.drop_duplicates("cluster_id").set_index("cluster_id")[["n_stations"]]
        cluster_summary["n_samples"] = 0

    _write_experiment_meta(
        manager=manager, cluster_merge=cluster_merge,
        merge_groups=parse_cluster_merge(cluster_merge),
        cluster_summary=cluster_summary, results_df=results_df,
        unified_results_df=unified_results_df,
        synthetic_csv=synthetic_csv,
        feature_groups=feature_groups, active_features=active_features,
        ablation_group=ablation_group,
    )


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    output_dir: str = "artifacts/lazy_clusters",
    cluster_merge=None,
    synthetic_csv=None,
    synth_n_above=None,
    synth_n_below: int = 0,
    extreme_percentile: float = 0.90,
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    exp_name=None,
    cluster_id=None,
    aggregate_only: bool = False,
    list_clusters: bool = False,
    feature_groups: str = "original,era5_18z,bt55",
    restrict_coverage: bool = False,
    ablation_group: str | None = None,
    validation_mode: str = "temporal",
    spatial_n_folds: int = 5,
    spatial_seed: int = 42,
    **_,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    manager = ArtifactManager(output_dir, exp_name)
    print(f"[lazy_clusters] Experimento: {manager.root}")

    active_features = resolve_feature_groups(feature_groups)
    print(f"[lazy_clusters] Grupos de features: {feature_groups} ({len(active_features)} features)")

    if aggregate_only:
        _build_aggregate(
            manager, cluster_merge, synthetic_csv, plt, sns,
            feature_groups=feature_groups, active_features=active_features,
            ablation_group=ablation_group,
        )
        return None

    # --list-clusters só precisa do INMET + shapefile (spatial join) pra
    # enumerar cluster_id — NÃO precisa do merge ERA5 completo
    # (load_extended(), climatologia, build_flat_dataframe). Resolve isso
    # ANTES de qualquer carregamento pesado, pra ficar rápido de verdade
    # (segundos, não minutos) — antes chamava load_extended() incondicional
    # e só descartava o resultado depois de já ter pago o custo caro.
    if list_clusters:
        ds_inmet_light = xr.open_dataset(Path(raw_dir) / "INMET_Stratified.nc")
        station_clusters_light = assign_station_clusters(ds_inmet_light, shp_dir)
        merge_groups_light = parse_cluster_merge(cluster_merge)
        if merge_groups_light:
            station_clusters_light = apply_cluster_merge(
                station_clusters_light, merge_groups_light
            )
        clusters_present_light = sorted(
            station_clusters_light["cluster_id"].unique(), key=lambda c: str(c)
        )
        print(f"EXP_NAME:{manager.root.name}")
        print("CLUSTERS_JSON:" + json.dumps([str(c) for c in clusters_present_light]))
        return None

    synth_df = None
    if synthetic_csv:
        synth_df = pd.read_csv(synthetic_csv)
        print(f"[lazy_clusters] Augment: {len(synth_df)} linhas sintéticas de {synthetic_csv}")

    print("[lazy_clusters] Carregando NetCDF...")
    ds_inmet, ds_era5 = NetCDFLoader(raw_dir).load_extended()
    print(f"  Estações: {len(ds_inmet.estacao.values)} | ERA5 features: {list(ds_era5.data_vars)}")

    print("[lazy_clusters] Atribuindo clusters...")
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    print(station_clusters.to_string(index=False))

    print("[lazy_clusters] Calculando climatologia...")
    ds_clim = get_climatology(ds_inmet, TARGET_VAR, slice(*TRAIN_SLICE))

    print("[lazy_clusters] Construindo DataFrame de features...")
    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim)
    print(f"  Shape total: {df.shape}")
    if restrict_coverage:
        df = restrict_to_feature_coverage(df, feature_groups)

    merge_groups = parse_cluster_merge(cluster_merge)
    if merge_groups:
        df = apply_cluster_merge(df, merge_groups)
        labels = sorted("-".join(str(m) for m in sorted(g)) for g in merge_groups)
        print(f"[lazy_clusters] Agregando clusters: {', '.join(labels)}")

    clusters_present = sorted(df["cluster_id"].unique(), key=lambda c: str(c))

    print(f"EXP_NAME:{manager.root.name}")
    print("CLUSTERS_JSON:" + json.dumps([str(c) for c in clusters_present]))

    if cluster_id is not None:
        targets = [c for c in clusters_present if str(c) == str(cluster_id)]
        if not targets:
            print(f"[lazy_clusters] Cluster {cluster_id} não encontrado.")
            return None
    else:
        targets = clusters_present

    from src.pipelines.common import SEASONS as _SEASONS
    season_list = ([None] + list(_SEASONS.keys())) if stratify_seasons else [None]

    neighbors = _compute_cluster_neighbors(df, n_neighbor_clusters)
    if neighbors:
        for cid, nbrs in neighbors.items():
            print(f"[lazy_clusters] Vizinhos de {cid}: {nbrs}")

    for cid in targets:
        group = df[df["cluster_id"].astype(str) == str(cid)]
        nbr_ids = neighbors.get(cid, [])
        neighbor_data = (
            df[df["cluster_id"].isin(nbr_ids)] if nbr_ids else None
        )
        for season in season_list:
            label = f"Cluster {cid}" + (f" / {season}" if season else "")
            slug = f"c{cid}" + (f"_{season}" if season else "")
            partial_path = manager.get_partial_path("clusters", f"cluster_{slug}.csv")
            if partial_path.exists():
                print(f"── {label}: resultado já salvo ({partial_path.name}) — pulando ──")
                continue
            print(f"── {label} | {group['estacao'].nunique()} estação(ões) ──")
            _process_one_cluster(
            cid, group, synth_df, manager, plt,
            synth_n_above=synth_n_above,
            synth_n_below=synth_n_below,
            extreme_percentile=extreme_percentile,
            season=season,
            neighbor_data=neighbor_data,
            eval_window=eval_window,
            active_features=active_features,
            validation_mode=validation_mode,
            spatial_n_folds=spatial_n_folds,
            spatial_seed=spatial_seed,
        )

    if cluster_id is None:
        _build_aggregate(
            manager, cluster_merge, synthetic_csv, plt, sns,
            feature_groups=feature_groups, active_features=active_features,
            ablation_group=ablation_group,
        )

    return None
