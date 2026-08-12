"""Utilitários compartilhados pelos scripts run_ablation_{mlp,lazy,lstm}.py.

Matriz de ablation, agregação de results.csv/predictions.csv por "experiment"
(não por "pipeline" — dentro de uma pipeline, o que varia entre braços é o
experimento) e plots comparativos, no mesmo estilo visual de
scripts/compare_pipelines.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORE_METRICS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
BAR_METRICS = ["R2", "RMSE", "Bias_P90", "RMSE_P90"]
METRIC_DIRECTIONS = {
    "R2": "higher", "RMSE": "lower", "Bias": "zero",
    "Bias_P90": "zero", "RMSE_P90": "lower",
}

ABLATION_MATRIX = [
    {"name": "original", "feature_groups": "original", "use_synthetic": False},
    {"name": "synthetic", "feature_groups": "original", "use_synthetic": True},
    {"name": "newfeatures", "feature_groups": "original,era5_18z,bt55", "use_synthetic": False},
    {"name": "all", "feature_groups": "original,era5_18z,bt55", "use_synthetic": True},
    {"name": "basin", "feature_groups": "original,era5_basin", "use_synthetic": False},
    {"name": "all_basin", "feature_groups": "original,era5_18z,bt55,era5_basin", "use_synthetic": True},
]


def exp_name_for(prefix: str, arm: str) -> str:
    """Nome de diretório de experimento pra um braço, com prefixo opcional
    (vazio por padrão — o diretório fica só com o nome do braço, ex:
    "original", em vez de "ablation_original")."""
    return f"{prefix}_{arm}" if prefix else arm


def comparison_dirname(prefix: str) -> str:
    """Nome do diretório de agregação/comparação, com o mesmo prefixo opcional."""
    return f"{prefix}_comparison" if prefix else "comparison"

# Mesma família de paleta de scripts/compare_pipelines.py (evita vermelho).
ARM_COLORS = {
    "original": "#2a78d6",     # blue
    "synthetic": "#1baf7a",    # aqua
    "newfeatures": "#c98a1f",  # amber
    "all": "#4a3aa7",          # violet
    "basin": "#3d8b3d",        # green
    "all_basin": "#8a5a2e",    # brown
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]


# ── Agregação ────────────────────────────────────────────────────────────

def load_ablation_results(output_dir: Path, exp_names: list[str]) -> pd.DataFrame:
    frames = []
    for name in exp_names:
        path = output_dir / name / "results.csv"
        if not path.exists():
            print(f"[ablation] AVISO: {name} — results.csv não encontrado em {path}, pulando.")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_ablation_predictions(output_dir: Path, exp_names: list[str]) -> pd.DataFrame:
    frames = []
    for name in exp_names:
        path = output_dir / name / "predictions" / "predictions.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def select_split(df: pd.DataFrame) -> pd.DataFrame:
    """Por experimento, prefere split='test'; senão usa o único disponível."""
    frames = []
    for exp, g in df.groupby("experiment"):
        splits = set(g["split"].unique())
        chosen = "test" if "test" in splits else sorted(splits)[0]
        frames.append(g[g["split"] == chosen])
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def aggregate_over_season(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, cluster_id), g in df.groupby(["experiment", "cluster_id"]):
        w = g["n_samples"]
        row = {"experiment": exp, "cluster_id": cluster_id, "n_samples": int(w.sum())}
        for m in CORE_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, g in df.groupby("experiment"):
        w = g["n_samples"]
        row = {"experiment": exp, "n_samples": int(w.sum())}
        for m in CORE_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots (mesmo estilo visual de compare_pipelines.py) ─────────────────────

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)


def plot_grouped_bars(cluster_df: pd.DataFrame, plots_dir: Path, plt, arm_names: dict[str, str]):
    experiments = sorted(cluster_df["experiment"].unique())
    clusters = sorted(cluster_df["cluster_id"].unique(), key=str)
    x = np.arange(len(clusters))
    width = 0.8 / max(len(experiments), 1)

    for metric in BAR_METRICS:
        fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.1), 4.5))
        for i, exp in enumerate(experiments):
            g = cluster_df[cluster_df["experiment"] == exp].set_index("cluster_id")
            values = [g[metric].get(c, np.nan) for c in clusters]
            ax.bar(
                x + i * width - width * (len(experiments) - 1) / 2, values, width,
                label=exp, color=ARM_COLORS.get(arm_names.get(exp, exp), "#898781"),
                edgecolor=SURFACE, linewidth=0.6, zorder=2,
            )
        style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        direction = METRIC_DIRECTIONS[metric]
        hint = {"higher": "maior é melhor", "lower": "menor é melhor", "zero": "mais perto de 0 é melhor"}[direction]
        ax.set_title(f"{metric} por cluster — combinações de ablation ({hint})", fontsize=12, color=INK_PRIMARY)
        ax.set_ylabel(metric, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        fig.tight_layout()
        path = plots_dir / f"bars_{metric.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"[ablation] Plot salvo: {path}")


def plot_heatmap(summary_df: pd.DataFrame, plots_dir: Path, plt, mcolors):
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQUENTIAL)
    experiments = summary_df["experiment"].tolist()

    score = pd.DataFrame(index=experiments, columns=CORE_METRICS, dtype=float)
    for m in CORE_METRICS:
        direction = METRIC_DIRECTIONS[m]
        vals = summary_df.set_index("experiment")[m]
        raw = vals if direction == "higher" else (-vals if direction == "lower" else -vals.abs())
        lo, hi = raw.min(), raw.max()
        score[m] = 0.5 if (hi - lo) < 1e-12 else (raw - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(5, len(CORE_METRICS) * 1.4), max(3, len(experiments) * 0.9 + 1)))
    ax.imshow(score.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CORE_METRICS)))
    ax.set_xticklabels(CORE_METRICS, color=INK_SECONDARY)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, color=INK_SECONDARY)
    raw_vals = summary_df.set_index("experiment")[CORE_METRICS]
    for i, exp in enumerate(experiments):
        for j, m in enumerate(CORE_METRICS):
            v = raw_vals.loc[exp, m]
            txt_color = INK_PRIMARY if score.iloc[i, j] < 0.6 else "#ffffff"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=txt_color)
    ax.set_title("Resumo por combinação — cor mais escura = melhor", fontsize=12, color=INK_PRIMARY, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = plots_dir / "summary_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[ablation] Plot salvo: {path}")


def plot_residual_boxplots(pred_df: pd.DataFrame, plots_dir: Path, plt, arm_names: dict[str, str]):
    if pred_df.empty:
        return
    pred_df = pred_df.copy()
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    experiments = sorted(pred_df["experiment"].unique())
    data = [pred_df.loc[pred_df["experiment"] == e, "residual"].dropna().values for e in experiments]

    fig, ax = plt.subplots(figsize=(max(5, len(experiments) * 1.8), 4.5))
    try:
        bp = ax.boxplot(
            data, tick_labels=experiments, patch_artist=True, showfliers=False,
            medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
            whiskerprops={"color": BASELINE}, capprops={"color": BASELINE},
        )
    except TypeError:
        # matplotlib < 3.9 não tem tick_labels (era "labels").
        bp = ax.boxplot(
            data, labels=experiments, patch_artist=True, showfliers=False,
            medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
            whiskerprops={"color": BASELINE}, capprops={"color": BASELINE},
        )
    for patch, exp in zip(bp["boxes"], experiments):
        patch.set_facecolor(ARM_COLORS.get(arm_names.get(exp, exp), "#898781"))
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    style_axes(ax)
    ax.set_title("Resíduos (predito − observado) por combinação", fontsize=12, color=INK_PRIMARY)
    ax.set_ylabel("Resíduo (m/s)", color=INK_SECONDARY)
    fig.tight_layout()
    path = plots_dir / "residual_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[ablation] Plot salvo: {path}")


# ── Orquestração ─────────────────────────────────────────────────────────

def run_summary(
    output_dir: Path, exp_names: list[str], summary_dir: Path, arm_names: dict[str, str],
) -> None:
    results = load_ablation_results(output_dir, exp_names)
    if results.empty:
        print("[ablation] Nenhum results.csv encontrado — pulando resumo.")
        return
    predictions = load_ablation_predictions(output_dir, exp_names)

    plots_dir = summary_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(summary_dir / "comparison_results.csv", index=False)
    print(f"[ablation] Tabela bruta salva: {summary_dir / 'comparison_results.csv'}")

    selected = select_split(results)
    selected_preds = select_split(predictions) if not predictions.empty else predictions

    summary_df = build_summary(selected)
    summary_df.to_csv(summary_dir / "comparison_summary.csv", index=False)
    print(f"[ablation] Resumo salvo: {summary_dir / 'comparison_summary.csv'}")
    print(summary_df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cluster_df = aggregate_over_season(selected)
    plot_grouped_bars(cluster_df, plots_dir, plt, arm_names)
    plot_heatmap(summary_df, plots_dir, plt, mcolors)
    plot_residual_boxplots(selected_preds, plots_dir, plt, arm_names)
