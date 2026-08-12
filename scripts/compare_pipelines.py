#!/usr/bin/env python3
"""Comparador lado a lado das pipelines de treinamento (lazy / mlp / lstm).

Descobre o experimento mais recente (ou um fixado via --<pipeline>-exp) de
cada pipeline, carrega o results.csv / predictions/predictions.csv de schema
comum (ver src/pipelines/metrics_schema.py) e gera uma tabela e um conjunto de
gráficos comparativos organizados lado a lado em artifacts/comparisons/<tag>/.

Uso:
    python3 scripts/compare_pipelines.py
    python3 scripts/compare_pipelines.py --pipelines mlp,lstm
    python3 scripts/compare_pipelines.py --lazy-exp exp3 --mlp-exp exp2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.pipelines.metrics_schema import CORE_PREDICTIONS_COLUMNS

CORE_METRICS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
BAR_METRICS = ["R2", "RMSE", "Bias_P90", "RMSE_P90"]
METRIC_DIRECTIONS = {
    "R2": "higher", "RMSE": "lower", "Bias": "zero",
    "Bias_P90": "zero", "RMSE_P90": "lower",
}

DEFAULT_DIRS = {
    "lazy": ROOT / "artifacts" / "lazy_modal" / "lazy_clusters",
    "mlp": ROOT / "artifacts" / "mlp_modal" / "mlp_clusters",
    "lstm": ROOT / "artifacts" / "modal" / "experiments",
}

# Paleta categórica (skill dataviz — ordem fixa, slots 1/2/5, alto contraste
# em superfície clara; evita o slot 6 (vermelho) para não sugerir "ruim").
PIPELINE_COLORS = {
    "lazy": "#2a78d6",   # blue
    "mlp": "#1baf7a",    # aqua
    "lstm": "#4a3aa7",   # violet
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
# Rampa sequencial azul (100→700, clara→escura) para heatmaps.
BLUE_SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]


# ── Descoberta de experimentos ──────────────────────────────────────────────

def _discover_latest_exp(base_dir: Path) -> Path | None:
    candidates = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "run_meta.json"
        results_path = d / "results.csv"
        if meta_path.exists():
            try:
                ts = json.loads(meta_path.read_text()).get("timestamp")
            except Exception:
                ts = None
            sort_key = ts or datetime.fromtimestamp(
                d.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            candidates.append((sort_key, d))
        elif results_path.exists():
            sort_key = datetime.fromtimestamp(
                d.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            candidates.append((sort_key, d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: str(t[0]))
    return candidates[-1][1]


def resolve_experiment(name: str, base_dir: Path, pinned_exp: str | None) -> Path | None:
    if pinned_exp:
        exp_dir = base_dir / pinned_exp
        if not exp_dir.exists():
            print(f"[compare] AVISO: {name} — experimento '{pinned_exp}' não encontrado em {base_dir}")
            return None
        return exp_dir
    if not base_dir.exists():
        print(f"[compare] AVISO: {name} — diretório base não encontrado: {base_dir}")
        return None
    exp_dir = _discover_latest_exp(base_dir)
    if exp_dir is None:
        print(f"[compare] AVISO: {name} — nenhum experimento com results.csv/run_meta.json em {base_dir}")
        return None
    return exp_dir


def load_pipeline_data(name: str, exp_dir: Path):
    results_path = exp_dir / "results.csv"
    if not results_path.exists():
        print(
            f"[compare] AVISO: {name} — results.csv não encontrado em {exp_dir} "
            "(schema unificado ausente; rode a pipeline novamente após a atualização)."
        )
        return None, None
    results_df = pd.read_csv(results_path)
    pred_path = exp_dir / "predictions" / "predictions.csv"
    predictions_df = pd.read_csv(pred_path) if pred_path.exists() else None
    print(f"[compare] {name}: usando experimento '{exp_dir.name}' — {len(results_df)} linha(s) de resultado")
    return results_df, predictions_df


# ── Seleção/agregação para comparação ───────────────────────────────────────

def select_comparison_split(df: pd.DataFrame) -> pd.DataFrame:
    """Por pipeline, prefere split='test'; senão usa o único split disponível."""
    frames = []
    for pipeline, g in df.groupby("pipeline"):
        splits = set(g["split"].unique())
        chosen = "test" if "test" in splits else sorted(splits)[0]
        print(f"[compare] {pipeline}: split='{chosen}' selecionado para comparação")
        frames.append(g[g["split"] == chosen])
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def aggregate_over_season(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa 'season' em uma linha por (pipeline, cluster_id), média ponderada por n_samples."""
    rows = []
    for (pipeline, cluster_id), g in df.groupby(["pipeline", "cluster_id"]):
        w = g["n_samples"]
        row = {"pipeline": pipeline, "cluster_id": cluster_id, "n_samples": int(w.sum())}
        for m in CORE_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pipeline, g in df.groupby("pipeline"):
        w = g["n_samples"]
        row = {"pipeline": pipeline, "n_samples": int(w.sum())}
        for m in CORE_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)


def plot_grouped_bars(cluster_df: pd.DataFrame, plots_dir: Path, plt):
    pipelines = sorted(cluster_df["pipeline"].unique())
    clusters = sorted(cluster_df["cluster_id"].unique(), key=str)
    x = np.arange(len(clusters))
    width = 0.8 / max(len(pipelines), 1)

    for metric in BAR_METRICS:
        fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.1), 4.5))
        for i, pipeline in enumerate(pipelines):
            g = cluster_df[cluster_df["pipeline"] == pipeline].set_index("cluster_id")
            values = [g[metric].get(c, np.nan) for c in clusters]
            ax.bar(
                x + i * width - width * (len(pipelines) - 1) / 2, values, width,
                label=pipeline, color=PIPELINE_COLORS.get(pipeline, "#898781"),
                edgecolor=SURFACE, linewidth=0.6, zorder=2,
            )
        _style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        direction = METRIC_DIRECTIONS[metric]
        hint = {"higher": "maior é melhor", "lower": "menor é melhor", "zero": "mais perto de 0 é melhor"}[direction]
        ax.set_title(f"{metric} por cluster — pipelines lado a lado ({hint})", fontsize=12, color=INK_PRIMARY)
        ax.set_ylabel(metric, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        fig.tight_layout()
        path = plots_dir / f"bars_{metric.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"[compare] Plot salvo: {path}")


def plot_heatmap(summary_df: pd.DataFrame, plots_dir: Path, plt, mcolors):
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQUENTIAL)
    pipelines = summary_df["pipeline"].tolist()

    score = pd.DataFrame(index=pipelines, columns=CORE_METRICS, dtype=float)
    for m in CORE_METRICS:
        direction = METRIC_DIRECTIONS[m]
        vals = summary_df.set_index("pipeline")[m]
        if direction == "higher":
            raw = vals
        elif direction == "lower":
            raw = -vals
        else:
            raw = -vals.abs()
        lo, hi = raw.min(), raw.max()
        score[m] = 0.5 if (hi - lo) < 1e-12 else (raw - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(5, len(CORE_METRICS) * 1.4), max(3, len(pipelines) * 0.9 + 1)))
    im = ax.imshow(score.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CORE_METRICS)))
    ax.set_xticklabels(CORE_METRICS, color=INK_SECONDARY)
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, color=INK_SECONDARY)
    raw_vals = summary_df.set_index("pipeline")[CORE_METRICS]
    for i, pipeline in enumerate(pipelines):
        for j, m in enumerate(CORE_METRICS):
            v = raw_vals.loc[pipeline, m]
            txt_color = INK_PRIMARY if score.iloc[i, j] < 0.6 else "#ffffff"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=txt_color)
    ax.set_title("Resumo por pipeline — cor mais escura = melhor", fontsize=12, color=INK_PRIMARY, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = plots_dir / "summary_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[compare] Plot salvo: {path}")


def plot_residual_boxplots(pred_df: pd.DataFrame, plots_dir: Path, plt):
    if pred_df.empty:
        return
    pred_df = pred_df.copy()
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pipelines = sorted(pred_df["pipeline"].unique())
    data = [pred_df.loc[pred_df["pipeline"] == p, "residual"].dropna().values for p in pipelines]

    fig, ax = plt.subplots(figsize=(max(5, len(pipelines) * 1.8), 4.5))
    bp = ax.boxplot(
        data, tick_labels=pipelines, patch_artist=True, showfliers=False,
        medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
        whiskerprops={"color": BASELINE}, capprops={"color": BASELINE},
    )
    for patch, pipeline in zip(bp["boxes"], pipelines):
        patch.set_facecolor(PIPELINE_COLORS.get(pipeline, "#898781"))
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    _style_axes(ax)
    ax.set_title("Resíduos (predito − observado) por pipeline", fontsize=12, color=INK_PRIMARY)
    ax.set_ylabel("Resíduo (m/s)", color=INK_SECONDARY)
    fig.tight_layout()
    path = plots_dir / "residual_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[compare] Plot salvo: {path}")


def plot_scatter_grid(pred_df: pd.DataFrame, plots_dir: Path, plt):
    if pred_df.empty:
        return
    pipelines = sorted(pred_df["pipeline"].unique())
    fig, axes = plt.subplots(1, len(pipelines), figsize=(5 * len(pipelines), 5), squeeze=False)
    for ax, pipeline in zip(axes[0], pipelines):
        g = pred_df[pred_df["pipeline"] == pipeline]
        yt, yp = g["y_true"].values, g["y_pred"].values
        ax.scatter(yt, yp, s=12, alpha=0.35, color=PIPELINE_COLORS.get(pipeline, "#898781"), zorder=2)
        if len(yt):
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "--", color=BASELINE, linewidth=1.2, zorder=1)
        _style_axes(ax)
        ax.set_title(f"{pipeline} (n={len(g)})", fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel("Observado (m/s)", color=INK_SECONDARY)
        ax.set_ylabel("Predito (m/s)", color=INK_SECONDARY)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Observado × Predito por pipeline", fontsize=13, color=INK_PRIMARY, y=1.03)
    fig.tight_layout()
    path = plots_dir / "scatter_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[compare] Plot salvo: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compara resultados das pipelines lazy/mlp/lstm lado a lado")
    parser.add_argument("--pipelines", type=str, default="lazy,mlp,lstm")
    parser.add_argument("--lazy-dir", type=str, default=str(DEFAULT_DIRS["lazy"]))
    parser.add_argument("--mlp-dir", type=str, default=str(DEFAULT_DIRS["mlp"]))
    parser.add_argument("--lstm-dir", type=str, default=str(DEFAULT_DIRS["lstm"]))
    parser.add_argument("--lazy-exp", type=str, default=None)
    parser.add_argument("--mlp-exp", type=str, default=None)
    parser.add_argument("--lstm-exp", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "artifacts" / "comparisons"))
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    requested = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    base_dirs = {"lazy": Path(args.lazy_dir), "mlp": Path(args.mlp_dir), "lstm": Path(args.lstm_dir)}
    pinned_exps = {"lazy": args.lazy_exp, "mlp": args.mlp_exp, "lstm": args.lstm_exp}

    all_results, all_preds = [], []
    for name in requested:
        if name not in base_dirs:
            print(f"[compare] AVISO: pipeline desconhecida '{name}' — ignorando.")
            continue
        exp_dir = resolve_experiment(name, base_dirs[name], pinned_exps[name])
        if exp_dir is None:
            continue
        results_df, predictions_df = load_pipeline_data(name, exp_dir)
        if results_df is None or results_df.empty:
            continue
        all_results.append(results_df)
        if predictions_df is not None and not predictions_df.empty:
            all_preds.append(predictions_df)

    if not all_results:
        print("[compare] Nenhuma pipeline com resultados unificados encontrada. Nada a comparar.")
        return

    comparison_results = pd.concat(all_results, ignore_index=True, sort=False)
    comparison_predictions = (
        pd.concat(all_preds, ignore_index=True, sort=False)
        if all_preds else pd.DataFrame(columns=CORE_PREDICTIONS_COLUMNS)
    )

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / tag
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison_results.to_csv(out_dir / "comparison_results.csv", index=False)
    print(f"[compare] Tabela bruta salva: {out_dir / 'comparison_results.csv'}")

    selected_results = select_comparison_split(comparison_results)
    selected_preds = (
        select_comparison_split(comparison_predictions)
        if not comparison_predictions.empty else comparison_predictions
    )

    summary_df = build_summary(selected_results)
    summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"[compare] Resumo salvo: {out_dir / 'comparison_summary.csv'}")
    print(summary_df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cluster_df = aggregate_over_season(selected_results)
    plot_grouped_bars(cluster_df, plots_dir, plt)
    plot_heatmap(summary_df, plots_dir, plt, mcolors)
    plot_residual_boxplots(selected_preds, plots_dir, plt)
    plot_scatter_grid(selected_preds, plots_dir, plt)

    print(f"\n[compare] Comparação concluída. Saída em: {out_dir}")


if __name__ == "__main__":
    main()
