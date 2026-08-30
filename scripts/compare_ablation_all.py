#!/usr/bin/env python3
"""Comparador cruzado: 3 pipelines × 6 braços de ablation, lado a lado.

Diferente de scripts/run_ablation*.py (compara os braços DENTRO de uma
pipeline) e de scripts/compare_pipelines.py (compara pipelines usando um
único experimento cada), este script junta as 18 combinações
(lazy/mlp/lstm × original/synthetic/newfeatures/all/basin/all_basin) numa
tabela e num conjunto de plots só, respondendo "qual pipeline se beneficia
mais de qual componente".

Uso:
    python3 scripts/compare_ablation_all.py
    python3 scripts/compare_ablation_all.py --lstm-dir artifacts/modal/experiments
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.compare_pipelines import (
    DEFAULT_DIRS, PIPELINE_COLORS, CORE_METRICS, METRIC_DIRECTIONS,
    BLUE_SEQUENTIAL, SURFACE, INK_PRIMARY, INK_SECONDARY, GRIDLINE, BASELINE,
)
from scripts._ablation_common import ABLATION_MATRIX, BAR_METRICS, exp_name_for

ARM_NAMES = [c["name"] for c in ABLATION_MATRIX]


# ── Descoberta + carga ───────────────────────────────────────────────────

def load_all(base_dirs: dict[str, Path], exp_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames, pred_frames = [], []
    for pipeline, base_dir in base_dirs.items():
        for arm in ARM_NAMES:
            exp_dir = base_dir / exp_name_for(exp_prefix, arm)
            res_path = exp_dir / "results.csv"
            if not res_path.exists():
                print(f"[compare-ablation] AVISO: {pipeline}/{arm} — results.csv não encontrado em {res_path}, pulando.")
                continue
            df = pd.read_csv(res_path)
            df["arm"] = arm
            result_frames.append(df)
            print(f"[compare-ablation] {pipeline}/{arm}: {len(df)} linha(s)")

            pred_path = exp_dir / "predictions" / "predictions.csv"
            if pred_path.exists():
                pdf = pd.read_csv(pred_path)
                pdf["arm"] = arm
                pred_frames.append(pdf)

    results = pd.concat(result_frames, ignore_index=True, sort=False) if result_frames else pd.DataFrame()
    predictions = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()
    return results, predictions


def select_split(df: pd.DataFrame) -> pd.DataFrame:
    """Por (pipeline, arm), prefere split='test'; senão usa o único disponível."""
    frames = []
    for (_, _), g in df.groupby(["pipeline", "arm"]):
        splits = set(g["split"].unique())
        chosen = "test" if "test" in splits else sorted(splits)[0]
        frames.append(g[g["split"] == chosen])
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipeline, arm), g in df.groupby(["pipeline", "arm"]):
        w = g["n_samples"]
        row = {"pipeline": pipeline, "arm": arm, "n_samples": int(w.sum())}
        for m in CORE_METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking(summary_df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    """Rankeia os braços DENTRO de cada pipeline por `rank_by` (rank 1 = melhor).

    R² geral mascara desempenho na cauda (ver gans_research.md, eixo TSTR) —
    'RMSE_P90'/'Bias_P90' são a régua recomendada pra julgar se um braço com
    dados sintéticos realmente ajuda nos extremos, não só no corpo da
    distribuição. `Bias_P90` rankeia por |Bias_P90| (mais perto de 0 é
    melhor, sinal não importa pra ranking).
    """
    df = summary_df.copy()
    df["_rank_key"] = df["Bias_P90"].abs() if rank_by == "Bias_P90" else df[rank_by]
    ascending = rank_by != "R2"  # R2: maior é melhor. RMSE_P90/|Bias_P90|: menor é melhor.
    df = df.sort_values(["pipeline", "_rank_key"], ascending=[True, ascending]).reset_index(drop=True)
    df["rank"] = df.groupby("pipeline").cumcount() + 1
    return df.drop(columns="_rank_key")


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


def plot_grouped_bars(summary_df: pd.DataFrame, plots_dir: Path, plt):
    pipelines = [p for p in PIPELINE_COLORS if p in summary_df["pipeline"].unique()]
    arms = [a for a in ARM_NAMES if a in summary_df["arm"].unique()]
    x = np.arange(len(arms))
    width = 0.8 / max(len(pipelines), 1)

    for metric in BAR_METRICS:
        fig, ax = plt.subplots(figsize=(max(6, len(arms) * 1.6), 4.5))
        for i, pipeline in enumerate(pipelines):
            g = summary_df[summary_df["pipeline"] == pipeline].set_index("arm")
            values = [g[metric].get(a, np.nan) for a in arms]
            ax.bar(
                x + i * width - width * (len(pipelines) - 1) / 2, values, width,
                label=pipeline, color=PIPELINE_COLORS.get(pipeline, "#898781"),
                edgecolor=SURFACE, linewidth=0.6, zorder=2,
            )
        _style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        direction = METRIC_DIRECTIONS[metric]
        hint = {"higher": "maior é melhor", "lower": "menor é melhor", "zero": "mais perto de 0 é melhor"}[direction]
        ax.set_title(f"{metric} por braço de ablation — pipelines lado a lado ({hint})", fontsize=12, color=INK_PRIMARY)
        ax.set_ylabel(metric, color=INK_SECONDARY)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        fig.tight_layout()
        path = plots_dir / f"bars_{metric.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"[compare-ablation] Plot salvo: {path}")


def plot_heatmap(summary_df: pd.DataFrame, plots_dir: Path, plt, mcolors):
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQUENTIAL)
    summary_df = summary_df.copy()
    summary_df["combo"] = summary_df["pipeline"] + " / " + summary_df["arm"]
    summary_df = summary_df.sort_values(["pipeline", "arm"])
    combos = summary_df["combo"].tolist()

    score = pd.DataFrame(index=combos, columns=CORE_METRICS, dtype=float)
    for m in CORE_METRICS:
        direction = METRIC_DIRECTIONS[m]
        vals = summary_df.set_index("combo")[m]
        raw = vals if direction == "higher" else (-vals if direction == "lower" else -vals.abs())
        lo, hi = raw.min(), raw.max()
        score[m] = 0.5 if (hi - lo) < 1e-12 else (raw - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(5, len(CORE_METRICS) * 1.4), max(4, len(combos) * 0.5 + 1)))
    ax.imshow(score.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CORE_METRICS)))
    ax.set_xticklabels(CORE_METRICS, color=INK_SECONDARY)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos, color=INK_SECONDARY, fontsize=8)
    raw_vals = summary_df.set_index("combo")[CORE_METRICS]
    for i, combo in enumerate(combos):
        for j, m in enumerate(CORE_METRICS):
            v = raw_vals.loc[combo, m]
            txt_color = INK_PRIMARY if score.iloc[i, j] < 0.6 else "#ffffff"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=txt_color)
    ax.set_title("Resumo — pipeline × braço de ablation (cor mais escura = melhor)", fontsize=12, color=INK_PRIMARY, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = plots_dir / "summary_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[compare-ablation] Plot salvo: {path}")


def plot_residual_boxplots(pred_df: pd.DataFrame, plots_dir: Path, plt):
    if pred_df.empty:
        return
    pred_df = pred_df.copy()
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["combo"] = pred_df["pipeline"] + "/" + pred_df["arm"]
    combos = sorted(pred_df["combo"].unique())
    data = [pred_df.loc[pred_df["combo"] == c, "residual"].dropna().values for c in combos]
    colors = [PIPELINE_COLORS.get(c.split("/")[0], "#898781") for c in combos]

    fig, ax = plt.subplots(figsize=(max(7, len(combos) * 1.0), 4.5))
    try:
        bp = ax.boxplot(
            data, tick_labels=combos, patch_artist=True, showfliers=False,
            medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
            whiskerprops={"color": BASELINE}, capprops={"color": BASELINE},
        )
    except TypeError:
        # matplotlib < 3.9 não tem tick_labels (era "labels").
        bp = ax.boxplot(
            data, labels=combos, patch_artist=True, showfliers=False,
            medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
            whiskerprops={"color": BASELINE}, capprops={"color": BASELINE},
        )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    _style_axes(ax)
    ax.set_title("Resíduos (predito − observado) — pipeline × braço", fontsize=12, color=INK_PRIMARY)
    ax.set_ylabel("Resíduo (m/s)", color=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = plots_dir / "residual_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[compare-ablation] Plot salvo: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compara as 3 pipelines × 4 braços de ablation lado a lado")
    parser.add_argument("--lazy-dir", type=str, default=str(DEFAULT_DIRS["lazy"]))
    parser.add_argument("--mlp-dir", type=str, default=str(DEFAULT_DIRS["mlp"]))
    parser.add_argument("--lstm-dir", type=str, default=str(DEFAULT_DIRS["lstm"]))
    parser.add_argument("--exp-prefix", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "artifacts" / "ablation_comparison_all"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--rank-by", choices=["R2", "RMSE_P90", "Bias_P90"], default="R2",
                        help="Métrica usada pra rankear os braços dentro de cada pipeline em "
                             "tstr_ranking.csv (rank 1 = melhor). Default R2 preserva o "
                             "comportamento de leitura anterior; RMSE_P90/Bias_P90 julgam pela "
                             "cauda (P90+), a régua recomendada pra decidir se um braço com "
                             "dados sintéticos ajuda de verdade nos extremos (ver gans_research.md).")
    args = parser.parse_args()

    base_dirs = {
        "lazy": Path(args.lazy_dir), "mlp": Path(args.mlp_dir), "lstm": Path(args.lstm_dir),
    }
    results, predictions = load_all(base_dirs, args.exp_prefix)
    if results.empty:
        print("[compare-ablation] Nenhum results.csv encontrado em nenhuma combinação — nada a comparar.")
        return

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / tag
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "comparison_results.csv", index=False)
    print(f"[compare-ablation] Tabela bruta salva: {out_dir / 'comparison_results.csv'}")

    selected = select_split(results)
    selected_preds = select_split(predictions) if not predictions.empty else predictions

    summary_df = build_summary(selected)
    summary_df.to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"[compare-ablation] Resumo salvo: {out_dir / 'comparison_summary.csv'}")
    print(summary_df.sort_values(["pipeline", "arm"]).to_string(index=False))

    ranking_df = build_ranking(summary_df, args.rank_by)
    ranking_df.to_csv(out_dir / "tstr_ranking.csv", index=False)
    print(f"\n[compare-ablation] Ranking por {args.rank_by} salvo: {out_dir / 'tstr_ranking.csv'}")
    print(ranking_df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    plot_grouped_bars(summary_df, plots_dir, plt)
    plot_heatmap(summary_df, plots_dir, plt, mcolors)
    plot_residual_boxplots(selected_preds, plots_dir, plt)

    print(f"\n[compare-ablation] Comparação concluída. Saída em: {out_dir}")


if __name__ == "__main__":
    main()
