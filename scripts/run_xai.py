#!/usr/bin/env python3
"""Relatório de XAI (SHAP + importância por ablação) de um experimento treinado.

Não treina nada: lê os artefatos de `<exp>/fitted_models/`, reconstrói a matriz
de avaliação com a partição gravada em cada artefato e escreve os resultados em
`<exp>/xai/`.

Uso:
    python3 scripts/run_xai.py --exp-dir artifacts/lazy_modal/lazy_clusters/lazy_c3_static_mb
    python3 scripts/run_xai.py --exp-dir <...> --clusters 3 --seasons ALL,DJF --strategy shuffle

Saídas por (cluster, trimestre):
    xai/shap_summary_c<cid>[_<season>].csv     |SHAP| médio e participação (%)
    xai/shap_values_c<cid>[_<season>].csv.gz   valores por linha (auditoria)
    xai/ablation_c<cid>[_<season>].csv         ΔRMSE, ΔBias@P90, ΔR² por feature
    xai/plots/*.png                            barras de |SHAP| e de ablação
    xai/summary.csv                            uma linha por (cluster, trimestre)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.xai.explain import (  # noqa: E402
    ABLATION_STRATEGIES, ablation_importance, build_eval_matrix, load_artifact,
    shap_summary, shap_values,
)

SEASONS = ("DJF", "MAM", "JJA", "SON")


def _discover(models_dir: Path) -> list[tuple[int, str | None]]:
    """(cluster, trimestre) com artefato salvo. `None` = modelo agrupado."""
    found = []
    for path in sorted(models_dir.glob("best_model_c*.joblib")):
        stem = path.stem.removeprefix("best_model_c")
        cid, _, season = stem.partition("_")
        if cid.isdigit():
            found.append((int(cid), season or None))
    return sorted(found, key=lambda t: (t[0], t[1] or ""))


def _plot_bars(series: pd.Series, title: str, xlabel: str, path: Path, color: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = series.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(top) + 1)))
    ax.barh(top.index, top.to_numpy(), color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True, help="diretório do experimento treinado")
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--shp-dir", default="dataset/shp")
    ap.add_argument("--clusters", default=None, help="ex.: '3' ou '3,4' (default: todos)")
    ap.add_argument("--seasons", default=None, help="ex.: 'ALL,DJF' (ALL = modelo agrupado)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--strategy", default="mean", choices=list(ABLATION_STRATEGIES))
    ap.add_argument("--n-repeats", type=int, default=5, help="repetições da estratégia 'shuffle'")
    ap.add_argument("--max-shap-samples", type=int, default=2000)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    models_dir = exp_dir / "fitted_models"
    if not models_dir.is_dir():
        raise SystemExit(f"{models_dir} não existe — o experimento tem modelos salvos?")

    targets = _discover(models_dir)
    if args.clusters:
        want = {int(c) for c in args.clusters.split(",") if c.strip()}
        targets = [t for t in targets if t[0] in want]
    if args.seasons:
        want_s = {s.strip().upper() for s in args.seasons.split(",") if s.strip()}
        targets = [t for t in targets if (t[1] or "ALL").upper() in want_s]
    if not targets:
        raise SystemExit("nenhum (cluster, trimestre) selecionado.")

    out_dir = exp_dir / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[xai] {len(targets)} alvo(s): {[(c, s or 'ALL') for c, s in targets]}")

    summary = []
    for cid, season in targets:
        tag = f"c{cid}" + (f"_{season}" if season else "")
        print(f"\n[xai] === cluster {cid} / {season or 'ALL'} ===", flush=True)
        art = load_artifact(models_dir, cid, season)
        data = build_eval_matrix(args.raw_dir, args.shp_dir, art, split_label=args.split)
        print(f"[xai] {len(data.x)} linhas × {len(art['features'])} features | modelo {art['model_name']}")

        row = {
            "cluster_id": cid, "season": season or "ALL", "split": args.split,
            "model": art["model_name"], "n_rows": len(data.x), "n_features": len(art["features"]),
        }

        if not args.skip_shap:
            vals, kind = shap_values(art, data, max_samples=args.max_shap_samples)
            summ = shap_summary(vals)
            summ.to_csv(out_dir / f"shap_summary_{tag}.csv")
            vals.to_csv(out_dir / f"shap_values_{tag}.csv.gz", index=False, compression="gzip")
            _plot_bars(summ["shap_abs_mean"], f"Cluster {cid}/{season or 'ALL'} — |SHAP| médio",
                       "|SHAP| médio (m/s)", out_dir / "plots" / f"shap_{tag}.png", "#2a78d6")
            row["shap_explainer"] = kind
            row["shap_top1"] = summ.index[0]
            row["shap_top1_share_pct"] = round(float(summ["share_pct"].iloc[0]), 2)
            print(f"[xai] SHAP ({kind}) — top: "
                  + ", ".join(f"{f} ({summ.loc[f, 'share_pct']:.1f}%)" for f in summ.index[:5]))

        if not args.skip_ablation:
            abl = ablation_importance(art, data, strategy=args.strategy,
                                      n_repeats=args.n_repeats)
            abl.to_csv(out_dir / f"ablation_{tag}.csv")
            _plot_bars(abl["d_RMSE"], f"Cluster {cid}/{season or 'ALL'} — ablação ({args.strategy})",
                       "Δ RMSE (m/s) ao neutralizar a feature",
                       out_dir / "plots" / f"ablation_{tag}.png", "#c98a1f")
            base = abl.attrs["baseline"]
            row["ablation_strategy"] = args.strategy
            row["baseline_RMSE"] = round(base["RMSE"], 4)
            row["baseline_Bias_P90"] = round(base["Bias_P90"], 4)
            row["ablation_top1"] = abl.index[0]
            row["ablation_top1_dRMSE"] = round(float(abl["d_RMSE"].iloc[0]), 4)
            print(f"[xai] Ablação ({args.strategy}) — maior ΔRMSE: "
                  + ", ".join(f"{f} (+{abl.loc[f, 'd_RMSE']:.3f})" for f in abl.index[:5]))

        summary.append(row)

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[xai] Resultados em {out_dir}")


if __name__ == "__main__":
    main()
