#!/usr/bin/env python3
"""Matriz de ablation: cluster_lazy em 3 combinações de features.

Mesma matriz de scripts/run_ablation.py (original / +features novas /
+ERA5-Basin), mas para o pipeline LazyPredict. Cada braço roda todos
os clusters (cluster_id=None) numa única chamada local — não usa o fan-out
por container do Modal (isso é feito pelo wrapper src/modal/cluster_lazy.py
quando rodado na nuvem).

Uso:
    python3 scripts/run_ablation_lazy.py
    python3 scripts/run_ablation_lazy.py --only original --skip-summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._ablation_common import ABLATION_MATRIX, comparison_dirname, exp_name_for, run_summary


def main():
    parser = argparse.ArgumentParser(description="Matriz de ablation para cluster_lazy")
    parser.add_argument("--raw-dir", default=str(ROOT / "dataset" / "raw"))
    parser.add_argument("--shp-dir", default=str(ROOT / "dataset" / "shp"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "lazy_clusters"))
    parser.add_argument("--no-stratify-seasons", action="store_false", dest="stratify_seasons")
    parser.add_argument("--n-neighbor-clusters", type=int, default=1)
    parser.add_argument("--eval-window", choices=["monthly", "biweekly"], default="monthly")
    parser.add_argument("--only", default=None,
                        help="Rodar só um braço (original|newfeatures|basin)")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="Só regenera o resumo/plots a partir de experimentos já rodados (não treina nada)")
    args = parser.parse_args()

    combos = ABLATION_MATRIX
    if args.only:
        combos = [c for c in ABLATION_MATRIX if c["name"] == args.only]
        if not combos:
            parser.error(f"--only inválido: {args.only!r} (válidos: {[c['name'] for c in ABLATION_MATRIX]})")

    output_dir = Path(args.output_dir)
    exp_names = [exp_name_for(args.exp_prefix, c["name"]) for c in combos]
    arm_names = {exp: c["name"] for exp, c in zip(exp_names, combos)}

    if not args.summary_only:
        from src.pipelines.cluster_lazy import run as run_cluster_lazy

        for combo, exp_name in zip(combos, exp_names):
            print(f"\n{'=' * 70}\n[ablation] Braço: {combo['name']}  (exp_name={exp_name})\n{'=' * 70}")
            run_cluster_lazy(
                raw_dir=args.raw_dir,
                shp_dir=args.shp_dir,
                output_dir=str(output_dir),
                cluster_merge=None,
                stratify_seasons=args.stratify_seasons,
                n_neighbor_clusters=args.n_neighbor_clusters,
                eval_window=args.eval_window,
                exp_name=exp_name,
                cluster_id=None,
                aggregate_only=False,
                list_clusters=False,
                feature_groups=combo["feature_groups"],
                ablation_group=combo["name"],
            )

    if not args.skip_summary:
        summary_dir = output_dir / comparison_dirname(args.exp_prefix)
        print(f"\n{'=' * 70}\n[ablation] Agregando resultados em: {summary_dir}\n{'=' * 70}")
        run_summary(output_dir, exp_names, summary_dir, arm_names)

    print(f"\n[ablation] Matriz concluída: {exp_names}")


if __name__ == "__main__":
    main()
