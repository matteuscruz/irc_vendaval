#!/usr/bin/env python3
"""Matriz de ablation: cluster_mlp em 4 combinações de dados/features.

Roda cluster_mlp.run() isoladamente para cada combinação — original / +
sintético / + features novas (ERA5-18UTC + BT55) / tudo junto — cada uma em
seu próprio experimento (plots completos + results.csv + run_meta.json com
"ablation_group" para filtro no dashboard), e agrega os 4 results.csv num
comparativo lado a lado (estilo scripts/compare_pipelines.py, mas agrupando
por "experiment" em vez de "pipeline", já que aqui todos os braços são mlp).

Uso:
    python3 scripts/run_ablation.py --synthetic-csv artifacts/gan_clusters/exp1/synthetic_augment.csv
    python3 scripts/run_ablation.py --only original --skip-summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._ablation_common import ABLATION_MATRIX, comparison_dirname, exp_name_for, run_summary


def main():
    parser = argparse.ArgumentParser(description="Matriz de ablation para cluster_mlp")
    parser.add_argument("--raw-dir", default=str(ROOT / "dataset" / "raw"))
    parser.add_argument("--shp-dir", default=str(ROOT / "dataset" / "shp"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "mlp_clusters"))
    parser.add_argument("--synthetic-csv", default=None,
                        help="Obrigatório se algum braço selecionado usar dados sintéticos")
    parser.add_argument("--exp-prefix", default="",
                        help="Prefixo opcional pro nome do diretório de experimento "
                             "(default: vazio — diretório fica só com o nome do braço, ex: 'original')")
    parser.add_argument("--hidden-layers", default="128,64")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--extreme-power", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--only", default=None,
                        help="Rodar só um braço (original|synthetic|newfeatures|all)")
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
        needs_synth = any(c["use_synthetic"] for c in combos)
        if needs_synth and not args.synthetic_csv:
            parser.error(
                "Um ou mais braços selecionados usam dados sintéticos — informe --synthetic-csv "
                "(gere com: python3 main.py cluster_gan --exp-name <nome>)."
            )

        from src.pipelines.cluster_mlp import run as run_cluster_mlp

        hidden_layers = tuple(int(x) for x in args.hidden_layers.split(","))

        for combo, exp_name in zip(combos, exp_names):
            print(f"\n{'=' * 70}\n[ablation] Braço: {combo['name']}  (exp_name={exp_name})\n{'=' * 70}")
            run_cluster_mlp(
                raw_dir=args.raw_dir,
                shp_dir=args.shp_dir,
                output_dir=str(output_dir),
                hidden_layers=hidden_layers,
                alpha=args.alpha,
                extreme_power=args.extreme_power,
                max_iter=args.max_iter,
                cluster_merge=None,
                synthetic_csv=args.synthetic_csv if combo["use_synthetic"] else None,
                exp_name=exp_name,
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
