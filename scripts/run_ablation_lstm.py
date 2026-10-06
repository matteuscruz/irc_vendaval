#!/usr/bin/env python3
"""Matriz de ablation: cluster_lstm em 3 combinações de features.

Mesma matriz de scripts/run_ablation.py (original / +features novas /
+ERA5-Basin), mas para o pipeline LSTM (schema v2). cluster_lstm é
YAML-driven — cada braço reusa a MESMA config base, sobrepondo
feature_groups/ablation_group/exp_name via parâmetros de run(), sem precisar
de um YAML por braço.

Uso:
    python3 scripts/run_ablation_lstm.py --config config/experiment_cluster_lstm_modal.yaml
    python3 scripts/run_ablation_lstm.py --config config/experiment_cluster_lstm_modal.yaml --only original --skip-summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from scripts._ablation_common import ABLATION_MATRIX, comparison_dirname, exp_name_for, run_summary


def main():
    parser = argparse.ArgumentParser(description="Matriz de ablation para cluster_lstm")
    parser.add_argument("--config", required=True,
                        help="YAML base (config/experiment_cluster_lstm_modal.yaml ou similar)")
    parser.add_argument("--exp-prefix", default="",
                        help="Prefixo opcional pro nome do diretório de experimento "
                             "(default: vazio — diretório fica só com o nome do braço, ex: 'original')")
    parser.add_argument("--only", default=None,
                        help="Rodar só um braço (original|newfeatures|basin)")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="Só regenera o resumo/plots a partir de experimentos já rodados (não treina nada)")
    parser.add_argument("--local-output-dir", default=None,
                        help="Onde procurar os results.csv já baixados — necessário quando "
                             "experiment.output_dir do YAML é um caminho absoluto pensado para "
                             "dentro do container Modal (ex: /artifacts/modal/experiments), que "
                             "não existe localmente após o download (ex: artifacts/modal/experiments)")
    args = parser.parse_args()

    combos = ABLATION_MATRIX
    if args.only:
        combos = [c for c in ABLATION_MATRIX if c["name"] == args.only]
        if not combos:
            parser.error(f"--only inválido: {args.only!r} (válidos: {[c['name'] for c in ABLATION_MATRIX]})")

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    output_dir = (
        Path(args.local_output_dir) if args.local_output_dir
        else Path(base_cfg["experiment"]["output_dir"])
    )

    exp_names = [exp_name_for(args.exp_prefix, c["name"]) for c in combos]
    arm_names = {exp: c["name"] for exp, c in zip(exp_names, combos)}

    if not args.summary_only:
        from src.pipelines.cluster_lstm import run as run_cluster_lstm

        for combo, exp_name in zip(combos, exp_names):
            print(f"\n{'=' * 70}\n[ablation] Braço: {combo['name']}  (exp_name={exp_name})\n{'=' * 70}")
            run_cluster_lstm(
                config=args.config,
                feature_groups=combo["feature_groups"],
                ablation_group=combo["name"],
                exp_name_override=exp_name,
            )

    if not args.skip_summary:
        summary_dir = output_dir / comparison_dirname(args.exp_prefix)
        print(f"\n{'=' * 70}\n[ablation] Agregando resultados em: {summary_dir}\n{'=' * 70}")
        run_summary(output_dir, exp_names, summary_dir, arm_names)

    print(f"\n[ablation] Matriz concluída: {exp_names}")


if __name__ == "__main__":
    main()
