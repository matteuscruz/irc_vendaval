"""Compara pipelines de IA e métodos clássicos de interpolação no domínio
onde ambos são comparáveis: extremos anuais (P95/P99 dos máximos anuais por
estação). Ver `src/comparison/unified_metrics.py` para a definição exata do
domínio e por que o domínio diário fica de fora.

Lê (sem recomputar nada pesado):
  - predictions_by_station.csv / spatial_holdout_predictions.csv das
    pipelines de IA já rodadas (`artifacts/{mlp,lazy}_clusters/<exp>/predictions/`)
  - interpolation comparisson/data/in/loocv_cache/loocv_consolidated_metrics.csv
    (já produzido por `interpolation comparisson/loocv/run_loocv_comparison.py`)

Uso:
    python scripts/compare_unified.py --tag v1 \
        --mlp-predictions artifacts/mlp_clusters/exp1/predictions/predictions_by_station.csv \
        --lazy-predictions artifacts/lazy_clusters/exp1/predictions/predictions_c3.csv \
        --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.comparison.unified_metrics import (
    load_ai_annual_extremes,
    load_classical_annual_extremes,
    build_unified_table,
)


# BASE_DIR de `interpolation comparisson/loocv/run_loocv_comparison.py` resolve
# para a RAIZ do irc_vendaval (3 níveis acima do script), não para dentro de
# "interpolation comparisson/" — herdado de quando esse projeto vivia solto
# na raiz de outro repo, antes de virar subpasta aqui. O caminho abaixo
# espelha essa mesma conta pra apontar pro lugar certo.
DEFAULT_LOOCV_METRICS = (
    Path(__file__).resolve().parents[1]
    / "data" / "in" / "loocv_cache" / "loocv_consolidated_metrics.csv"
)
OUT_DIR = Path("artifacts/unified_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mlp-predictions", default=None,
                         help="Caminho para predictions_by_station.csv ou spatial_holdout_predictions.csv do cluster_mlp")
    parser.add_argument("--lazy-predictions", default=None,
                         help="Caminho para predictions_*.csv ou spatial_holdout_predictions.csv do cluster_lazy")
    parser.add_argument("--split", default=None,
                         help="Filtra por split (ex.: 'test') antes de derivar percentis anuais — só válido "
                              "para predictions_by_station.csv, que tem coluna 'split'")
    parser.add_argument("--loocv-metrics", default=str(DEFAULT_LOOCV_METRICS),
                         help="loocv_consolidated_metrics.csv do projeto clássico")
    args = parser.parse_args()

    pred_paths = {}
    if args.mlp_predictions:
        pred_paths["mlp"] = args.mlp_predictions
    if args.lazy_predictions:
        pred_paths["lazy"] = args.lazy_predictions
    if not pred_paths:
        raise SystemExit("Forneça pelo menos --mlp-predictions ou --lazy-predictions.")

    ai_df = load_ai_annual_extremes(pred_paths, split_filter=args.split)
    classical_df = load_classical_annual_extremes(args.loocv_metrics)

    unified = build_unified_table(ai_df, classical_df)

    out_dir = OUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_metrics.csv"
    unified.to_csv(out_path, index=False)
    print(f"[compare_unified] Salvo: {out_path}")

    if not unified.empty:
        summary_path = out_dir / "unified_summary.csv"
        summary = (
            unified.sort_values(["pct", "rmse"])
            [["family", "pipeline", "pct", "bias", "rmse", "mae", "corr", "n"]]
        )
        summary.to_csv(summary_path, index=False)
        print(f"[compare_unified] Resumo (ranqueado por RMSE): {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
