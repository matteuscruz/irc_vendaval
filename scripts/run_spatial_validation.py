"""Orquestra a validação cruzada espacial (station-holdout) das pipelines de
IA — análogo a `interpolation comparisson/loocv/run_loocv_comparison.py`.

Roda `cluster_mlp.run()` e/ou `cluster_lazy.run()` (chamada direta em
processo, sem subprocess) com `validation_mode="spatial-kfold"|"loocv"`,
depois consolida os `spatial_holdout_predictions.csv` de cada pipeline num
único CSV, junto com os `results.csv` completos (que já trazem as linhas
`split=train/val/test` do split temporal, lado a lado com as novas linhas
`split=spatial_holdout`) — usado por `generate_spatial_validation_report.py`.

Uso:
    python scripts/run_spatial_validation.py --tag holdout_v1 \
        --validation-mode spatial-kfold --spatial-n-folds 5

    # só uma pipeline, LOOCV completo:
    python scripts/run_spatial_validation.py --tag loocv_mlp \
        --pipelines mlp --validation-mode loocv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONSOLIDATED_DIR = Path("artifacts/spatial_validation")


def _run_mlp(args, exp_name: str) -> Path:
    from src.pipelines.cluster_mlp import run as run_mlp

    run_mlp(
        raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        output_dir="artifacts/mlp_clusters", exp_name=exp_name,
        validation_mode=args.validation_mode,
        spatial_n_folds=args.spatial_n_folds, spatial_seed=args.spatial_seed,
    )
    return Path("artifacts/mlp_clusters") / exp_name


def _run_lazy(args, exp_name: str) -> Path:
    from src.pipelines.cluster_lazy import run as run_lazy

    run_lazy(
        raw_dir=args.raw_dir, shp_dir=args.shp_dir,
        output_dir="artifacts/lazy_clusters", exp_name=exp_name,
        # station-holdout só roda na passada season=None (ver cluster_lazy.py);
        # desliga a estratificação sazonal aqui pra não pagar o custo do
        # screening completo do LazyPredict por trimestre à toa.
        stratify_seasons=False,
        validation_mode=args.validation_mode,
        spatial_n_folds=args.spatial_n_folds, spatial_seed=args.spatial_seed,
    )
    return Path("artifacts/lazy_clusters") / exp_name


def _collect(exp_dir: Path, pipeline: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Lê results.csv (train/val/test/spatial_holdout) e
    spatial_holdout_predictions.csv de um experimento, se existirem."""
    results_path = exp_dir / "results.csv"
    holdout_path = exp_dir / "predictions" / "spatial_holdout_predictions.csv"

    results_df = None
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        results_df["pipeline"] = pipeline

    holdout_df = None
    if holdout_path.exists():
        holdout_df = pd.read_csv(holdout_path)
        holdout_df["pipeline"] = pipeline
    else:
        print(f"[run_spatial_validation] [AVISO] {holdout_path} não existe "
              f"(pipeline '{pipeline}' pode não ter gerado station-holdout).")

    return results_df, holdout_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Nome do lote de validação espacial")
    parser.add_argument("--pipelines", default="mlp,lazy",
                         help="Subconjunto separado por vírgula: mlp,lazy (default: ambos)")
    parser.add_argument("--validation-mode", choices=["spatial-kfold", "loocv"], default="spatial-kfold")
    parser.add_argument("--spatial-n-folds", type=int, default=5)
    parser.add_argument("--spatial-seed", type=int, default=42)
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--shp-dir", default="dataset/shp")
    args = parser.parse_args()

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    all_results, all_holdout = [], []
    for pipeline in pipelines:
        exp_name = f"{args.tag}_{pipeline}"
        print(f"\n{'='*70}\n[run_spatial_validation] Rodando '{pipeline}' → {exp_name}\n{'='*70}")
        if pipeline == "mlp":
            exp_dir = _run_mlp(args, exp_name)
        elif pipeline == "lazy":
            exp_dir = _run_lazy(args, exp_name)
        else:
            raise ValueError(f"pipeline desconhecida: {pipeline!r} (use 'mlp' ou 'lazy')")

        results_df, holdout_df = _collect(exp_dir, pipeline)
        if results_df is not None:
            all_results.append(results_df)
        if holdout_df is not None:
            all_holdout.append(holdout_df)

    out_dir = CONSOLIDATED_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(out_dir / "consolidated_results.csv", index=False)
        print(f"[run_spatial_validation] Resultados consolidados: {out_dir / 'consolidated_results.csv'}")
    if all_holdout:
        pd.concat(all_holdout, ignore_index=True).to_csv(out_dir / "consolidated_predictions.csv", index=False)
        print(f"[run_spatial_validation] Predições consolidadas: {out_dir / 'consolidated_predictions.csv'}")
    else:
        print("[run_spatial_validation] [AVISO] Nenhuma predição de station-holdout coletada.")


if __name__ == "__main__":
    main()
