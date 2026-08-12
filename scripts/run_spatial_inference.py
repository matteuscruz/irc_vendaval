#!/usr/bin/env python3
"""Inferência Espacial ERA5 — Vendaval V3.

Carrega os melhores modelos salvos por qualquer uma das 3 pipelines
(cluster_lazy, cluster_mlp: joblib | cluster_lstm: keras), prediz nas
estações, interpola via IDW para a grade ERA5 e gera o NetCDF + mapa.

O tipo de modelo é detectado automaticamente pelo conteúdo de --models-dir
(presença de dl_metadata.joblib + *.keras → LSTM; best_model_c*.joblib → lazy/mlp).
Use --pipeline para forçar.

Uso:
    python3 scripts/run_spatial_inference.py \\
        --models-dir artifacts/lazy_modal/lazy_clusters/exp1/fitted_models \\
        --year 2023

    python3 scripts/run_spatial_inference.py \\
        --models-dir artifacts/modal/experiments/cluster_lstm_v1/fitted_models \\
        --pipeline lstm --year 2023
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _detect_pipeline(models_dir: Path) -> str:
    if (models_dir / "dl_metadata.joblib").exists() or list(models_dir.glob("best_model_c*.keras")):
        return "lstm"
    return "classic"


def main():
    parser = argparse.ArgumentParser(description="Inferência Espacial ERA5 (Vendaval V3)")
    parser.add_argument("--raw-dir", type=str, default=str(ROOT / "dataset" / "raw"))
    parser.add_argument("--shp-dir", type=str, default=str(ROOT / "dataset" / "shp"))
    parser.add_argument("--models-dir", type=str, required=True,
                        help="Diretório com fitted_models (best_model_c*.joblib ou .keras + dl_metadata.joblib)")
    parser.add_argument("--pipeline", choices=["auto", "classic", "lstm"], default="auto",
                        help="Força o tipo de corretor; 'classic' serve lazy e mlp (ambos usam joblib)")
    parser.add_argument("--year", type=str, default="2023")
    parser.add_argument("--smoothing", choices=["none", "gaussian"], default="gaussian")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "output_corrigido_v3"))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    pipeline = _detect_pipeline(models_dir) if args.pipeline == "auto" else args.pipeline

    if pipeline == "lstm":
        from src.inference.spatial_correction_dl import DLSpatialCorrector as Corrector
    else:
        from src.inference.spatial_correction import SpatialCorrector as Corrector

    out_dir = Path(args.out_dir)
    out_file = out_dir / f"era5_corrigido_{args.year}_{args.smoothing}.nc"

    print("=" * 60)
    print(" INFERÊNCIA ESPACIAL — VENDAVAL V3")
    print("=" * 60)
    print(f" Pipeline:    {pipeline} ({Corrector.__name__})")
    print(f" Modelos:     {args.models_dir}")
    print(f" Ano:         {args.year}")
    print(f" Suavização:  {args.smoothing}")
    print(f" Saída:       {out_file}")
    print("=" * 60, "\n")

    t0 = time.time()

    corrector = Corrector(args.raw_dir, args.shp_dir, args.models_dir)
    corrector.prepare()

    time_slice = (f"{args.year}-01-01", f"{args.year}-12-31")
    ds_out = corrector.infer_grid(time_slice, smoothing=args.smoothing)
    corrector.save(ds_out, out_file)

    print(f"\n✓ Concluído em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
