#!/usr/bin/env python3
"""Comparação espacial lado a lado entre pipelines (lazy / mlp / lstm).

Roda (ou reaproveita, com --skip-existing) a inferência espacial de cada
pipeline informada via --<pipeline>-models-dir, e gera um painel nacional
comparativo (mapas absolutos + diffs vs uma referência) no estilo do
plot.py / SpatialCorrector._plot_map.

Uso:
    python3 scripts/compare_spatial.py \\
        --lazy-models-dir artifacts/lazy_modal/lazy_clusters/exp1/fitted_models \\
        --mlp-models-dir artifacts/mlp_modal/mlp_clusters/exp1/fitted_models \\
        --lstm-models-dir artifacts/modal/experiments/cluster_lstm_v1/fitted_models \\
        --year 2023
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_or_load(pipeline: str, models_dir: Path, raw_dir: str, shp_dir: str,
                  year: str, smoothing: str, cache_dir: Path, skip_existing: bool):
    import xarray as xr

    nc_path = cache_dir / f"{pipeline}_{year}_{smoothing}.nc"
    if skip_existing and nc_path.exists():
        print(f"[compare_spatial] {pipeline}: reaproveitando {nc_path}")
        return xr.open_dataset(nc_path)

    if pipeline == "lstm":
        from src.inference.spatial_correction_dl import DLSpatialCorrector as Corrector
    else:
        from src.inference.spatial_correction import SpatialCorrector as Corrector

    print(f"\n[compare_spatial] {pipeline}: rodando inferência espacial ({models_dir})...")
    corrector = Corrector(raw_dir, shp_dir, models_dir)
    corrector.prepare()
    time_slice = (f"{year}-01-01", f"{year}-12-31")
    ds = corrector.infer_grid(time_slice, smoothing=smoothing)
    corrector.save(ds, nc_path)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Comparação espacial lado a lado (lazy/mlp/lstm)")
    parser.add_argument("--raw-dir", type=str, default=str(ROOT / "dataset" / "raw"))
    parser.add_argument("--shp-dir", type=str, default=str(ROOT / "dataset" / "shp"))
    parser.add_argument("--lazy-models-dir", type=str, default=None)
    parser.add_argument("--mlp-models-dir", type=str, default=None)
    parser.add_argument("--lstm-models-dir", type=str, default=None)
    parser.add_argument("--year", type=str, default="2023")
    parser.add_argument("--smoothing", choices=["none", "gaussian"], default="gaussian")
    parser.add_argument("--reference", choices=["lazy", "mlp", "lstm"], default=None,
                        help="Pipeline usada como base dos painéis de diferença (default: a primeira disponível)")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "artifacts" / "spatial_comparisons"))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reaproveita .nc já gerados no cache em vez de rodar a inferência de novo")
    args = parser.parse_args()

    models_dirs = {
        "lazy": args.lazy_models_dir,
        "mlp": args.mlp_models_dir,
        "lstm": args.lstm_models_dir,
    }
    requested = {k: Path(v) for k, v in models_dirs.items() if v}
    if not requested:
        print("[compare_spatial] Informe pelo menos um --<pipeline>-models-dir.")
        return

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / tag
    cache_dir = out_dir / "nc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    named_max = {}
    for pipeline, models_dir in requested.items():
        if not models_dir.exists():
            print(f"[compare_spatial] AVISO: {pipeline} — models-dir não encontrado: {models_dir}")
            continue
        try:
            ds = _run_or_load(
                pipeline, models_dir, args.raw_dir, args.shp_dir,
                args.year, args.smoothing, cache_dir, args.skip_existing,
            )
        except Exception as e:
            print(f"[compare_spatial] AVISO: {pipeline} falhou — {e}")
            continue
        da_max = ds["rajada_max_corrigida"].max(dim="time").load()
        da_max = da_max.sortby(["latitude", "longitude"])
        named_max[pipeline] = da_max

    if len(named_max) < 2:
        print("[compare_spatial] Menos de 2 pipelines com dados válidos — nada a comparar lado a lado.")
        return

    from src.visualization.spatial_plots import plot_pipeline_comparison

    plot_path = out_dir / f"comparacao_espacial_{args.year}_{args.smoothing}.png"
    plot_pipeline_comparison(
        named_max, plot_path, reference=args.reference,
        title=f"Comparação Espacial entre Pipelines — Rajada Máxima ({args.year})",
    )
    print(f"\n[compare_spatial] Concluído. Saída em: {out_dir}")


if __name__ == "__main__":
    main()
