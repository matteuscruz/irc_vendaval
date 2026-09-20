#!/usr/bin/env python3
"""Gera (ou regenera) o cache do merge ERA5 completo — INMET × ERA5_Stratified
× ERA5-Basin, já sincronizado na janela de tempo comum e com
ORIGINAL_FEATURES reescrito via ERA5-Basin. As features novas
(dataset/raw/new_features) ficam fora do cache e são sempre relidas.

Roda o merge caro (NetCDFLoader.load_extended(use_cache=False)) UMA vez e
salva o resultado em dataset/raw/era5_merged_cache.nc. Depois disso, todo
NetCDFLoader(...).load_extended() (padrão use_cache=True) carrega direto do
cache em vez de recalcular — usado por GAN, LazyPredict, MLP e LSTM igualmente,
garantindo que todos treinem sobre exatamente o mesmo dataset sincronizado.

Uso:
    python3 scripts/build_synced_dataset.py
    python3 scripts/build_synced_dataset.py --raw-dir dataset/raw

Depois de gerar localmente, o cache sobe pro volume Modal junto com o resto
de dataset/raw/ na próxima vez que qualquer wrapper rodar `_ensure_dataset()`
(ex: --only-upload-dataset --force-dataset-upload).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(ROOT / "dataset" / "raw"))
    parser.add_argument(
        "--interp-method", default="nearest", choices=["nearest", "bilinear"],
        help="Grade→estação do ERA5-Basin. 'bilinear' (LSTM v2) grava "
             "era5_merged_cache_bilinear.nc; 'nearest' mantém o cache atual.",
    )
    args = parser.parse_args()

    from src.data.netcdf_loader import NetCDFLoader

    raw_dir = Path(args.raw_dir)
    cache_path = raw_dir / NetCDFLoader.merged_cache_name(args.interp_method)

    print(f"[build_synced_dataset] Gerando cache {cache_path.name} "
          f"(interp={args.interp_method}, ignora cache existente)...")
    t0 = time.time()
    ds_inmet, ds_era5 = NetCDFLoader(str(raw_dir)).load_extended(
        use_cache=False, interp_method=args.interp_method,
    )
    elapsed = time.time() - t0

    print(f"[build_synced_dataset] Merge concluído em {elapsed:.1f}s.")
    print(f"[build_synced_dataset] Estações: {len(ds_inmet.estacao)} | "
          f"Timesteps: {len(ds_era5.time)} | Variáveis ERA5: {len(ds_era5.data_vars)}")

    print(f"[build_synced_dataset] Salvando cache em {cache_path}...")
    if cache_path.exists():
        cache_path.unlink()
    NetCDFLoader.without_new_features(ds_era5).to_netcdf(cache_path)

    size_mb = cache_path.stat().st_size / 1024**2
    print(f"[build_synced_dataset] Cache salvo: {cache_path} ({size_mb:.1f} MB).")
    print(
        "[build_synced_dataset] Pronto — GAN/LazyPredict/MLP/LSTM vão carregar "
        "esse cache automaticamente (load_extended() padrão). Lembre de subir "
        "pro volume Modal: modal run src/modal/cluster_lstm.py "
        "--only-upload-dataset --force-dataset-upload"
    )


if __name__ == "__main__":
    main()
