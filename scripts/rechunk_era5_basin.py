"""Rechunkeia ERA5_Features_Basin_2000_2026.nc pra extração pontual rápida.

Problema: o arquivo original foi salvo sem chunking interno (contiguous=True,
chunksizes=None) — pra extrair um único ponto (lat, lon) ao longo do tempo,
o HDF5 não tem como pular direto pro bloco relevante, então cada leitura
pontual (via .sel(nearest).load()) acaba varrendo o array inteiro por
variável. Isso ficou medido em ~2359s pra extrair 271 estações de uma vez
(bloco único) — e qualquer tentativa de DIVIDIR essa leitura (por variável,
por lote de estações) piora ainda mais, porque cada chamada .load() paga
esse custo de varredura completa de novo (confirmado em 3 tentativas
distintas: leitura por variável ficou 3.3x mais lenta, leitura em lotes de
40 estações projetou ~98min vs 39min do bloco único).

Fix definitivo: reescrever o arquivo uma única vez com chunking espacial
pequeno (4×4 pixels) e tempo inteiro num chunk só — assim, extrair um ponto
só toca o chunk 4×4 que o contém, não o array inteiro. Roda uma vez;
todas as leituras futuras (aqui e no Modal) ficam rápidas depois disso.

Uso: python3 scripts/rechunk_era5_basin.py
"""
from __future__ import annotations

import time
from pathlib import Path

import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"
SRC_PATH = RAW_DIR / "ERA5_Features_Basin_2000_2026.nc"
DST_PATH = RAW_DIR / "ERA5_Features_Basin_2000_2026.rechunked.nc"

# Espacialmente pequeno (poucos pixels por chunk — extração pontual só lê o
# chunk que contém a estação), tempo inteiro num chunk só (não há motivo pra
# fatiar o tempo, séries são sempre lidas por completo).
CHUNK_LAT = 4
CHUNK_LON = 4


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Não encontrado: {SRC_PATH}")

    print(f"Abrindo {SRC_PATH.name} ({SRC_PATH.stat().st_size / 1024**3:.1f} GB)...")
    ds = xr.open_dataset(SRC_PATH)

    n_time = ds.sizes["time"]
    chunks = {"time": n_time, "latitude": CHUNK_LAT, "longitude": CHUNK_LON}
    print(f"Rechunkeando pra {chunks}...")
    ds = ds.chunk(chunks)

    encoding = {
        v: {
            "chunksizes": (n_time, CHUNK_LAT, CHUNK_LON),
            "zlib": True,
            "complevel": 4,
        }
        for v in ds.data_vars
    }

    print(f"Escrevendo {DST_PATH.name}...")
    _t0 = time.time()
    ds.to_netcdf(DST_PATH, encoding=encoding, engine="netcdf4")
    print(f"Concluído em {time.time() - _t0:.0f}s: "
          f"{DST_PATH.stat().st_size / 1024**3:.1f} GB.")
    ds.close()


if __name__ == "__main__":
    main()
