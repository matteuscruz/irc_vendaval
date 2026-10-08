"""Regrava NetCDF com zlib nível 5 (e checa que os valores são idênticos). Uso: compactar_nc.py ORIGEM DESTINO"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr

NIVEL = 5


def compactar(origem: str, destino: str) -> None:
    with xr.open_dataset(origem) as d:
        enc = {v: {"zlib": True, "complevel": NIVEL, "shuffle": True} for v in d.data_vars}
        d.to_netcdf(destino, encoding=enc)
    with xr.open_dataset(origem) as a, xr.open_dataset(destino) as b:
        for v in a.data_vars:
            if not np.array_equal(a[v].values, b[v].values, equal_nan=True):
                raise SystemExit(f"{v}: valores diferem após compactar")
    print(f"{origem} -> {destino}: valores idênticos")


if __name__ == "__main__":
    compactar(sys.argv[1], sys.argv[2])
