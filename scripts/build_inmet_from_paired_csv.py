#!/usr/bin/env python3
"""Reconstrói INMET_Stratified.nc a partir de dataset/raw/Training_Dataset_
INMET_ERA5_Paired.csv — fonte mais rica que a antiga (615 estações
nacionais vs. 243 regionais, dado real desde 2000 vs. só 2020).

Usa só `rajada_max_inmet_ms`/`direcao_inmet_graus` (leitura REAL do INMET,
sem o fallback ERA5/ZERO/VIZINHOS que o CSV já embute em `rajada_max_final_
ms`) — mantém a mesma semântica "observação real, com buracos" que o
pipeline (climatologia, split temporal, etc.) já espera do arquivo atual.

O CSV pareado é NACIONAL (615 estações, do Amazonas ao Rio Grande do Sul),
mas a rede atual (243 estações) já vem pré-filtrada pra dentro da bacia —
nenhum código em src/ filtra cluster_id==-1 porque isso nunca foi preciso
antes. Por isso filtramos aqui via assign_station_clusters (mesmo spatial
join que o resto do pipeline usa) ANTES de escrever o .nc, senão as ~344
estações fora da bacia vazariam como um cluster "-1" espúrio pipeline
adentro.

`daily_wind_speed_mean` (presente no arquivo antigo) não existe no CSV
pareado e não é usada em lugar nenhum do pipeline (TARGET_VAR = "daily_
wind_gust_max") — omitida aqui de propósito, não é um esquecimento.

Uso
---
python3 scripts/build_inmet_from_paired_csv.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.cluster_assigner import assign_station_clusters


def build(csv_path: Path, shp_dir: str) -> xr.Dataset:
    df = pd.read_csv(
        csv_path,
        usecols=["data", "codigo_estacao", "latitude", "longitude",
                  "rajada_max_inmet_ms", "direcao_inmet_graus"],
        parse_dates=["data"],
    )
    df = df.rename(columns={
        "data": "time", "codigo_estacao": "estacao",
        "rajada_max_inmet_ms": "daily_wind_gust_max",
        "direcao_inmet_graus": "daily_wind_direction_at_gust_max",
    })

    latlon = df.groupby("estacao")[["latitude", "longitude"]].first()

    gust = df.pivot(index="time", columns="estacao", values="daily_wind_gust_max")
    direction = df.pivot(index="time", columns="estacao", values="daily_wind_direction_at_gust_max")
    gust, direction = gust.align(direction, join="outer")
    latlon = latlon.reindex(gust.columns)

    ds = xr.Dataset(
        {
            "daily_wind_gust_max": (("time", "estacao"), gust.to_numpy(dtype=np.float64)),
            "daily_wind_direction_at_gust_max": (("time", "estacao"), direction.to_numpy(dtype=np.float64)),
        },
        coords={
            "time": gust.index.to_numpy(),
            "estacao": gust.columns.to_numpy(dtype=object),
            "latitude": ("estacao", latlon["latitude"].to_numpy(dtype=np.float64)),
            "longitude": ("estacao", latlon["longitude"].to_numpy(dtype=np.float64)),
        },
    )
    ds["daily_wind_gust_max"].attrs = {"units": "m/s", "long_name": "Daily Maximum Wind Gust"}
    ds["daily_wind_direction_at_gust_max"].attrs = {
        "units": "degrees", "long_name": "Wind Direction at Daily Maximum Gust",
    }
    ds.attrs = {
        "source": "Training_Dataset_INMET_ERA5_Paired.csv (fonte_valor_final == INMET only)",
    }
    ds = ds.sortby("time")

    clusters = assign_station_clusters(ds, shp_dir)
    in_basin = clusters.loc[clusters["cluster_id"] != -1, "estacao"]
    n_before = ds.sizes["estacao"]
    ds = ds.sel(estacao=ds.estacao.isin(in_basin.to_numpy()))
    print(f"[build_inmet_from_paired_csv] {n_before} estações no CSV nacional -> "
          f"{ds.sizes['estacao']} dentro da bacia (shapefile) — "
          f"{n_before - ds.sizes['estacao']} descartadas por ficarem fora.")

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default="dataset/raw/Training_Dataset_INMET_ERA5_Paired.csv",
    )
    parser.add_argument("--out", default="dataset/raw/INMET_Stratified.nc")
    parser.add_argument("--shp-dir", default="dataset/shp")
    args = parser.parse_args()

    ds = build(Path(args.csv), args.shp_dir)
    n_stations = ds.sizes["estacao"]
    n_days = ds.sizes["time"]
    n_valid = int(ds["daily_wind_gust_max"].notnull().sum())
    print(f"[build_inmet_from_paired_csv] {n_stations} estações, {n_days} dias "
          f"({ds.time.min().values} -> {ds.time.max().values})")
    print(f"[build_inmet_from_paired_csv] {n_valid} observações reais de rajada "
          f"({n_valid / (n_stations * n_days):.1%} de preenchimento)")

    ds.to_netcdf(args.out)
    print(f"[build_inmet_from_paired_csv] escrito em {args.out}")
