#!/usr/bin/env python3
"""Gera a base de rajada máxima diária por PREDIÇÃO DIRETA em cada célula da
grade ERA5-Basin (2000-2024), sem passar por estações nem por IDW de resíduo.

Saída: NetCDF com dims (time, cell) — só as 2129 células da bacia, empilhadas,
com latitude/longitude/cluster_id como coordenadas. Formato escolhido para
ficar literalmente SEM NaN: no retângulo 86×77 do produto anterior, 67,85% das
células ficam fora da bacia e não têm nenhuma feature ERA5 de entrada, logo
não são preditíveis (a máscara da bacia já vem embutida no arquivo de origem
e coincide exatamente com a união dos 14 polígonos de cluster).

Uso:
    python3 scripts/generate_grid_direct.py \\
        --winners /caminho/winners_train2000.csv \\
        --out artifacts/corrected_grid/v2_direct_2000_2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.grid_direct_predict import predict_cluster  # noqa: E402
from src.data.new_features import load_new_features_grid  # noqa: E402
from src.inference.grid_features import (  # noqa: E402
    assign_cell_clusters, load_grid_features, new_features_for_cells, stack_basin_cells,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--shp-dir", default="dataset/shp")
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--winners", required=True,
                    help="CSV da tabela de vencedores (cluster × trimestre)")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", required=True, help="Diretório de saída")
    args = ap.parse_args()

    winners = pd.read_csv(args.winners)
    winners = winners[winners["season"] != "ALL"]

    print("[grid_direct] Montando features na grade...", flush=True)
    ds = load_grid_features(args.raw_dir).sel(time=slice(args.start, args.end))
    cells = stack_basin_cells(ds)
    clusters = assign_cell_clusters(cells, args.shp_dir)
    n_cells, n_times = cells.sizes["cell"], cells.sizes["time"]
    print(f"[grid_direct] {n_cells} células × {n_times} dias "
          f"= {n_cells * n_times:,} predições", flush=True)

    times = pd.DatetimeIndex(cells["time"].values)
    lat = cells["latitude"].values
    lon = cells["longitude"].values
    cid_arr = clusters.to_numpy()

    out = np.full((n_times, n_cells), np.nan, dtype="float32")
    nf_grid = load_new_features_grid(args.raw_dir)

    for cid in sorted(pd.unique(cid_arr)):
        sel = np.where(cid_arr == cid)[0]
        print(f"\n[grid_direct] Cluster {cid}: {len(sel)} células", flush=True)
        sub = new_features_for_cells(cells.isel(cell=sel), nf_grid)

        feats = [v for v in sub.data_vars
                 if set(sub[v].dims) <= {"time", "cell"}]
        # ordem (cell, time) — predict_cluster reshape (n_cells, n_times, ...)
        df = (sub[feats].to_dataframe().reset_index()
              .sort_values(["cell", "time"]).reset_index(drop=True))
        df["cluster_id"] = cid
        from src.pipelines.common import month_to_season
        df["season"] = month_to_season(df["time"].dt.month)

        preds = predict_cluster(df, int(cid), winners, args.artifacts_root,
                                len(sel), n_times)

        arr = preds.reshape(len(sel), n_times).T  # (time, cell)
        out[:, sel] = arr.astype("float32")
        del df, preds, arr

    ds_out = xr.Dataset(
        {"rajada_max_prevista": (["time", "cell"], out)},
        coords={
            "time": times,
            "cell": np.arange(n_cells),
            "latitude": ("cell", lat),
            "longitude": ("cell", lon),
            "cluster_id": ("cell", cid_arr.astype("int16")),
        },
        attrs={
            "source": "IRC Vendaval — predição direta por célula (sem IDW)",
            "metodo": "modelo vencedor por cluster × trimestre aplicado às "
                      "features ERA5-Basin da própria célula",
            "dominio": "2129 células = união dos 14 clusters = máscara da bacia",
            "winners_csv": str(args.winners),
        },
    )
    ds_out["rajada_max_prevista"].attrs = {"units": "m/s"}

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"grid_direct_{times[0].year}_{times[-1].year}.nc"
    ds_out.to_netcdf(path)

    n_nan = int(np.isnan(out).sum())
    print(f"\n[grid_direct] Salvo: {path}")
    print(f"[grid_direct] shape={out.shape}  NaN={n_nan} "
          f"({100 * n_nan / out.size:.4f}%)")
    if n_nan:
        falta = np.unique(cid_arr[np.where(np.isnan(out).any(axis=0))[0]])
        print(f"[grid_direct] AVISO: clusters com NaN: {falta}")


if __name__ == "__main__":
    main()
