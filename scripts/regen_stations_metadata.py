"""Regenera stations_metadata.csv (estacao, cluster_id, latitude, longitude)
diretamente do INMET_Stratified.nc + shapefiles de cluster, sem precisar
rodar nenhum treino (é o mesmo bloco que cluster_mlp.py roda antes do loop
de modelos — ver src/pipelines/cluster_mlp.py). Útil pra corrigir o
dashboard depois que a rede de estações do INMET muda (ex: 30 -> 243
estações) sem esperar um re-treino completo no Modal.

Uso
---
python3 scripts/regen_stations_metadata.py \\
    --raw-dir dataset/raw --shp-dir dataset/shp \\
    --out-dir ../irc_vendaval_dashboard/artifacts/ablation/mlp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from src.data.cluster_assigner import assign_station_clusters


def build_stations_metadata(raw_dir: str, shp_dir: str) -> "pd.DataFrame":
    ds_inmet = xr.open_dataset(Path(raw_dir) / "INMET_Stratified.nc")
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    latlon = (
        ds_inmet[["latitude", "longitude"]].to_dataframe()
        .groupby("estacao").first().reset_index()
    )
    return station_clusters.merge(latlon, on="estacao", how="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--shp-dir", default="dataset/shp")
    parser.add_argument("--out-dir", required=True, help="artifacts/ablation/mlp no repo do dashboard")
    parser.add_argument(
        "--arms", nargs="+",
        default=["original", "synthetic", "newfeatures", "all", "basin", "all_basin"],
    )
    args = parser.parse_args()

    stations_meta = build_stations_metadata(args.raw_dir, args.shp_dir)
    print(f"[regen_stations_metadata] {len(stations_meta)} estações, "
          f"{stations_meta['cluster_id'].nunique()} clusters.")

    out_base = Path(args.out_dir)
    n_written = 0
    for arm in args.arms:
        arm_dir = out_base / arm
        if not arm_dir.is_dir():
            continue
        stations_meta.to_csv(arm_dir / "stations_metadata.csv", index=False)
        n_written += 1
        print(f"  -> {arm_dir / 'stations_metadata.csv'}")

    print(f"[regen_stations_metadata] {n_written} braço(s) atualizados.")
