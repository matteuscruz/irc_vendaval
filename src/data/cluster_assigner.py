from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr


def assign_station_clusters(ds_inmet: xr.Dataset, shp_dir: str) -> pd.DataFrame:
    """
    Faz spatial join das estações INMET com os polígonos de cluster.

    Retorna DataFrame com colunas [estacao, cluster_id] onde cluster_id é um
    inteiro. Estações que não intersectam nenhum polígono recebem cluster_id=-1.
    """
    shp_dir = Path(shp_dir)
    shp_file = next(shp_dir.glob("*.shp"))
    clusters_gdf = gpd.read_file(shp_file)

    # Reprojetar para WGS84 se necessário (EPSG:4674 ≈ EPSG:4326 mas sjoin precisa
    # que ambos estejam no mesmo CRS para não haver confusão)
    if clusters_gdf.crs is None:
        clusters_gdf = clusters_gdf.set_crs("EPSG:4674")
    clusters_gdf = clusters_gdf.to_crs("EPSG:4326")

    # Extrair lat/lon por estação (são constantes ao longo do tempo)
    stations_df = (
        ds_inmet[["latitude", "longitude"]]
        .to_dataframe()
        .groupby("estacao")
        .first()
        .reset_index()
    )
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df["longitude"], stations_df["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        stations_gdf[["estacao", "geometry"]],
        clusters_gdf[["cluster", "geometry"]],
        predicate="within",
        how="left",
    )

    # Converter cluster string ('01','02',...) para inteiro; NaN → -1
    result = joined[["estacao", "cluster"]].copy()
    result["cluster_id"] = (
        pd.to_numeric(result["cluster"], errors="coerce").fillna(-1).astype(int)
    )

    # Manter apenas uma linha por estação (sjoin pode duplicar se tocar dois polígonos)
    result = result.drop_duplicates(subset="estacao").reset_index(drop=True)

    return result[["estacao", "cluster_id"]]
