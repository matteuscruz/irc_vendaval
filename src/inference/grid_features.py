"""Constrói o vetor de features do modelo em CADA CÉLULA da grade ERA5-Basin,
para predição direta (sem passar por estações nem por IDW de resíduo).

Diferença para o caminho tradicional (`spatial_correction.py` /
`grid_generator.py`): lá o modelo prediz nas ESTAÇÕES e o resíduo é
interpolado (IDW) para a grade, o que amarra a cobertura à densidade de
estações naquele dia — em 2000 (2 estações ativas) ~42% da bacia fica sem
valor. Aqui o modelo é aplicado diretamente na célula, usando as features
ERA5 daquela célula, então toda célula da bacia recebe valor todo dia.

Todas as features do modelo são derivadas do ERA5 (ou de calendário/
coordenadas) — o INMET entra só como alvo no treino — então a célula tem
exatamente o mesmo vetor de entrada que a estação teve.

As features derivadas do ERA5 são construídas com o MESMO código canônico do
treino (`rebuild_original_from_basin`, `get_climatology`) — que é aritmética
xarray agnóstica a dimensões, e portanto vale igual para (time, estacao) e
para (time, latitude, longitude). Isso é o que garante que a distribuição das
features na grade bata com a que os modelos viram no treino.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

from src.data.climatology import get_climatology
from src.data.new_features import new_features_to_points
from src.data.original_features_basin import rebuild_original_from_basin
from src.pipelines.common import ERA5_GUST_PROXY, TRAIN_SLICE, month_to_season

ERA5_BASIN_FILE = "ERA5_Features_Basin_2000_2026.nc"


def _to_basin_names(ds: xr.Dataset) -> xr.Dataset:
    """Sufixa `_basin` nas variáveis cruas da grade, que é o que
    `rebuild_original_from_basin` espera (no caminho das estações esse sufixo
    vem do merge do ERA5BasinLoader)."""
    return ds.rename({v: f"{v}_basin" for v in ds.data_vars})


def load_grid_features(raw_dir: str | Path) -> xr.Dataset:
    """Grade ERA5-Basin com TODAS as features derivadas do ERA5 já montadas.

    Mantém dims (time, latitude, longitude); ainda inclui as células fora da
    bacia (todas NaN) — `stack_basin_cells` remove depois.
    """
    path = Path(raw_dir) / ERA5_BASIN_FILE
    ds = xr.open_dataset(path, chunks={"time": 365})
    ds = _to_basin_names(ds)

    # Features ORIGINAL_FEATURES derivadas do ERA5 — código canônico do treino.
    ds = rebuild_original_from_basin(ds)

    # Sazonalidade cíclica (idêntico a build_flat_dataframe).
    doy = ds["time"].dt.dayofyear
    month = ds["time"].dt.month
    ds = ds.assign(
        day_sin=np.sin(2 * np.pi * doy / 365.25),
        day_cos=np.cos(2 * np.pi * doy / 365.25),
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
    )

    # Climatologia por dia-do-ano, POR CÉLULA, a partir do proxy ERA5. Hoje é
    # só um preenchimento inicial da coluna: lazy/mlp e LSTM ajustam a
    # climatologia por série HARMÔNICA nos dias de treino da sua partição, e
    # `grid_direct_predict` sobrescreve esta coluna com aquela definição antes
    # de predizer (é o que mantém a feature na distribuição de treino). Só o
    # cluster_gan segue nesta versão por dia-do-ano.
    clim = get_climatology(ds, ERA5_GUST_PROXY, slice(*TRAIN_SLICE))
    ds = ds.assign(era5_clim_wind=clim.sel(dayofyear=doy).drop_vars("dayofyear"))

    # `latitude`/`longitude` já são features do modelo (no treino, as da
    # estação). Não precisam ser materializadas como variáveis: depois de
    # `stack_basin_cells` elas viram coordenadas 1-D ao longo de `cell` e
    # `.to_dataframe().reset_index()` já as entrega como colunas.
    return ds


def stack_basin_cells(ds: xr.Dataset) -> xr.Dataset:
    """Empilha (latitude, longitude) numa dimensão `cell` e mantém só as
    células com dado ERA5 — as 2129 da bacia. O resto do retângulo é NaN em
    TODAS as variáveis de entrada (máscara da bacia embutida no arquivo de
    origem), então não há o que predizer lá."""
    stacked = ds.stack(cell=("latitude", "longitude"))
    valid = np.isfinite(
        stacked[ERA5_GUST_PROXY].isel(time=0).values
    )
    stacked = stacked.isel(cell=np.where(valid)[0])

    # `stack` deixa `cell` como MultiIndex (latitude, longitude) — nesse
    # formato `.to_dataframe().reset_index()` expande nos dois níveis e a
    # coluna `cell` não existe. Troca por um índice inteiro simples,
    # preservando lat/lon como coordenadas 1-D ao longo de `cell`.
    lat = stacked["latitude"].values
    lon = stacked["longitude"].values
    stacked = stacked.drop_vars(["cell", "latitude", "longitude"],
                                errors="ignore")
    return stacked.assign_coords(
        cell=np.arange(len(lat)),
        latitude=("cell", lat),
        longitude=("cell", lon),
    )


def assign_cell_clusters(ds_cells: xr.Dataset, shp_dir: str | Path) -> pd.Series:
    """cluster_id de cada célula, por point-in-polygon no shapefile dos 14
    clusters. Toda célula da bacia cai em exatamente um cluster (verificado:
    2129/2129), então o resultado não tem NaN."""
    import geopandas as gpd
    from shapely.geometry import Point

    shp = gpd.read_file(Path(shp_dir) / "shp_vento.shp").to_crs("EPSG:4326")
    lats = ds_cells["latitude"].values
    lons = ds_cells["longitude"].values
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lons, lats)], crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, shp[["cluster", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return pd.Series(joined["cluster"].astype(float).astype("Int64").values,
                     name="cluster_id")


def new_features_for_cells(ds_cells: xr.Dataset, nf_grid: xr.Dataset | None) -> xr.Dataset:
    """Features novas (`nf_*`, grade de src/data/new_features.py) nas células
    da bacia. As duas grades são a mesma (os nós da região são nós do
    ERA5-Basin), então o nearest é leitura exata do nó; células fora da
    região ficam NaN e não recebem predição de modelos que usam essas
    features."""
    if nf_grid is None:
        return ds_cells
    pts = new_features_to_points(
        nf_grid, ds_cells["latitude"].values, ds_cells["longitude"].values,
        ds_cells["cell"].values, "cell", ds_cells["time"].values, "nearest",
    )
    return ds_cells.assign({v: pts[v] for v in pts.data_vars})


def iter_feature_frames(
    ds_cells: xr.Dataset,
    cluster_ids: pd.Series,
    years: list[int],
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Gera um DataFrame longo (linha = célula × dia) por ano.

    Fatiar por ano é necessário por memória: 2129 células × 9132 dias × ~46
    features passaria de 7 GB num frame só; por ano fica em ~285 MB.
    """
    feature_vars = [
        v for v in ds_cells.data_vars
        if set(ds_cells[v].dims) <= {"time", "cell"}
    ]
    for year in years:
        sub = ds_cells.sel(time=str(year))
        if sub.sizes["time"] == 0:
            continue
        df = sub[feature_vars].to_dataframe().reset_index()
        df["cluster_id"] = df["cell"].map(cluster_ids)
        df["season"] = month_to_season(df["time"].dt.month)

        yield year, df
