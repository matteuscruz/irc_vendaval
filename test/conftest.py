"""Fixtures compartilhadas — dados sintéticos tiny para os smoke tests de
pipeline (test_pipeline_smoke.py) e futuros testes que precisem de um
raw_dir/shp_dir válidos sem depender de dataset/raw/ (gitignorado, pesado,
não existe em CI)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

# 6 códigos+lat/lon de estações INMET reais (extraídos de
# dataset/raw/INMET_Stratified.nc), 3 de cada uma de duas estações climáticas
# reais distintas (cluster_id 9 e 10 em dataset/shp/shp_vento.shp) — só pra
# ter coordenadas plausíveis dentro da bacia. dataset/shp/ NÃO é versionado
# no git (mesma regra `dataset/*` do .gitignore que exclui dataset/raw/, só
# existe localmente) — build_synthetic_shp_dir() abaixo gera um shapefile
# próprio (um quadrado pequeno ao redor de cada estação, mesmo cluster_id),
# então os testes não dependem de nenhum arquivo fora do controle de versão.
_SYNTHETIC_STATIONS = [
    ("A504", -21.4500, -45.9500, "09"),
    ("A509", -22.8617, -46.0433, "09"),
    ("A515", -21.5664, -45.4042, "09"),
    ("A501", -19.9500, -44.0833, "10"),
    ("A502", -21.2283, -43.7678, "10"),
    ("A505", -19.6058, -46.9497, "10"),
]


def _synthetic_dates() -> pd.DatetimeIndex:
    """Dia 15 de cada mês, 2008-2025 — cobre TRAIN_SLICE (2008-2018),
    VAL_SLICE (2019) e TEST_SLICE (2020-2025) de src/pipelines/common.py
    com datas de calendário reais, mas em baixa densidade (216 pontos em vez
    de ~6500 dias) para os testes ficarem rápidos."""
    return pd.date_range("2008-01-01", "2025-12-01", freq="MS") + pd.Timedelta(days=14)


def build_synthetic_raw_dir(raw_dir: Path) -> Path:
    """Escreve um INMET_Stratified.nc + ERA5_Stratified.nc minúsculos em
    raw_dir, replicando o schema (dims/coords/variáveis) dos arquivos reais
    o suficiente para NetCDFLoader.load()/load_extended() rodar sem
    modificação.

    ERA5-18UTC/BT55/ERA5-Basin ficam ausentes de propósito — load_extended()
    já tolera isso (avisa e segue sem essas fontes, ver
    src/data/netcdf_loader.py), então os pipelines caem automaticamente para
    o subconjunto de ORIGINAL_FEATURES que só depende de INMET+ERA5 base.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    stations = [s for s, _, _, _ in _SYNTHETIC_STATIONS]
    lats = [lat for _, lat, _, _ in _SYNTHETIC_STATIONS]
    lons = [lon for _, _, lon, _ in _SYNTHETIC_STATIONS]
    dates = _synthetic_dates()

    rng = np.random.default_rng(42)
    n_t, n_s = len(dates), len(stations)

    u = rng.normal(0, 3, size=(n_t, n_s))
    v = rng.normal(0, 3, size=(n_t, n_s))
    wind_mag = np.sqrt(u**2 + v**2)
    t2m = rng.normal(293, 5, size=(n_t, n_s))
    dewpoint = t2m - rng.uniform(2, 8, size=(n_t, n_s))
    pressure = rng.normal(96000, 500, size=(n_t, n_s))
    precip = rng.exponential(0.5, size=(n_t, n_s))

    ds_era5 = xr.Dataset(
        {
            "10m_u_component_of_wind": (("time", "estacao"), u),
            "10m_v_component_of_wind": (("time", "estacao"), v),
            "2m_temperature": (("time", "estacao"), t2m),
            "2m_dewpoint_temperature": (("time", "estacao"), dewpoint),
            "surface_pressure": (("time", "estacao"), pressure),
            "total_precipitation": (("time", "estacao"), precip),
        },
        coords={
            "time": dates,
            "estacao": stations,
            "latitude": ("estacao", lats),
            "longitude": ("estacao", lons),
        },
    )
    ds_era5.to_netcdf(raw_dir / "ERA5_Stratified.nc")
    ds_era5.close()

    # Rajada correlacionada com o vento ERA5 + ruído + eventos extremos
    # ocasionais (~5%) — não precisa ser fisicamente realista, só dar sinal
    # suficiente pros modelos treinarem/avaliarem sem degenerar (ex.: GPD/EVT
    # do GAN precisa de alguns extremos de verdade pra ajustar).
    extreme = rng.random((n_t, n_s)) > 0.95
    gust = wind_mag * 1.8 + rng.normal(0, 2, size=(n_t, n_s))
    gust = np.where(extreme, gust + rng.uniform(10, 20, size=(n_t, n_s)), gust)
    gust = np.clip(gust, 0.5, None)
    direction = rng.uniform(0, 360, size=(n_t, n_s))

    ds_inmet = xr.Dataset(
        {
            "daily_wind_gust_max": (("time", "estacao"), gust),
            "daily_wind_direction_at_gust_max": (("time", "estacao"), direction),
        },
        coords={
            "time": dates,
            "estacao": stations,
            "latitude": ("estacao", lats),
            "longitude": ("estacao", lons),
        },
    )
    ds_inmet.to_netcdf(raw_dir / "INMET_Stratified.nc")
    ds_inmet.close()
    return raw_dir


def build_synthetic_shp_dir(shp_dir: Path) -> Path:
    """Escreve um shapefile minúsculo em shp_dir: um quadrado de ~0.1° de
    lado ao redor de cada estação sintética, rotulado com o cluster_id
    correspondente — o bastante pra assign_station_clusters() (spatial join
    'within') atribuir cada estação ao cluster certo sem depender do
    shapefile real (não versionado, ver comentário de _SYNTHETIC_STATIONS).
    """
    shp_dir.mkdir(parents=True, exist_ok=True)
    half = 0.05
    rows = [
        {"cluster": cluster, "geometry": box(lon - half, lat - half, lon + half, lat + half)}
        for _, lat, lon, cluster in _SYNTHETIC_STATIONS
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(shp_dir / "synthetic_clusters.shp")
    return shp_dir


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    return build_synthetic_raw_dir(tmp_path / "raw")


@pytest.fixture
def synthetic_shp_dir(tmp_path: Path) -> Path:
    return build_synthetic_shp_dir(tmp_path / "shp")
