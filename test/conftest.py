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


def _synthetic_daily_dates() -> pd.DatetimeIndex:
    """Calendário diário contínuo 2016-2019 — a LSTM v2 janela dias
    consecutivos e purga janelas que cruzam fronteira de mês de teste; com as
    datas mensais de _synthetic_dates() não sobraria nenhuma janela."""
    return pd.date_range("2016-01-01", "2019-12-31", freq="D")


HOURLY_FILENAME = "test_cluster_3_hourly.nc"
HOURLY_VARS = [
    "ws_h", "wd_h", "sin_dir_h", "cos_dir_h", "msl_h", "t2m_h",
    "rh_h", "td_dep_h", "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h",
]


def build_synthetic_hourly_file(raw_dir: Path) -> Path:
    """Arquivo horário mínimo (2018-2019) para as 3 estações do cluster 9
    sintético — mesmo schema (time horário × estacao) do arquivo real."""
    stations = [s for s, _, _, c in _SYNTHETIC_STATIONS if c == "09"]
    hours = pd.date_range("2018-01-01", "2019-12-31 23:00", freq="h")
    rng = np.random.default_rng(7)
    data = {
        v: (("time", "estacao"), rng.normal(size=(len(hours), len(stations))).astype("float32"))
        for v in HOURLY_VARS
    }
    xr.Dataset(data, coords={"time": hours, "estacao": stations}).to_netcdf(raw_dir / HOURLY_FILENAME)
    return raw_dir


BASIN_FILENAME = "ERA5_Features_Basin_2000_2026.nc"
BASIN_LATS = np.arange(-19.5, -23.01, -0.5)   # decrescente, como o arquivo real
BASIN_LONS = np.arange(-47.0, -43.49, 0.5)
BASIN_VARS = [
    "ws_mean", "ws_max", "ws_std", "ws_p90", "sin_dir_mean", "cos_dir_mean",
    "msl_mean", "msl_min", "msl_range", "t2m_mean", "t2m_max", "t2m_min",
    "t2m_range", "rh_mean", "rh_p10", "td_dep_mean", "tp_sum", "tp_max",
    "d_msl", "tp_lag1", "tp_lag2", "tp_roll3",
]


def _synthetic_basin(dates: pd.DatetimeIndex, rng: np.random.Generator) -> xr.Dataset:
    shape = (len(dates), len(BASIN_LATS), len(BASIN_LONS))
    ws_mean = rng.uniform(1, 6, shape)
    ws_max = ws_mean + rng.uniform(0.5, 8, shape)
    t2m = rng.normal(20, 5, shape)
    msl = rng.normal(101300, 500, shape)
    tp = rng.exponential(1.0, shape)
    angle = rng.uniform(-np.pi, np.pi, shape)
    fields = {
        "ws_mean": ws_mean, "ws_max": ws_max, "ws_std": rng.uniform(0.2, 2, shape),
        "ws_p90": ws_mean + rng.uniform(0.2, 4, shape),
        "sin_dir_mean": np.sin(angle), "cos_dir_mean": np.cos(angle),
        "msl_mean": msl, "msl_min": msl - rng.uniform(0, 300, shape),
        "msl_range": rng.uniform(0, 600, shape),
        "t2m_mean": t2m, "t2m_max": t2m + 4, "t2m_min": t2m - 4, "t2m_range": np.full(shape, 8.0),
        "rh_mean": rng.uniform(40, 95, shape), "rh_p10": rng.uniform(20, 60, shape),
        "td_dep_mean": rng.uniform(1, 10, shape),
        "tp_sum": tp, "tp_max": tp / 3, "d_msl": rng.normal(0, 200, shape),
        "tp_lag1": rng.exponential(1.0, shape), "tp_lag2": rng.exponential(1.0, shape),
        "tp_roll3": rng.exponential(1.0, shape),
    }
    return xr.Dataset(
        {k: (("time", "latitude", "longitude"), fields[k].astype("float32")) for k in BASIN_VARS},
        coords={"time": dates, "latitude": BASIN_LATS, "longitude": BASIN_LONS},
    )


def build_synthetic_raw_dir(raw_dir: Path, dates: pd.DatetimeIndex | None = None) -> Path:
    """Escreve um INMET_Stratified.nc + ERA5_Stratified.nc minúsculos em
    raw_dir, replicando o schema (dims/coords/variáveis) dos arquivos reais
    o suficiente para NetCDFLoader.load()/load_extended() rodar sem
    modificação.

    Inclui um ERA5_Features_Basin mínimo (grade de 0.5° sobre as estações):
    as pipelines não imputam nada e exigem todas as features pedidas, então
    sem ele ORIGINAL_FEATURES ficaria incompleto. A rajada INMET segue o
    ws_max do nó mais próximo de cada estação, para os modelos terem sinal.
    new_features fica ausente (ver build_synthetic_new_features).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    stations = [s for s, _, _, _ in _SYNTHETIC_STATIONS]
    lats = [lat for _, lat, _, _ in _SYNTHETIC_STATIONS]
    lons = [lon for _, _, lon, _ in _SYNTHETIC_STATIONS]
    dates = _synthetic_dates() if dates is None else dates

    rng = np.random.default_rng(42)
    n_t, n_s = len(dates), len(stations)

    u = rng.normal(0, 3, size=(n_t, n_s))
    v = rng.normal(0, 3, size=(n_t, n_s))
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
    basin = _synthetic_basin(dates, rng)
    basin.to_netcdf(raw_dir / BASIN_FILENAME)
    node_ws_max = basin["ws_max"].sel(
        latitude=xr.DataArray(lats, dims="estacao"),
        longitude=xr.DataArray(lons, dims="estacao"),
        method="nearest",
    ).transpose("time", "estacao").values

    extreme = rng.random((n_t, n_s)) > 0.95
    gust = node_ws_max * 1.8 + rng.normal(0, 2, size=(n_t, n_s))
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


NF_REGION = "cluster9"
NF_LATS = np.arange(-21.0, -23.01, -0.5)   # recorte da grade do Basin sintético
NF_LONS = np.arange(-46.5, -44.99, 0.5)    # cobre só as 3 estações do cluster 9


def build_synthetic_new_features(raw_dir: Path, years: range) -> Path:
    """dataset/raw/new_features/cluster9 no layout real: sl/<nome>_<AAAA>.nc
    horário com dim `valid_time`, static/<nome>.nc com valid_time=1. `u100`
    só existe no último ano — variável incompleta, deve ser descartada."""
    region = raw_dir / "new_features" / NF_REGION
    (region / "sl").mkdir(parents=True, exist_ok=True)
    (region / "static").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    coords = {"latitude": NF_LATS, "longitude": NF_LONS}
    for year in years:
        hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        shape = (len(hours), len(NF_LATS), len(NF_LONS))
        fields = {
            "u10": rng.normal(0, 3, shape), "v10": rng.normal(0, 3, shape),
            "cape": rng.exponential(300, shape),
        }
        if year == years[-1]:
            fields["u100"] = rng.normal(0, 5, shape)
        for name, arr in fields.items():
            xr.Dataset(
                {name: (("valid_time", "latitude", "longitude"), arr.astype("float32"))},
                coords={"valid_time": hours, **coords, "number": 0},
            ).to_netcdf(region / "sl" / f"{name}_{year}.nc")
    xr.Dataset(
        {"sdor": (("valid_time", "latitude", "longitude"),
                  rng.uniform(0, 200, (1, len(NF_LATS), len(NF_LONS))).astype("float32"))},
        coords={"valid_time": [pd.Timestamp("2024-06-15 12:00")], **coords},
    ).to_netcdf(region / "static" / "sdor.nc")
    return raw_dir


@pytest.fixture
def synthetic_nf_raw_dir(tmp_path: Path) -> Path:
    """Diário contínuo 2016-2019 + new_features do cluster 9."""
    dates = _synthetic_daily_dates()
    raw = build_synthetic_raw_dir(tmp_path / "raw_nf", dates)
    return build_synthetic_new_features(raw, range(dates[0].year, dates[-1].year + 1))


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    return build_synthetic_raw_dir(tmp_path / "raw")


@pytest.fixture
def synthetic_shp_dir(tmp_path: Path) -> Path:
    return build_synthetic_shp_dir(tmp_path / "shp")


@pytest.fixture
def synthetic_daily_raw_dir(tmp_path: Path) -> Path:
    """INMET + ERA5 diários contínuos (2016-2019) — fonte diária da LSTM v2."""
    return build_synthetic_raw_dir(tmp_path / "raw_daily", _synthetic_daily_dates())


@pytest.fixture
def synthetic_hourly_raw_dir(tmp_path: Path) -> Path:
    """Diário (INMET, alvo) + arquivo horário do cluster 9 — fonte horária."""
    raw = build_synthetic_raw_dir(tmp_path / "raw_hourly", _synthetic_daily_dates())
    return build_synthetic_hourly_file(raw)
