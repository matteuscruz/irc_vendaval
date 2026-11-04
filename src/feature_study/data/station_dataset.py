"""Base de treino por estação INMET para qualquer conjunto de clusters, a partir da grade diária no pico.

Usada para treinar modelos novos fora do cluster 3 (ex.: RS/SC, clusters 1–4). Cada linha é (estação, dia):
as colunas da célula ERA5 mais próxima da estação, na hora de maior rajada ERA5 do dia, e o alvo
`daily_wind_gust_max` do INMET.

O split é o MESMO do estudo de features:
  - teste = jan/abr/jul/out de 2000–2024 (`default_month_block_split`);
  - validação = blocos (ano, mês) sorteados dentro dos meses de treino;
  - purga de `purge_days(colunas)` dias nas fronteiras (`prepare.purge_split_boundaries`);
  - 2025 fica fora do split, com `_split = "novo"`: é o ano que nenhum modelo viu.
O split é função só do calendário, então é o mesmo para todas as estações e clusters: o "teste" de um
cluster e o de outro são os mesmos dias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.core.prepare import purge_split_boundaries
from src.feature_study.data.era5_features import haversine
from src.feature_study.data.groups_source import REFERENCE_COLUMN, STUDY_END
from src.feature_study.data.purge import purge_days
from src.pipelines.common import TARGET_VAR, default_month_block_split

NOVO = "novo"


def stations_of(inmet_path, shp_dir, clusters, lat, lon, min_dias: int = 1) -> pd.DataFrame:
    """Estações INMET dos `clusters`, com a célula da grade mais próxima e o número de dias com alvo."""
    from src.data.cluster_assigner import assign_station_clusters

    ds = xr.open_dataset(inmet_path)
    cl = assign_station_clusters(ds, str(shp_dir))
    cl = cl[cl.cluster_id.isin(list(clusters))]
    est = cl.estacao.tolist()
    st = pd.DataFrame({"estacao": est, "cluster_id": cl.cluster_id.to_numpy(),
                       "latitude": ds.latitude.sel(estacao=est).values,
                       "longitude": ds.longitude.sel(estacao=est).values})
    st["grid_lat"] = [float(lat[np.abs(lat - a).argmin()]) for a in st.latitude]
    st["grid_lon"] = [float(lon[np.abs(lon - b).argmin()]) for b in st.longitude]
    st["dist_celula_km"] = haversine(st.latitude, st.longitude, st.grid_lat, st.grid_lon)
    st["dias_inmet"] = ds.daily_wind_gust_max.sel(estacao=est).notnull().sum("time").values
    return st[st.dias_inmet >= min_dias].reset_index(drop=True)


def calendar_split(seed: int = 42):
    """O split do estudo sobre o calendário completo 2000–2024: o mesmo para qualquer conjunto de estações ou
    regiões (a validação é sorteada por blocos (ano, mês) e não depende de quais dias têm dado)."""
    return default_month_block_split(pd.date_range("2000-01-01", STUDY_END, freq="D").values, seed=seed)


def build(grade: pd.DataFrame, stations: pd.DataFrame, inmet_path, columns, seed: int = 42,
          split=None) -> pd.DataFrame:
    """Linhas (estação, dia) com alvo, colunas, `season`, `_split` e `era5_gust_max` (o `gust10fg` no pico).

    Dias sem alvo ou com alguma coluna faltando são descartados (nada é imputado). `split` (um
    `MonthBlockSplit`) fixa a partição; sem ele, é sorteada sobre os dias presentes. Para juntar regiões, passe
    o mesmo `calendar_split()` a todas."""
    columns = list(columns)
    g = grade.assign(lat=grade.lat.round(3), lon=grade.lon.round(3))
    st = stations.assign(grid_lat=stations.grid_lat.round(3), grid_lon=stations.grid_lon.round(3))
    obs = (xr.open_dataset(inmet_path)[TARGET_VAR].sel(estacao=st.estacao.tolist())
           .to_dataframe().reset_index()[["estacao", "time", TARGET_VAR]].dropna())
    obs = obs.merge(st[["estacao", "cluster_id", "grid_lat", "grid_lon"]], on="estacao")
    d = obs.merge(g, left_on=["grid_lat", "grid_lon", "time"], right_on=["lat", "lon", "time"], how="inner")
    d = d.dropna(subset=columns).reset_index(drop=True)
    d[REFERENCE_COLUMN] = d["gust10fg"].astype("float32")
    d["year"], d["month"] = d.time.dt.year, d.time.dt.month

    hist = d.time <= STUDY_END
    if split is None:
        split = default_month_block_split(np.unique(d.loc[hist, "time"].values), seed=seed)
    d["_split"] = NOVO
    d.loc[hist, "_split"] = split.label(pd.DatetimeIndex(d.loc[hist, "time"]))
    h, n = d[hist], d[~hist]
    h, _ = purge_split_boundaries(h.reset_index(drop=True), purge_days(columns))
    out = pd.concat([h, n], ignore_index=True)
    out["row_id"] = np.arange(len(out))
    return out
