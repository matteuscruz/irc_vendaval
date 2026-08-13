"""
GERAÇÃO DE SÉRIES PAREADAS (INMET + ERA5)
-----------------------------------------
Descrição:
    Este script realiza o pareamento espacial e temporal entre estações automáticas
    do INMET e o modelo de reanálise ERA5.
    
    Responsabilidades:
    1. Carregar os dados do INMET (y) e do ERA5 (x).
    2. Construir índices espaciais (KDTree) para localizar vizinhos e pontos de grade.
    3. Preencher lacunas (gap-filling) na rajada do INMET usando vizinhos próximos.
    4. Consolidar métricas de magnitude e direção em um CSV tabular.

Saída:
    - CSV contendo séries temporais pareadas para cada estação.
"""

import os
import gc
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Gerencia o carregamento e pré-processamento de Datasets (INMET/ERA5)."""
    def __init__(self, inmet_path: str, era5_pattern: str, years: range):
        self.inmet_path = inmet_path
        self.era5_pattern = era5_pattern
        self.years = years
        self.full_date_range = pd.date_range(
            start=f"{years.start}-01-01", end=f"{years.stop-1}-12-31", freq='D'
        )

    def load_inmet(self) -> xr.Dataset:
        logging.info("Carregando dados INMET...")
        ds = xr.open_dataset(self.inmet_path)
        return ds.reindex(time=self.full_date_range)

    def load_era5_gust(self, var_name='i10fg') -> xr.Dataset:
        logging.info("Carregando e agregando ERA5 (Rajada)...")
        files = [self.era5_pattern.format(ano=ano) for ano in self.years]
        
        with xr.open_mfdataset(files) as ds:
            if 'valid_time' in ds.coords:
                ds = ds.rename({'valid_time': 'time'})
            ds_daily = ds[var_name].resample(time='D').max(skipna=True).load()
        
        return ds_daily.reindex(time=self.full_date_range)


class SpatialIndexer:
    """Resolve relacionamentos espaciais entre pontos de grade e estações."""
    def __init__(self, inmet_lats, inmet_lons, era5_lats, era5_lons):
        self.inmet_coords = np.vstack([inmet_lats, inmet_lons]).T
        
        era_lon_grid, era_lat_grid = np.meshgrid(era5_lons, era5_lats)
        self.era5_coords = np.vstack([era_lat_grid.ravel(), era_lon_grid.ravel()]).T

        self._build_trees()

    def _build_trees(self):
        logging.info("Construindo KDTrees espaciais...")
        self.inmet_tree = KDTree(self.inmet_coords)
        self.era5_tree = KDTree(self.era5_coords)

    def get_era5_neighbors(self):
        _, indices = self.era5_tree.query(self.inmet_coords)
        return indices

    def get_inmet_neighbors(self, k=5):
        _, indices = self.inmet_tree.query(self.inmet_coords, k=k+1)
        return indices[:, 1:]


class SeriesBuilder:
    """Constrói a série temporal final aplicando regras de negócio e imputação."""
    def __init__(self, ds_inmet, ds_era5, spatial_indexer: SpatialIndexer):
        self.ds_inmet = ds_inmet
        self.ds_era5 = ds_era5
        self.indexer = spatial_indexer
        
        self.era5_shape = (ds_era5.latitude.size, ds_era5.longitude.size)
        self.era5_indices = self.indexer.get_era5_neighbors()
        self.inmet_neighbor_indices = self.indexer.get_inmet_neighbors(k=5)

    def build_all(self) -> pd.DataFrame:
        results = []
        stations = self.ds_inmet.estacao.values
        
        gust_array = self.ds_inmet['daily_wind_gust_max']
        dir_array = self.ds_inmet.get('daily_wind_direction_at_gust_max', 
                                      xr.full_like(gust_array, np.nan))

        for i, station_code in tqdm(enumerate(stations), total=len(stations), desc="Processando"):
            df_station = self._process_single_station(i, station_code, gust_array, dir_array)
            results.append(df_station)
        
        return pd.concat(results)

    def _process_single_station(self, idx, code, gust_array, dir_array):
        s_inmet_gust = gust_array.isel(estacao=idx).to_series().replace(0, np.nan)
        s_inmet_dir = dir_array.isel(estacao=idx).to_series()
        
        lat_idx, lon_idx = np.unravel_index(self.era5_indices[idx], self.era5_shape)
        s_era5 = self.ds_era5.isel(latitude=lat_idx, longitude=lon_idx).to_series().replace(0, np.nan)

        s_combined = s_inmet_gust.combine(s_era5, np.fmax)
        s_neighbors_max = self._calculate_neighbor_max(idx, s_combined)
        
        s_final_gust = s_combined.fillna(s_neighbors_max).fillna(0)

        s_source = self._determine_source(s_inmet_gust, s_era5, s_neighbors_max)

        df = pd.DataFrame({
            'data': s_final_gust.index,
            'codigo_estacao': code,
            'latitude': self.ds_inmet.latitude.values[idx],
            'longitude': self.ds_inmet.longitude.values[idx],
            'rajada_max_inmet_ms': s_inmet_gust.values,
            'direcao_inmet_graus': s_inmet_dir.values,
            'rajada_max_era5_ms': s_era5.values,
            'rajada_max_final_ms': s_final_gust.values,
            'fonte_valor_final': s_source
        })
        return df

    def _calculate_neighbor_max(self, station_idx, current_series):
        missing_dates = current_series.index[current_series.isna()]
        if missing_dates.empty:
            return pd.Series(np.nan, index=current_series.index)

        neighbor_idxs = self.inmet_neighbor_indices[station_idx]
        neighbor_data = self.ds_inmet['daily_wind_gust_max'].isel(estacao=neighbor_idxs).sel(time=missing_dates)
        
        max_vals = neighbor_data.max(dim='estacao').to_series()
        
        full_series = pd.Series(np.nan, index=current_series.index)
        full_series.update(max_vals)
        return full_series

    def _determine_source(self, s_inmet, s_era, s_neighbors):
        conds = [
            (s_inmet.notna()) & (s_inmet >= s_era.fillna(-9999)),
            (s_inmet.notna()) & (s_era > s_inmet.fillna(-9999)),
            (s_inmet.isna()) & (s_neighbors.notna())
        ]
        choices = ['INMET', 'ERA5', 'VIZINHOS']
        return np.select(conds, choices, default='ZERO')


def main():
    config = {
        'era5_pattern': "era5_data_{ano}.nc",
        'inmet_path': "/path/to/inmet_features.nc",
        'output_csv': "output/serie_consolidada.csv",
        'years': range(2000, 2026)
    }

    loader = DataLoader(config['inmet_path'], config['era5_pattern'], config['years'])
    ds_inmet = loader.load_inmet()
    ds_era5 = loader.load_era5_gust()

    indexer = SpatialIndexer(
        ds_inmet.latitude.values, ds_inmet.longitude.values,
        ds_era5.latitude.values, ds_era5.longitude.values
    )

    builder = SeriesBuilder(ds_inmet, ds_era5, indexer)
    df_final = builder.build_all()

    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    df_final.to_csv(config['output_csv'], index=False, float_format='%.4f')
    logging.info("Processo concluído.")

if __name__ == "__main__":
    main()