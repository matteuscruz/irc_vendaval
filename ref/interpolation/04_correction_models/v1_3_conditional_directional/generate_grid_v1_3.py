"""
MODELO DE CORREÇÃO V1.3: MAGNITUDE CONDICIONAL + DIREÇÃO
--------------------------------------------------------
Descrição:
    Gera grids diários corrigidos contendo:
    1. Rajada Máxima: Corrigida via IDW condicional. Apenas diferenças
       positivas (INMET > ERA5) geram resíduos. Onde ERA5 >= INMET,
       o resíduo é zero (mantendo o limite conservador do ERA5).
    2. Direção do Vento: Interpolada espacialmente a partir dos dados 
       do INMET usando decomposição vetorial (u, v).
"""

import os
import gc
import logging
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorIDWInterpolator:
    def __init__(self, grid_lats, grid_lons, station_lats, station_lons, k=15):
        self._setup_weights(grid_lats, grid_lons, station_lats, station_lons, k)

    def _setup_weights(self, glat, glon, slat, slon, k):
        grid_pts = np.column_stack((np.meshgrid(glon, glat)[1].ravel(), np.meshgrid(glon, glat)[0].ravel()))
        st_pts = np.column_stack((slat, slon))

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(st_pts)
        dists, self.indices = nbrs.kneighbors(grid_pts)
        
        weights = 1.0 / (np.maximum(dists, 1e-6) ** 2.0)
        self.weights_norm = weights / weights.sum(axis=1)[:, np.newaxis]
        self.grid_shape = (len(glat), len(glon))

    def interpolate_scalar(self, data_matrix):
        return self._weighted_average(data_matrix)

    def interpolate_direction(self, direction_matrix):
        rads = np.radians(direction_matrix)
        u = -np.sin(rads)
        v = -np.cos(rads)
        
        u_grid = self._weighted_average(u)
        v_grid = self._weighted_average(v)
        
        res_rad = np.arctan2(-u_grid, -v_grid)
        return (np.degrees(res_rad) + 360) % 360

    def _weighted_average(self, values):
        vals = np.nan_to_num(values, nan=0.0)
        n_time = vals.shape[0]
        output = np.zeros((n_time, self.weights_norm.shape[0]), dtype=np.float32)
        
        for t in range(n_time):
            output[t, :] = np.sum(self.weights_norm * vals[t, self.indices], axis=1)
            
        return output.reshape((n_time, *self.grid_shape))


class GridProcessorV2_1:
    def __init__(self, csv_path, era_pattern, output_dir):
        self.era_pattern = era_pattern
        self.output_dir = output_dir
        
        self.df = pd.read_csv(csv_path)
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.stations = self.df[['codigo_estacao', 'latitude', 'longitude']].drop_duplicates('codigo_estacao')

    def process_year(self, year):
        logging.info(f"Processando V2.1 (Mag Condicional + Dir) para {year}...")
        
        try:
            ds_era = xr.open_mfdataset(self.era_pattern.replace('*', str(year)), chunks={'valid_time': 365})
            ds_era = ds_era['i10fg'].resample(valid_time='D').max().load()
        except Exception as e:
            logging.warning(f"Erro ao ler ERA5 {year}: {e}")
            return

        df_year = self.df[self.df['data'].dt.year == year]
        if df_year.empty: return

        common_dates = df_year['data'].unique()
        
        # --- CORREÇÃO: Pega apenas a interseção de datas que existem em AMBAS as bases ---
        era_dates = pd.to_datetime(ds_era.valid_time.values)
        common_dates = np.intersect1d(common_dates, era_dates)
        
        if len(common_dates) == 0:
            logging.warning(f"Nenhuma data coincidente entre INMET e ERA5 para o ano {year}.")
            return
        # ----------------------------------------------------------------------------------

        mat_gust = df_year.pivot(index='data', columns='codigo_estacao', values='rajada_max_inmet_ms').reindex(common_dates)
        mat_dir = df_year.pivot(index='data', columns='codigo_estacao', values='direcao_inmet_graus').reindex(common_dates)
        
        era_pts = ds_era.sel(
            latitude=xr.DataArray(self.stations.latitude.values, dims='s'),
            longitude=xr.DataArray(self.stations.longitude.values, dims='s'),
            method='nearest'
        ).sel(valid_time=common_dates).values
        
        interpolator = VectorIDWInterpolator(
            ds_era.latitude.values, ds_era.longitude.values,
            self.stations.latitude.values, self.stations.longitude.values
        )

        raw_residuals = mat_gust.values - era_pts
        
        conditional_residuals = np.maximum(raw_residuals, 0.0) 
        
        grid_bias = interpolator.interpolate_scalar(conditional_residuals)
        
        grid_dir = interpolator.interpolate_direction(mat_dir.values)

        self._export_nc(ds_era, grid_bias, grid_dir, common_dates, year)

    def _export_nc(self, ds_era, bias, direction, dates, year):
        coords = {'valid_time': dates, 'latitude': ds_era.latitude, 'longitude': ds_era.longitude}
        
        da_gust = (ds_era.sel(valid_time=dates) + bias).clip(min=0)
        da_dir = xr.DataArray(direction, coords=coords, name='direcao_vento')
        
        ds_out = xr.Dataset({'rajada_max_corrigida': da_gust, 'direcao_vento': da_dir})
        
        comp = {'zlib': True, 'complevel': 4}
        enc = {v: comp for v in ds_out.data_vars}
        
        out_path = os.path.join(self.output_dir, f"grid_v2_1_{year}.nc")
        ds_out.to_netcdf(out_path, encoding=enc)
        logging.info(f"Salvo: {out_path}")
        gc.collect()

def main():
    output_directory = "output_corrigido_v2_1/"
    os.makedirs(output_directory, exist_ok=True)
    
    proc = GridProcessorV2_1("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/data/in/Training_Dataset_INMET_ERA5_Paired.csv", "/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/data/in/era5/era5_data_*.nc", output_directory)
    for ano in range(2000, 2026):
        proc.process_year(ano)

if __name__ == "__main__":
    main()