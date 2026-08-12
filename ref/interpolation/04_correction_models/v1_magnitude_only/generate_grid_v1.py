"""
MODELO DE CORREÇÃO V1: MAGNITUDE (IDW)
--------------------------------------
Descrição:
    Gera grids diários de correção de viés para rajadas de vento.
    Calcula o resíduo (Observado - ERA5) nas estações e interpolar
    esse erro espacialmente usando IDW (Inverse Distance Weighting).

    O resultado é um NetCDF anual contendo a rajada corrigida.

Saída:
    - Arquivos NetCDF anuais (era5_corrigido_diario_YYYY.nc).
"""

import os
import gc
import logging
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IDWInterpolator:
    """Motor de interpolação espacial otimizado via álgebra linear."""
    def __init__(self, grid_lats, grid_lons, station_lats, station_lons, k=15, p=2.0):
        self.k = k
        self.p = p
        self._precompute_weights(grid_lats, grid_lons, station_lats, station_lons)

    def _precompute_weights(self, glat, glon, slat, slon):
        logging.info("Pré-calculando pesos IDW...")
        grid_pts = np.column_stack((np.meshgrid(glon, glat)[1].ravel(), np.meshgrid(glon, glat)[0].ravel()))
        st_pts = np.column_stack((slat, slon))

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(st_pts)
        dists, self.indices = nbrs.kneighbors(grid_pts)
        
        dists = np.maximum(dists, 1e-6) # Evita div/0
        weights = 1.0 / (dists ** self.p)
        self.weights_norm = weights / weights.sum(axis=1)[:, np.newaxis]
        self.grid_shape = (len(glat), len(glon))

    def predict(self, values_at_stations: np.ndarray) -> np.ndarray:
        """
        Recebe matriz (Tempo, Estações) e retorna (Tempo, Grid_Lat, Grid_Lon).
        Trata NaNs como 0.0 na soma ponderada.
        """
        vals_filled = np.nan_to_num(values_at_stations, nan=0.0)
        n_time = vals_filled.shape[0]
        n_grid = self.weights_norm.shape[0]
        
        output = np.zeros((n_time, n_grid), dtype=np.float32)
        
        for t in range(n_time):
            neighbors_vals = vals_filled[t, self.indices]
            output[t, :] = np.sum(self.weights_norm * neighbors_vals, axis=1)
            
        return output.reshape((n_time, *self.grid_shape))


class CorrectionPipeline:
    """Orquestra o cálculo de resíduos e aplicação da correção."""
    def __init__(self, csv_path, era_pattern, output_dir):
        self.era_pattern = era_pattern
        self.output_dir = output_dir
        
        self.df_inmet = pd.read_csv(csv_path)
        self.df_inmet['data'] = pd.to_datetime(self.df_inmet['data'])
        self.stations = self.df_inmet[['codigo_estacao', 'latitude', 'longitude']].drop_duplicates('codigo_estacao')

    def execute_year(self, year):
        logging.info(f"Iniciando correção para o ano {year}")
        
        try:
            ds_era = self._load_era5_year(year)
        except IOError:
            return

        residuals, valid_dates = self._calculate_residuals(ds_era, year)
        if residuals is None:
            return

        interpolator = IDWInterpolator(
            ds_era.latitude.values, ds_era.longitude.values,
            self.stations.latitude.values, self.stations.longitude.values
        )
        correction_grid = interpolator.predict(residuals)

        self._save_corrected(ds_era, correction_grid, valid_dates, year)
        gc.collect()

    def _load_era5_year(self, year) -> xr.Dataset:
        fpath = self.era_pattern.replace('*', str(year))
        ds = xr.open_mfdataset(fpath, chunks={'valid_time': 365})
        return ds['i10fg'].resample(valid_time='D').max().load()

    def _calculate_residuals(self, ds_era, year):
        mask = self.df_inmet['data'].dt.year == year
        df_year = self.df_inmet[mask]
        
        if df_year.empty: return None, None

        inmet_matrix = df_year.pivot(index='data', columns='codigo_estacao', values='rajada_max_inmet_ms')
        
        era_points = ds_era.sel(
            latitude=xr.DataArray(self.stations.latitude.values, dims='s'),
            longitude=xr.DataArray(self.stations.longitude.values, dims='s'),
            method='nearest'
        ).to_pandas()
        era_points.columns = self.stations.codigo_estacao.values

        common = inmet_matrix.index.intersection(era_points.index)
        residuals = inmet_matrix.loc[common] - era_points.loc[common]
        
        return residuals.values, common

    def _save_corrected(self, ds_era, correction_data, dates, year):
        da_corr = xr.DataArray(
            correction_data, 
            coords={'valid_time': dates, 'latitude': ds_era.latitude, 'longitude': ds_era.longitude},
            name='bias'
        )
        
        ds_final = (ds_era.sel(valid_time=dates) + da_corr).clip(min=0)
        ds_final.name = 'rajada_max_corrigida'
        
        out_path = os.path.join(self.output_dir, f"era5_corrigido_diario_{year}.nc")
        ds_final.to_netcdf(out_path, encoding={'rajada_max_corrigida': {'zlib': True, 'complevel': 4}})
        logging.info(f"Salvo: {out_path}")


def main():
    config = {
        'csv': "output/serie_consolidada.csv",
        'era_pattern': "era5_data_*.nc",
        'out_dir': "output_corrigido_v1/"
    }
    os.makedirs(config['out_dir'], exist_ok=True)
    
    pipeline = CorrectionPipeline(config['csv'], config['era_pattern'], config['out_dir'])
    
    for ano in range(2000, 2026):
        pipeline.execute_year(ano)

if __name__ == "__main__":
    main()