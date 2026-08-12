"""
CONSOLIDAÇÃO ANUAL V1
---------------------
Descrição:
    Combina arquivos NetCDF anuais gerados pelo script de grid
    em um único arquivo histórico contínuo e comprimido.

Saída:
    - Arquivo NetCDF único consolidado.
"""

import os
import glob
import logging
import dask
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NetCDFMerger:
    """Manipula a unificação de múltiplos arquivos NetCDF."""
    def __init__(self, input_dir, pattern):
        self.search_path = os.path.join(input_dir, pattern)
        self.files = sorted(glob.glob(self.search_path))
    
    def process(self, output_path):
        if not self.files:
            logging.error(f"Nenhum arquivo encontrado em: {self.search_path}")
            return

        logging.info(f"Consolidando {len(self.files)} arquivos...")
        
        ds = xr.open_mfdataset(self.files, chunks={'valid_time': 365}, parallel=True)
        
        encoding = {
            'rajada_max_corrigida': {
                'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': -9999.0
            }
        }

        logging.info(f"Escrevendo arquivo final: {output_path}")
        with dask.config.set(scheduler='single-threaded'):
            ds.to_netcdf(output_path, encoding=encoding)
        
        logging.info("Consolidação concluída.")

def main():
    config = {
        'input_dir': 'output_corrigido_v1/',
        'pattern': 'era5_corrigido_diario_*.nc',
        'output_file': 'serie_historica_rajada_v1.nc'
    }
    
    merger = NetCDFMerger(config['input_dir'], config['pattern'])
    merger.process(config['output_file'])

if __name__ == "__main__":
    main()