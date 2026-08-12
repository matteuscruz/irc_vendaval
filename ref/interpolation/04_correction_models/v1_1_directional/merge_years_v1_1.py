"""
CONSOLIDAÇÃO ANUAL V2
---------------------
Descrição:
    Combina arquivos NetCDF anuais do modelo V2 (Magnitude + Direção).
"""

import os
import glob
import logging
import xarray as xr
import dask

CONFIG = {
    'INPUT_DIR': 'output_corrigido_final/',
    'INPUT_PATTERN': 'grid_corrigido_*.nc',
    
    'FINAL_FILE': 'serie_historica_rajada_2000_2025_corrigida.nc',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    search_path = os.path.join(CONFIG['INPUT_DIR'], CONFIG['INPUT_PATTERN'])
    logger.info(f"Buscando arquivos anuais em: {search_path}")
    
    files = sorted(glob.glob(search_path))
    
    if not files:
        logger.error("Nenhum arquivo encontrado! Verifique o caminho.")
        return

    logger.info(f"Encontrados {len(files)} arquivos. Iniciando unificação (Lazy Load)...")
    
    ds_combined = xr.open_mfdataset(files, chunks={'valid_time': 365}, parallel=True)
    
    encoding = {
        'rajada_max_corrigida': {
            'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': -9999.0
        },
        'direcao_vento': {
            'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': -9999.0
        }
    }
    
    logger.info(f"Salvando arquivo consolidado em: {CONFIG['FINAL_FILE']}")
    logger.info("Isso pode demorar alguns minutos. O Dask está processando...")
    
    with dask.config.set(scheduler='single-threaded'):
        ds_combined.to_netcdf(CONFIG['FINAL_FILE'], encoding=encoding)
    
    logger.info("✅ Arquivo consolidado salvo com sucesso!")
    
    logger.info("Verificando integridade do arquivo final...")
    ds_check = xr.open_dataset(CONFIG['FINAL_FILE'])
    logger.info(f"Dimensões finais: {ds_check.sizes}")
    logger.info(f"Variáveis finais: {list(ds_check.data_vars)}")

if __name__ == "__main__":
    main()