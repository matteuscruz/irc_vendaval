"""
PROJETO VENDAVAL - CATALOGAÇÃO DE EXTREMOS (TORNADOS)
=====================================================

Este script realiza o enriquecimento da série histórica corrigida (NetCDF) 
através da integração de eventos discretos de tornados provenientes de uma base CSV.

FLUXO DE PROCESSAMENTO:
-----------------------
1. Carregamento: O dataset NetCDF consolidado (V2.2 Post-Processed) é carregado em memória.
2. Identificação: Para cada registro no CSV, o script localiza o passo temporal (valid_time)
   e o nó de grade (latitude/longitude) mais próximo da ocorrência.
3. Validação de Magnitude: A alteração só ocorre se a velocidade do tornado no CSV 
   for SUPERIOR à velocidade já presente na grade (postura conservadora).
4. Propagação Espacial: É aplicada uma máscara circular de 20km de raio. Todos os 
   pixels dentro desta área que possuírem ventos inferiores ao do tornado são 
   elevados ao valor do evento.
5. Categorização: O script marca a origem do dado na variável 'categoria'.

DICIONÁRIO DE CATEGORIAS:
-------------------------
As categorias na variável 'categoria' podem ser traduzidas da seguinte forma:

0: PADRÃO (BASELINE)
   O dado provém da nossa base corrigida.
   Não houve influência de registros de tornados externos neste pixel/data.

1: TORNADO (ALVO/EPICENTRO)
   Identifica o pixel exato (nó de grade mais próximo) onde o registro do 
   tornado foi catalogado geograficamente. É o ponto de impacto direto.

2: TORNADO-CONTAMINADO-PROXIMIDADE
   Identifica pixels que estão dentro do raio de influência (20km) do tornado. 
   Estes pontos tiveram sua velocidade original alterada para refletir a 
   propagação do evento extremo na vizinhança.



DETALHES TÉCNICOS:
------------------
- Raio de Propagação: 20km (aprox. 0.18 graus decimais).
- Regra de Decisão: V_final = MAX(V_grade, V_tornado).
- Saída: Novo arquivo NetCDF com as variáveis 'rajada_max_corrigida' atualizada 
  e a nova variável 'categoria' (int8).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def get_propagation_mask(
    ds: xr.Dataset, 
    target_lat: float, 
    target_lon: float, 
    radius_km: float = 20.0
) -> xr.DataArray:
    """Gera uma máscara booleana de pontos dentro de um raio de N km."""
    degree_radius: float = radius_km / 111.0
    
    # CORREÇÃO: Ordem exata de X (lon) e Y (lat)
    lon_grid, lat_grid = np.meshgrid(ds.longitude, ds.latitude)
    dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    
    return xr.DataArray(
        dist <= degree_radius, 
        coords=[ds.latitude, ds.longitude], 
        dims=['latitude', 'longitude']
    )

def apply_events_to_chunk(chunk_ds: xr.Dataset, df_tornados: pd.DataFrame) -> xr.Dataset:
    """
    Função processada por bloco via map_blocks.
    O chunk_ds será processado in-memory contendo apenas a fração definida de tempo.
    """
    chunk_ds = chunk_ds.copy(deep=True)
    
    times = pd.to_datetime(chunk_ds.valid_time.values).normalize()
    datas_unicas_chunk = set(times)
    
    mask = df_tornados['data_evento'].isin(datas_unicas_chunk)
    relevant_tornados = df_tornados[mask]
    
    if relevant_tornados.empty:
        return chunk_ds
        
    for _, row in relevant_tornados.iterrows():
        data = str(row['data_evento']).split(' ')[0]
        v_tornado = float(row['velocidade_vento'])
        t_lat, t_lon = float(row['lat']), float(row['lon'])

        try:
            # 1. Identificar o pixel exato (Impacto Direto) no array EM MEMÓRIA DO PEDACINHO
            target_slice = chunk_ds.sel(valid_time=data)
            pixel_match = target_slice.rajada_max_corrigida.sel(
                latitude=t_lat, longitude=t_lon, method='nearest'
            )
            
            nearest_lat = pixel_match.latitude.values
            nearest_lon = pixel_match.longitude.values

            if v_tornado > pixel_match.values:
                # A) Gerar máscara de proximidade (20km)
                mask_20km = get_propagation_mask(chunk_ds, t_lat, t_lon, radius_km=20.0)
                
                # B) Atualizar a Magnitude na Grade
                grid_vento_t = chunk_ds.rajada_max_corrigida.loc[dict(valid_time=data)]
                to_update = (grid_vento_t < v_tornado) & mask_20km
                
                chunk_ds.rajada_max_corrigida.loc[dict(valid_time=data)] = xr.where(
                    to_update, v_tornado, grid_vento_t
                )
                
                # C) Atualizar Categorias
                grid_cat_t = chunk_ds.categoria.loc[dict(valid_time=data)]
                new_cats = xr.where(to_update, 2, grid_cat_t)
                
                chunk_ds.categoria.loc[dict(
                    valid_time=data, 
                    latitude=nearest_lat, 
                    longitude=nearest_lon
                )] = 1
                
                chunk_ds.categoria.loc[dict(valid_time=data)] = xr.where(
                    (grid_cat_t == 1), 1, new_cats
                )

        except KeyError:
            continue
        except Exception as e:
            # Como executa em worler thread, logging padrão pode se perder ou travar.
            print(f"Erro no registro {data}: {e}")
            continue

    return chunk_ds

def apply_extreme_propagation(
    ds: xr.Dataset, 
    df_tornados: pd.DataFrame
) -> xr.Dataset:
    """Aplica a sobreposição de extremos sem estourar memória, utilizando Dask e map_blocks."""
    # A base original não precisa (.load()) inteiro.
    
    LOGGER.info("Iniciando enriquecimento e categorização agrupada via chunks (xarray.map_blocks)...")
    
    # Criamos o placeholder preguiçoso para a categoria
    if 'categoria' not in ds.data_vars:
        categoria_da = da.zeros(
            ds.rajada_max_corrigida.shape, 
            dtype=np.int8, 
            chunks=ds.rajada_max_corrigida.chunks
        )
        ds['categoria'] = xr.DataArray(
            categoria_da,
            coords=ds.rajada_max_corrigida.coords,
            dims=ds.rajada_max_corrigida.dims,
            name='categoria'
        )
    
    # O DS passa a servir como nosso template completo de tipos e dimensões.
    # O map_blocks injeta apenas os tensores em memória em cada rodada
    ds_enriquecido = xr.map_blocks(
        apply_events_to_chunk,
        ds,
        kwargs={'df_tornados': df_tornados},
        template=ds
    )
    
    return ds_enriquecido

if __name__ == "__main__":
    # Caminhos mantidos conforme sua estrutura
    BASE_PATH: Path = Path("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval")
    NC_PATH: Path = Path('/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/v1_grid_xavier.nc')
    CSV_PATH: Path = BASE_PATH / "data/out/tornados_geoprocessados_v2.csv"
    OUTPUT_NC: Path = Path('v2_inicial.nc')

    try:
        # Abrir o arquivo ativando o modo Dask distribuído em blocos menores!
        # Chunk apenas no tempo: 'valid_time': 100 divide longas séries em fatias temporais manuseáveis
        with xr.open_dataset(NC_PATH, chunks={'valid_time': 100}) as ds:
            LOGGER.info("Dataset NetCDF lazy-loaded (usando Dask)")
            
            df_tornados = pd.read_csv(CSV_PATH)
            # Garantir o truncamento na parte do dia
            df_tornados['data_evento'] = pd.to_datetime(df_tornados['data_evento']).dt.normalize()
            
            ds_enriquecido = apply_extreme_propagation(ds, df_tornados)
            ds_enriquecido.categoria.attrs['description'] = "0: Padrao/Inmet/ERA5, 1: Tornado"
            
            LOGGER.info(f"Gravando novo netcdf em disco gradativamente para: {OUTPUT_NC}")
            
            # Executa de fato todo o processo do map_blocks mandando para o IO incremental
            ds_enriquecido.to_netcdf(OUTPUT_NC)
            
            LOGGER.info("Concluído!")
            
    except Exception as err:
        LOGGER.error(f"Falha na pipeline de extremos: {err}")