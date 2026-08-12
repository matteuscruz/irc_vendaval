"""Reescreve ORIGINAL_FEATURES a partir do ERA5-Basin (grade regional, 236/243
estações) em vez do ERA5_Stratified.nc (só 30 estações).

Motivação: com o INMET expandido pra 243 estações, o ERA5_Stratified.nc (dado
horário co-localizado ponto-a-ponto) não cobre as 213 estações novas — usá-lo
deixaria a maioria das features de ORIGINAL_FEATURES como NaN pra 87% das
estações. O ERA5-Basin (grade 0.25°, interpolado por nearest-neighbor em
era5_basin_loader.py) cobre 236/243, então vira a fonte dessas features.

Os NOMES ANTIGOS são preservados (`wind_mag`, `wind_mag_max`, `surface_pressure`
etc.) — só a proveniência muda — porque cluster_mlp.py (hard-check e baseline
ERA5) e bias_corrector.py consomem esses nomes diretamente.

Limitações conhecidas, aceitas pelo usuário:
- `surface_pressure` vira `msl_mean_basin` (pressão ao nível do mar, não de
  superfície) — mesma ordem de grandeza, mas introduz viés sistemático pequeno
  em estações de altitude alta.
- `10m_u/v_component_of_wind` são RECONSTRUÍDOS via velocidade×direção (o
  ERA5-Basin não expõe componentes u/v brutos) — suficiente para os
  consumidores atuais (correção de viés estatística via QDM), mas não são
  valores ERA5 brutos.
- `wind_mag_min` não tem equivalente no ERA5-Basin (só mean/max/std/p90) e é
  removida de ORIGINAL_FEATURES (ver src/pipelines/common.py).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

REQUIRED_BASIN_COLUMNS = [
    "ws_mean_basin", "ws_max_basin", "ws_std_basin",
    "sin_dir_mean_basin", "cos_dir_mean_basin",
    "t2m_mean_basin", "t2m_range_basin", "td_dep_mean_basin",
    "rh_mean_basin", "msl_mean_basin", "d_msl_basin", "tp_sum_basin",
]


def rebuild_original_from_basin(ds_era5: xr.Dataset) -> xr.Dataset:
    """Sobrescreve as colunas de ORIGINAL_FEATURES usando o ERA5-Basin.

    Espera que `ds_era5` já tenha passado pelo merge do ERA5BasinLoader
    (colunas `*_basin` presentes). Levanta KeyError com mensagem clara se
    alguma coluna esperada estiver faltando.
    """
    missing = [c for c in REQUIRED_BASIN_COLUMNS if c not in ds_era5.data_vars]
    if missing:
        raise KeyError(
            f"rebuild_original_from_basin: colunas do ERA5-Basin ausentes: {missing}. "
            "Essa função deve rodar depois do merge de ERA5BasinLoader em load_extended()."
        )

    wind_mean_safe = ds_era5["ws_mean_basin"].clip(0.1)

    # Nomes com dígito inicial ("2m_...", "10m_...") não são identificadores
    # Python válidos como kwarg de .assign() — precisam de dict + **.
    stage1 = {
        # Diretas (mesmo nome antigo, fonte trocada)
        "wind_mag": ds_era5["ws_mean_basin"],
        "wind_mag_max": ds_era5["ws_max_basin"],
        "wind_mag_std": ds_era5["ws_std_basin"],
        "wind_dir_sin": ds_era5["sin_dir_mean_basin"],
        "wind_dir_cos": ds_era5["cos_dir_mean_basin"],
        "2m_temperature": ds_era5["t2m_mean_basin"],
        "t2m_range": ds_era5["t2m_range_basin"],
        "relative_humidity": ds_era5["rh_mean_basin"],
        "surface_pressure": ds_era5["msl_mean_basin"],
        "pressure_tendency": ds_era5["d_msl_basin"],  # já é a diferença diária
        "total_precipitation": ds_era5["tp_sum_basin"],
        # Reconstruídas
        "2m_dewpoint_temperature": ds_era5["t2m_mean_basin"] - ds_era5["td_dep_mean_basin"],
        "10m_u_component_of_wind": ds_era5["ws_mean_basin"] * ds_era5["cos_dir_mean_basin"],
        "10m_v_component_of_wind": ds_era5["ws_mean_basin"] * ds_era5["sin_dir_mean_basin"],
        "gust_factor": ds_era5["ws_max_basin"] / wind_mean_safe,
        # Persistência — recalculada sobre o wind_mag_max novo
        "lag1_wind_mag_max": ds_era5["ws_max_basin"].shift(time=1),
        "lag2_wind_mag_max": ds_era5["ws_max_basin"].shift(time=2),
        "lag3_wind_mag_max": ds_era5["ws_max_basin"].shift(time=3),
        "lag7_wind_mag_max": ds_era5["ws_max_basin"].shift(time=7),
        "rolling7d_wind_mag_max": ds_era5["ws_max_basin"].rolling(
            time=7, min_periods=3
        ).mean(),
        "rolling3d_wind_mag_max": ds_era5["ws_max_basin"].rolling(
            time=3, min_periods=2
        ).mean(),
    }
    ds_era5 = ds_era5.assign(**stage1)

    ds_era5 = ds_era5.assign(
        wind_mag_max_anom=ds_era5["wind_mag_max"] - ds_era5["rolling7d_wind_mag_max"],
        lag1_pressure_tendency=ds_era5["pressure_tendency"].shift(time=1),
    )

    return ds_era5
