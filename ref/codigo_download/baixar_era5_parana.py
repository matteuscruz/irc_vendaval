"""
Download ERA5 - Paraná, 18 UTC, 2000-2024
==========================================
Baixa dados do ERA5 para análise de vento forte / tempestade severa.
- Pressure levels: t, u, v, q, z em 6 níveis
- Single levels: cape, msl, 10m wind, 100m wind, 2m T, 2m Td

Organização: um arquivo por variável por ano.

Pré-requisitos:
  - Conta no CDS (https://cds.climate.copernicus.eu/)
  - Arquivo ~/.cdsapirc configurado com a chave
  - Termos aceitos para os dois datasets (era5-pressure-levels e era5-single-levels)
  - pip install cdsapi
"""

import cdsapi
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
OUT_BASE = Path(r"C:\Users\piazz\Downloads\Wind_50_100\era5_parana_18utc")
OUT_PL = OUT_BASE / "pl"   # pressure levels
OUT_SL = OUT_BASE / "sl"   # single levels
OUT_PL.mkdir(parents=True, exist_ok=True)
OUT_SL.mkdir(parents=True, exist_ok=True)

ANOS = list(range(2000, 2025))   # 2000 a 2024

# Área: [Norte, Oeste, Sul, Leste]  -> Paraná
AREA = [-22, -55, -27, -48]

HORARIO = "18:00"   # 18 UTC = 15h local Paraná - pico convectivo

MESES = [f"{m:02d}" for m in range(1, 13)]
DIAS  = [f"{d:02d}" for d in range(1, 32)]

# Variáveis em pressure levels (uma por arquivo)
VARS_PL = {
    "geopotencial":  "geopotential",
    "temperatura":   "temperature",
    "vento_u":       "u_component_of_wind",
    "vento_v":       "v_component_of_wind",
    "umidade":       "specific_humidity",
}
NIVEIS_PL = ["300", "500", "700", "850", "925", "1000"]

# Variáveis em single levels (uma por arquivo)
# Atenção: 100m wind só existe a partir de 2008
VARS_SL = {
    "cape":     "convective_available_potential_energy",
    "mslp":     "mean_sea_level_pressure",
    "u10":      "10m_u_component_of_wind",
    "v10":      "10m_v_component_of_wind",
    "u100":     "100m_u_component_of_wind",
    "v100":     "100m_v_component_of_wind",
    "t2m":      "2m_temperature",
    "d2m":      "2m_dewpoint_temperature",
}

# Quantos downloads simultâneos. O CDS aceita ~3 em paralelo por usuário.
PARALELO = 3


# =============================================================================
# FUNÇÕES DE DOWNLOAD
# =============================================================================
def baixar_pl(var_short, var_long, ano):
    """Baixa uma variável de pressure levels para um ano."""
    out = OUT_PL / f"{var_short}_pl_{ano}.nc"
    if out.exists():
        print(f"  [skip] {out.name} já existe")
        return
    print(f"  baixando {out.name} ...")
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "download_format": "unarchived",
            "variable": var_long,
            "pressure_level": NIVEIS_PL,
            "year": str(ano),
            "month": MESES,
            "day": DIAS,
            "time": HORARIO,
            "area": AREA,
        },
        str(out),
    )
    print(f"  [ok]   {out.name}")


def baixar_sl(var_short, var_long, ano):
    """Baixa uma variável de single levels para um ano."""
    out = OUT_SL / f"{var_short}_{ano}.nc"
    if out.exists():
        print(f"  [skip] {out.name} já existe")
        return
    # 100m wind só a partir de 2008
    if var_short in ("u100", "v100") and ano < 2008:
        print(f"  [skip] {out.name} (100m wind indisponível antes de 2008)")
        return
    print(f"  baixando {out.name} ...")
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "download_format": "unarchived",
            "variable": var_long,
            "year": str(ano),
            "month": MESES,
            "day": DIAS,
            "time": HORARIO,
            "area": AREA,
        },
        str(out),
    )
    print(f"  [ok]   {out.name}")


# =============================================================================
# EXECUÇÃO
# =============================================================================
def montar_tarefas():
    tarefas = []
    for var_short, var_long in VARS_PL.items():
        for ano in ANOS:
            tarefas.append(("pl", var_short, var_long, ano))
    for var_short, var_long in VARS_SL.items():
        for ano in ANOS:
            tarefas.append(("sl", var_short, var_long, ano))
    return tarefas


def executar(tipo, var_short, var_long, ano):
    try:
        if tipo == "pl":
            baixar_pl(var_short, var_long, ano)
        else:
            baixar_sl(var_short, var_long, ano)
    except Exception as e:
        print(f"  ERRO {var_short} {ano}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    tarefas = montar_tarefas()
    print(f"Total de pedidos: {len(tarefas)}")
    print(f"Paralelismo: {PARALELO}")
    print(f"Saída pl: {OUT_PL}")
    print(f"Saída sl: {OUT_SL}\n")

    with ThreadPoolExecutor(max_workers=PARALELO) as ex:
        futuros = [ex.submit(executar, *t) for t in tarefas]
        for f in as_completed(futuros):
            f.result()

    print("\nConcluído!")
