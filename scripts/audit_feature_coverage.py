"""Auditoria de cobertura real (não-nula) das features era5_18z / bt55 por
estação x ano, comparando 2000-2019 vs 2020-2024.

Fase 1 do plano de correção ERA5 2000-2024. Roda local, leitura apenas. Usa
os mesmos loaders que a pipeline de treino usa quando `feature_groups` inclui
"era5_18z"/"bt55" (src/data/era5_18utc_loader.py, src/data/bt55_loader.py),
mas evita carregar o ERA5_Stratified.nc horário completo — só precisamos de
lat/lon/time do INMET pra alinhar os loaders.

Uso
---
python3 scripts/audit_feature_coverage.py [--raw-dir dataset/raw]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from src.data.bt55_loader import BT55Loader
from src.data.era5_18utc_loader import ERA518UTCLoader


def coverage_by_station_year(ds: xr.Dataset, varname: str) -> pd.DataFrame:
    """Fração de dias não-nulos por (estacao, ano) para uma variável (time, estacao)."""
    da = ds[varname]
    df = da.to_dataframe(name="val").reset_index()
    df["year"] = df["time"].dt.year
    cov = df.groupby(["estacao", "year"])["val"].apply(lambda s: s.notna().mean())
    return cov.reset_index(name="coverage")


def summarize(cov: pd.DataFrame, label: str) -> None:
    pre2020 = cov[cov["year"] <= 2019]
    post2020 = cov[cov["year"] >= 2020]
    print(f"\n=== {label} ===")
    print(
        f"  2000-2019: mean coverage={pre2020['coverage'].mean():.3f}  "
        f"median={pre2020['coverage'].median():.3f}  "
        f"n_station_years={len(pre2020)}  "
        f"frac_station_years_with_any_data={(pre2020['coverage'] > 0).mean():.3f}"
    )
    print(
        f"  2020-2024: mean coverage={post2020['coverage'].mean():.3f}  "
        f"median={post2020['coverage'].median():.3f}  "
        f"n_station_years={len(post2020)}  "
        f"frac_station_years_with_any_data={(post2020['coverage'] > 0).mean():.3f}"
    )
    by_year = cov.groupby("year")["coverage"].mean()
    print("  Cobertura média por ano:")
    for y, v in by_year.items():
        print(f"    {y}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="dataset/raw")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    print("Abrindo INMET_Stratified.nc (só coords, sem ERA5 horário)...")
    ds_inmet = xr.open_dataset(raw_dir / "INMET_Stratified.nc")
    print(
        f"Estações: {ds_inmet.sizes['estacao']}, período: "
        f"{ds_inmet.time.min().values} .. {ds_inmet.time.max().values}"
    )

    # Alvo INMET em si — densidade de estações ativas ao longo do tempo,
    # referência independente das features era5_18z/bt55 (explica por que
    # TRAIN_SLICE começa em 2008: ver README impresso abaixo).
    cov_target = coverage_by_station_year(ds_inmet, "daily_wind_gust_max")
    summarize(cov_target, "INMET daily_wind_gust_max (densidade de estações)")

    print("\n--- Carregando BT55 ---")
    ds_bt = BT55Loader(str(raw_dir)).load(ds_inmet)
    summarize(coverage_by_station_year(ds_bt, "bt55_flag"), "BT55 (bt55_flag)")
    ds_bt.close()

    print("\n--- Carregando ERA5-18UTC (Paraná) ---")
    ds_18z = ERA518UTCLoader(str(raw_dir)).load(ds_inmet)
    target_var = "wind10m_mag_18z" if "wind10m_mag_18z" in ds_18z else next(iter(ds_18z.data_vars))
    summarize(coverage_by_station_year(ds_18z, target_var), f"ERA5-18UTC ({target_var})")
    ds_18z.close()

    ds_inmet.close()
    print("\nOK - auditoria concluida.")


if __name__ == "__main__":
    main()
