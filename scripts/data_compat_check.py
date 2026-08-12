#!/usr/bin/env python3
"""Diagnóstico de compatibilidade: ERA5-18UTC + BT55 vs pipeline existente.

Execução:
    python scripts/data_compat_check.py

Verifica:
  1. Carregamento dos novos datasets
  2. Sobreposição espacial com estações INMET
  3. Sobreposição temporal com ERA5_Stratified / INMET
  4. Extração de features para estações compatíveis
  5. Sanidade dos valores (ranges físicos plausíveis)
  6. Merge via load_extended()
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Adicionar raiz do projeto ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main() -> None:
    import xarray as xr
    from src.data.netcdf_loader import NetCDFLoader
    from src.data.era5_18utc_loader import ERA518UTCLoader
    from src.data.bt55_loader import BT55Loader

    raw_dir = ROOT / "dataset" / "raw"
    results = {}

    # ── 1. Dados existentes ──────────────────────────────────────────────
    separator("1. DADOS EXISTENTES")
    ds_inmet = xr.open_dataset(raw_dir / "INMET_Stratified.nc")
    ds_era5 = xr.open_dataset(raw_dir / "ERA5_Stratified.nc")

    n_stations = len(ds_inmet.estacao)
    lats = ds_inmet.latitude.values.astype(float)
    lons = ds_inmet.longitude.values.astype(float)
    station_ids = ds_inmet.estacao.values

    print(f"INMET: {n_stations} estações, "
          f"lat [{lats.min():.2f}, {lats.max():.2f}], "
          f"lon [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"  Time: {str(ds_inmet.time.values[0])[:10]} → {str(ds_inmet.time.values[-1])[:10]} "
          f"({len(ds_inmet.time)} dias)")
    print(f"ERA5_Stratified: {list(ds_era5.data_vars)}")
    print(f"  Time: {str(ds_era5.time.values[0])[:10]} → {str(ds_era5.time.values[-1])[:10]} "
          f"({len(ds_era5.time)} timesteps)")
    results["existing_stations"] = n_stations

    # ── 2. ERA5-18UTC ────────────────────────────────────────────────────
    separator("2. ERA5-18UTC — CARREGAMENTO")
    t0 = time.time()
    try:
        loader_18 = ERA518UTCLoader(str(raw_dir))
        ds_18 = loader_18.load(ds_inmet)
        t_18 = time.time() - t0
        print(f"\nTempo de carregamento: {t_18:.1f}s")
        print(f"Variáveis: {list(ds_18.data_vars)}")
        print(f"Shape: time={len(ds_18.time)}, estacao={len(ds_18.estacao)}")

        # Verificar estações com dados
        for v in list(ds_18.data_vars)[:5]:
            arr = ds_18[v].values
            valid_stations = (~np.isnan(arr)).any(axis=0)
            valid_count = int(valid_stations.sum())
            valid_ids = [str(station_ids[i]) for i in range(len(station_ids)) if valid_stations[i]]
            val_min = float(np.nanmin(arr)) if valid_count > 0 else float("nan")
            val_max = float(np.nanmax(arr)) if valid_count > 0 else float("nan")
            val_mean = float(np.nanmean(arr)) if valid_count > 0 else float("nan")
            print(f"  {v}: {valid_count}/{n_stations} estações | "
                  f"range=[{val_min:.2f}, {val_max:.2f}] | mean={val_mean:.2f}")

        results["era5_18_vars"] = len(ds_18.data_vars)
        results["era5_18_load_time_s"] = round(t_18, 1)
        results["era5_18_status"] = "OK"
    except Exception as e:
        print(f"ERRO: {e}")
        results["era5_18_status"] = f"ERRO: {e}"

    # ── 3. BT55 ──────────────────────────────────────────────────────────
    separator("3. BT55 — CARREGAMENTO")
    t0 = time.time()
    try:
        loader_bt = BT55Loader(str(raw_dir))
        ds_bt = loader_bt.load(ds_inmet)
        t_bt = time.time() - t0
        print(f"\nTempo de carregamento: {t_bt:.1f}s")
        print(f"Variáveis: {list(ds_bt.data_vars)}")
        print(f"Shape: time={len(ds_bt.time)}, estacao={len(ds_bt.estacao)}")

        for v in ds_bt.data_vars:
            arr = ds_bt[v].values
            valid_stations = (~np.isnan(arr)).any(axis=0)
            valid_count = int(valid_stations.sum())
            val_min = float(np.nanmin(arr)) if valid_count > 0 else float("nan")
            val_max = float(np.nanmax(arr)) if valid_count > 0 else float("nan")
            val_mean = float(np.nanmean(arr)) if valid_count > 0 else float("nan")
            print(f"  {v}: {valid_count}/{n_stations} estações | "
                  f"range=[{val_min:.4f}, {val_max:.4f}] | mean={val_mean:.4f}")

        results["bt55_vars"] = len(ds_bt.data_vars)
        results["bt55_load_time_s"] = round(t_bt, 1)
        results["bt55_status"] = "OK"
    except Exception as e:
        print(f"ERRO: {e}")
        results["bt55_status"] = f"ERRO: {e}"

    # ── 4. Verificação de ranges físicos ─────────────────────────────────
    separator("4. SANIDADE — RANGES FÍSICOS")
    checks = {
        "cape_18z": (0, 8000, "J/kg"),          # CAPE: 0–8000 J/kg
        "msl_18z": (90000, 108000, "Pa"),        # MSLP: 900–1080 hPa
        "wind10m_mag_18z": (0, 50, "m/s"),       # Vento 10m: 0–50 m/s
        "wind100m_mag_18z": (0, 80, "m/s"),      # Vento 100m: 0–80 m/s
        "thickness_1000_500_18z": (4800, 6000, "m"),  # Espessura: 4800–6000 m
        "lapse_rate_850_500_18z": (0, 15, "K/km"),    # Lapse rate: 0–15 K/km
        "bt55_flag": (0, 1, "flag"),             # Binário
        "bt55_rolling3d": (0, 1, "frac"),        # Fração
    }

    all_ok = True
    for var, (vmin, vmax, unit) in checks.items():
        ds_check = ds_18 if "18z" in var else ds_bt
        if var not in ds_check:
            print(f"  {var}: NÃO ENCONTRADA (pulando)")
            continue
        arr = ds_check[var].values
        actual_min = float(np.nanmin(arr))
        actual_max = float(np.nanmax(arr))
        ok = actual_min >= vmin * 0.8 and actual_max <= vmax * 1.2  # 20% margem
        status = "✓" if ok else "✗ FORA DO ESPERADO"
        if not ok:
            all_ok = False
        print(f"  {var}: [{actual_min:.2f}, {actual_max:.2f}] {unit} "
              f"(esperado: [{vmin}, {vmax}]) {status}")

    results["physical_sanity"] = "OK" if all_ok else "WARNINGS"

    # ── 5. Merge via load_extended() ─────────────────────────────────────
    separator("5. MERGE — load_extended()")
    t0 = time.time()
    try:
        loader = NetCDFLoader(str(raw_dir))
        ds_inmet_ext, ds_era5_ext = loader.load_extended()
        t_ext = time.time() - t0
        print(f"\nTempo total load_extended: {t_ext:.1f}s")
        print(f"Variáveis totais no Dataset: {len(ds_era5_ext.data_vars)}")
        print(f"Lista completa:")
        for i, v in enumerate(sorted(ds_era5_ext.data_vars)):
            has_data = (~np.isnan(ds_era5_ext[v].values)).any()
            print(f"  {i+1:2d}. {v} {'✓' if has_data else '(all NaN)'}")

        results["extended_total_vars"] = len(ds_era5_ext.data_vars)
        results["extended_load_time_s"] = round(t_ext, 1)
        results["extended_status"] = "OK"
    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()
        results["extended_status"] = f"ERRO: {e}"

    # ── 6. Resumo ────────────────────────────────────────────────────────
    separator("6. RESUMO")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Salvar JSON
    import json
    out_path = ROOT / "artifacts" / "data_compat_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo: {out_path}")


if __name__ == "__main__":
    main()
