"""Loader para dados ERA5 diários (snapshot 18 UTC) — grid Paraná.

Variáveis Single-Level (SL):
    cape, d2m, mslp (msl), t2m, u10, v10, u100, v100

Variáveis Pressure-Level (PL):
    z (geopotencial), t (temperatura), q (umidade), u, v (vento)
    Níveis: 1000, 925, 850, 700, 500, 300 hPa

Os arquivos estão separados por variável×ano.  Este loader:
  1. Concatena todos os anos via open_mfdataset
  2. Interpola nearest-neighbor para as coordenadas das estações INMET
  3. Calcula features derivadas (wind shear, espessura, CAPE/CIN proxies)
  4. Retorna xr.Dataset com dims (time, estacao) — compatível com NetCDFLoader
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr


class ERA518UTCLoader:
    """Carrega ERA5 18UTC (Paraná grid) e interpola para estações INMET."""

    # Variáveis SL esperadas (prefixo → nome NetCDF)
    SL_VARS = {
        "cape": "cape",
        "d2m": "d2m",
        "mslp": "msl",
        "t2m": "t2m",
        "u10": "u10",
        "v10": "v10",
        "u100": "u100",
        "v100": "v100",
    }

    # Variáveis PL
    PL_VARS = {
        "geopotencial_pl": "z",
        "temperatura_pl": "t",
        "umidade_pl": "q",
        "vento_u_pl": "u",
        "vento_v_pl": "v",
    }

    # Níveis de pressão de interesse para features derivadas
    _LEVELS_OF_INTEREST = [850.0, 500.0, 300.0, 1000.0]

    def __init__(self, raw_dir: str) -> None:
        self.base_dir = Path(raw_dir) / "dados_era5_parana_18utc"
        self.sl_dir = self.base_dir / "sl"
        self.pl_dir = self.base_dir / "pl"

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
        """Renomeia `valid_time` → `time` se necessário."""
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        return ds

    @staticmethod
    def _drop_scalar_vars(ds: xr.Dataset) -> xr.Dataset:
        """Remove variáveis auxiliares (number, expver) que vêm do CDS."""
        for v in ("number", "expver"):
            if v in ds:
                ds = ds.drop_vars(v)
            if v in ds.coords:
                ds = ds.drop_vars(v)
        return ds

    def _load_sl_var(self, prefix: str) -> xr.Dataset | None:
        """Carrega uma variável SL concatenando todos os anos disponíveis."""
        pattern = f"{prefix}_*.nc"
        files = sorted(self.sl_dir.glob(pattern))
        if not files:
            return None
        ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
        ds = self._normalize_time(ds)
        ds = self._drop_scalar_vars(ds)
        return ds

    def _load_pl_var(self, prefix: str) -> xr.Dataset | None:
        """Carrega uma variável PL concatenando todos os anos disponíveis."""
        pattern = f"{prefix}_*.nc"
        files = sorted(self.pl_dir.glob(pattern))
        if not files:
            return None
        ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
        ds = self._normalize_time(ds)
        ds = self._drop_scalar_vars(ds)
        return ds

    @staticmethod
    def _interp_to_stations(
        ds: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        station_ids: np.ndarray,
    ) -> xr.Dataset:
        """Interpolação nearest do grid para coordenadas das estações.

        Vetorizado: uma única chamada `.sel()` com indexadores compartilhando
        a dim "estacao" faz as N buscas nearest-neighbor de uma vez (indexação
        "pointwise" do xarray), em vez de um loop Python de N chamadas `.sel()`
        sequenciais — evita releitura repetida do arquivo/grafo dask (chamado
        13x, uma por variável SL/PL; com 243 estações o loop antigo ficava
        muito lento). Estações fora do domínio recebem NaN.
        """
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
        in_bounds = (
            (lats >= lat_min) & (lats <= lat_max)
            & (lons >= lon_min) & (lons <= lon_max)
        )

        lat_da = xr.DataArray(lats, dims="estacao", coords={"estacao": station_ids})
        lon_da = xr.DataArray(lons, dims="estacao", coords={"estacao": station_ids})
        ds_sta = ds.sel(latitude=lat_da, longitude=lon_da, method="nearest")
        ds_sta = ds_sta.drop_vars(
            [c for c in ("latitude", "longitude") if c in ds_sta.coords]
        )

        if not in_bounds.all():
            mask = xr.DataArray(in_bounds, dims="estacao", coords={"estacao": station_ids})
            ds_sta = ds_sta.where(mask)

        return ds_sta

    # ── public API ─────────────────────────────────────────────────────────

    def load(
        self,
        ds_inmet: xr.Dataset,
    ) -> xr.Dataset:
        """Carrega ERA5-18UTC e retorna Dataset alinhado com (time, estacao).

        Parameters
        ----------
        ds_inmet : xr.Dataset
            Dataset INMET com coordenadas latitude/longitude por estação.

        Returns
        -------
        xr.Dataset
            Dataset com variáveis SL e PL derivadas, dims (time, estacao).
            Estações fora do grid Paraná recebem NaN.
        """
        lats = ds_inmet.latitude.values.astype(float)
        lons = ds_inmet.longitude.values.astype(float)
        station_ids = ds_inmet.estacao.values

        print("[ERA5-18UTC] Carregando variáveis Single-Level...")
        sl_datasets = {}
        for prefix, ncvar in self.SL_VARS.items():
            ds = self._load_sl_var(prefix)
            if ds is None:
                print(f"  AVISO: {prefix} não encontrado, pulando.")
                continue
            ds_sta = self._interp_to_stations(ds, lats, lons, station_ids)
            # Renomear variável para incluir sufixo _18z
            for v in list(ds_sta.data_vars):
                ds_sta = ds_sta.rename({v: f"{v}_18z"})
            sl_datasets[prefix] = ds_sta
            ds.close()
            print(f"  {prefix}: OK ({ncvar}_18z)")

        print("[ERA5-18UTC] Carregando variáveis Pressure-Level...")
        pl_datasets = {}
        for prefix, ncvar in self.PL_VARS.items():
            ds = self._load_pl_var(prefix)
            if ds is None:
                print(f"  AVISO: {prefix} não encontrado, pulando.")
                continue
            ds_sta = self._interp_to_stations(ds, lats, lons, station_ids)
            pl_datasets[prefix] = ds_sta
            ds.close()
            print(f"  {prefix}: OK ({ncvar})")

        # ── Merge SL ─────────────────────────────────────────────────
        if sl_datasets:
            ds_sl = xr.merge(list(sl_datasets.values()), compat="override")
        else:
            raise FileNotFoundError("Nenhum arquivo SL encontrado.")

        # ── Features derivadas SL ────────────────────────────────────
        ds_sl = self._compute_sl_derived(ds_sl)

        # ── Features derivadas PL ────────────────────────────────────
        ds_pl_feat = self._compute_pl_derived(pl_datasets)

        # ── Merge final ──────────────────────────────────────────────
        if ds_pl_feat is not None:
            ds_out = xr.merge([ds_sl, ds_pl_feat], compat="override")
        else:
            ds_out = ds_sl

        # Alinhar ao período do INMET
        t_min = ds_inmet.time.min().values
        t_max = ds_inmet.time.max().values
        ds_out = ds_out.sel(time=slice(t_min, t_max))

        # dim="time" (não axis posicional) evita depender da ordem das dims
        # do resultado do .sel() vetorizado.
        first_var = ds_out[next(iter(ds_out.data_vars))]
        n_valid = int(first_var.notnull().any(dim="time").sum())
        print(f"[ERA5-18UTC] Concluído: {len(ds_out.data_vars)} variáveis, "
              f"{n_valid}/{len(station_ids)} estações com dados válidos.")

        return ds_out

    def _compute_sl_derived(self, ds: xr.Dataset) -> xr.Dataset:
        """Calcula features derivadas das variáveis single-level."""
        assigns = {}

        # Magnitude do vento a 10m às 18UTC
        if "u10_18z" in ds and "v10_18z" in ds:
            assigns["wind10m_mag_18z"] = np.sqrt(ds["u10_18z"]**2 + ds["v10_18z"]**2)

        # Magnitude do vento a 100m às 18UTC
        if "u100_18z" in ds and "v100_18z" in ds:
            assigns["wind100m_mag_18z"] = np.sqrt(ds["u100_18z"]**2 + ds["v100_18z"]**2)

        # Wind shear 10m vs 100m (cisalhamento superficial)
        if "wind10m_mag_18z" in assigns and "wind100m_mag_18z" in assigns:
            assigns["wind_shear_sfc_18z"] = assigns["wind100m_mag_18z"] - assigns["wind10m_mag_18z"]

        # Tendência de pressão ao nível do mar (variação diária)
        if "msl_18z" in ds:
            assigns["mslp_tendency_18z"] = ds["msl_18z"].diff("time")

        if assigns:
            ds = ds.assign(**assigns)

        return ds

    def _compute_pl_derived(
        self, pl_datasets: dict[str, xr.Dataset]
    ) -> xr.Dataset | None:
        """Calcula features derivadas das variáveis pressure-level."""
        if not pl_datasets:
            return None

        assigns = {}

        # Wind shear 850–300 hPa
        if "vento_u_pl" in pl_datasets and "vento_v_pl" in pl_datasets:
            ds_u = pl_datasets["vento_u_pl"]
            ds_v = pl_datasets["vento_v_pl"]
            try:
                u850 = ds_u["u"].sel(pressure_level=850.0)
                v850 = ds_v["v"].sel(pressure_level=850.0)
                u300 = ds_u["u"].sel(pressure_level=300.0)
                v300 = ds_v["v"].sel(pressure_level=300.0)

                wind850 = np.sqrt(u850**2 + v850**2)
                wind300 = np.sqrt(u300**2 + v300**2)
                assigns["wind_shear_850_300_18z"] = (wind300 - wind850)

                u500 = ds_u["u"].sel(pressure_level=500.0)
                v500 = ds_v["v"].sel(pressure_level=500.0)
                wind500 = np.sqrt(u500**2 + v500**2)
                assigns["wind_shear_850_500_18z"] = (wind500 - wind850)
            except (KeyError, ValueError) as e:
                print(f"  AVISO: wind shear PL falhou: {e}")

        # Espessura 1000–500 hPa (proporcional à temperatura média da camada)
        if "geopotencial_pl" in pl_datasets:
            ds_z = pl_datasets["geopotencial_pl"]
            try:
                z1000 = ds_z["z"].sel(pressure_level=1000.0) / 9.80665
                z500 = ds_z["z"].sel(pressure_level=500.0) / 9.80665
                assigns["thickness_1000_500_18z"] = z500 - z1000
            except (KeyError, ValueError) as e:
                print(f"  AVISO: espessura PL falhou: {e}")

        # Lapse rate 850–500 hPa (instabilidade)
        if "temperatura_pl" in pl_datasets:
            ds_t = pl_datasets["temperatura_pl"]
            try:
                t850 = ds_t["t"].sel(pressure_level=850.0)
                t500 = ds_t["t"].sel(pressure_level=500.0)
                # Lapse rate em K/km (positivo = instável)
                # Altura aproximada: 850hPa ≈ 1.5km, 500hPa ≈ 5.5km → Δz ≈ 4km
                assigns["lapse_rate_850_500_18z"] = (t850 - t500) / 4.0
            except (KeyError, ValueError) as e:
                print(f"  AVISO: lapse rate PL falhou: {e}")

        # Umidade específica em 850 hPa (proxy de umidade disponível)
        if "umidade_pl" in pl_datasets:
            ds_q = pl_datasets["umidade_pl"]
            try:
                assigns["q850_18z"] = ds_q["q"].sel(pressure_level=850.0)
            except (KeyError, ValueError):
                pass

        if not assigns:
            return None

        # Drop pressure_level dim de cada DataArray
        cleaned = {}
        for k, v in assigns.items():
            if "pressure_level" in v.dims:
                v = v.drop_vars("pressure_level", errors="ignore")
            if "pressure_level" in v.coords:
                v = v.drop_vars("pressure_level", errors="ignore")
            cleaned[k] = v

        return xr.Dataset(cleaned)
