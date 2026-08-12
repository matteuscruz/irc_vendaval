"""
GPM_MERGIR / MERGE-IR - versao otimizada para BT <= -55 C.

Objetivo:
  1) manter o mesmo produto mensal por pixel do script original;
  2) criar cache diario comprimido para permitir retomada sem repetir o mes;
  3) paralelizar a leitura/processamento dos NetCDFs baixados em cada dia;
  4) registrar logs diarios e mensais mais claros.

Produto final:
  BASE_DIR/04_pixel_monthly/bt55_pixels_YYYY_MM.parquet

Cache intermediario:
  BASE_DIR/03_daily_cache/YYYY/MM/bt55_daily_YYYY_MM_DD.npz

Cada cache diario contem:
  mask, valid, btmin_k, lats, lons

Nos arquivos mensais, alem do BT55 original, sao preservados limiares de
-50/-55/-60/-65 C, temperatura minima e intensidade abaixo de -55 C.

Variaveis de ambiente uteis:
  MERGIR_START_DATE=2003-06-01
  MERGIR_END_DATE=2024-12-31
  MERGIR_KEEP_RAW_NC=false
  MERGIR_MAX_RETRIES=3
  MERGIR_DOWNLOAD_THREADS=8
  MERGIR_READ_WORKERS=4
  MERGIR_CLOUD_HOSTED=false
  MERGIR_LOGIN_STRATEGY=interactive
  MERGIR_FORCE_MONTH=false
  MERGIR_FORCE_DAYS=false
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import calendar
import gc
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
import earthaccess
from tqdm import tqdm


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "sim", "s"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_date(name: str, default: date) -> date:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    y, m, d = [int(x) for x in value.split("-")]
    return date(y, m, d)


def env_bbox(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Le bbox W,S,E,N de uma variavel como ``-53.5,-25.5,-44,-19.5``."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parts = tuple(float(x.strip()) for x in value.split(","))
    if len(parts) != 4:
        raise ValueError(f"{name} deve conter oeste,sul,leste,norte")
    west, south, east, north = parts
    if not (west < east and south < north):
        raise ValueError(f"{name} invalido: {parts}")
    return parts


@dataclass(frozen=True)
class Config:
    base_dir: Path = Path(os.environ.get(
        "MERGIR_BASE_DIR",
        str(Path.home() / "Documents" / "wind_50_100" / "goes_merge_ir_python"),
    ))
    short_name: str = "GPM_MERGIR"
    version: str = "1"
    start_date: date = env_date("MERGIR_START_DATE", date(2000, 6, 1))
    end_date: date = env_date("MERGIR_END_DATE", date(2024, 12, 31))
    bbox: tuple[float, float, float, float] = env_bbox(
        "MERGIR_BBOX", (-54.9, -26.8, -47.8, -22.3)
    )
    bt_threshold_k: float = 218.15
    keep_raw_nc: bool = env_bool("MERGIR_KEEP_RAW_NC", False)
    max_retries: int = env_int("MERGIR_MAX_RETRIES", 3)
    download_threads: int = env_int("MERGIR_DOWNLOAD_THREADS", 8)
    read_workers: int = env_int("MERGIR_READ_WORKERS", 4)
    cloud_hosted: bool = env_bool("MERGIR_CLOUD_HOSTED", False)
    login_strategy: str = os.environ.get("MERGIR_LOGIN_STRATEGY", "interactive")
    force_month: bool = env_bool("MERGIR_FORCE_MONTH", False)
    force_days: bool = env_bool("MERGIR_FORCE_DAYS", False)
    sleep_seconds: int = env_int("MERGIR_SLEEP_SECONDS", 1)

    @property
    def dir_nc(self) -> Path:
        return self.base_dir / "01_nc_brutos"

    @property
    def dir_daily_cache(self) -> Path:
        return self.base_dir / "03_daily_cache"

    @property
    def dir_pixel_monthly(self) -> Path:
        return self.base_dir / "04_pixel_monthly"

    @property
    def dir_log(self) -> Path:
        return self.base_dir / "00_logs"

    @property
    def log_path(self) -> Path:
        return self.dir_log / "log_mergir_pixel_monthly_otimizado.csv"


CFG = Config()


def ensure_dirs() -> None:
    for folder in [CFG.dir_nc, CFG.dir_daily_cache, CFG.dir_pixel_monthly, CFG.dir_log]:
        folder.mkdir(parents=True, exist_ok=True)


def valid_file(path: Path, min_size: int = 1024) -> bool:
    return path.exists() and path.stat().st_size > min_size


def clean_folder(folder: Path) -> None:
    if not folder.exists():
        return
    for p in folder.iterdir():
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as exc:
            warnings.warn(f"Nao consegui apagar {p}: {exc}")


def month_range(start_day: date, end_day: date):
    current = date(start_day.year, start_day.month, 1)
    while current <= end_day:
        last_day = calendar.monthrange(current.year, current.month)[1]
        m_start = max(current, start_day)
        m_end = min(date(current.year, current.month, last_day), end_day)
        yield m_start, m_end
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)


def daterange(start_day: date, end_day: date):
    current = start_day
    while current <= end_day:
        yield current
        current += timedelta(days=1)


def day_temp_folder(day: date) -> Path:
    return CFG.dir_nc / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def daily_cache_file(day: date) -> Path:
    folder = CFG.dir_daily_cache / f"{day.year:04d}" / f"{day.month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"bt55_daily_{day.year:04d}_{day.month:02d}_{day.day:02d}.npz"


def pixel_month_file(year: int, month: int) -> Path:
    return CFG.dir_pixel_monthly / f"bt55_pixels_{year:04d}_{month:02d}.parquet"


def list_nc_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ["*.nc", "*.nc4", "*.NC", "*.NC4"]:
        files.extend(folder.glob(pattern))
    out = []
    for f in sorted(set(files)):
        if valid_file(f):
            out.append(f)
        else:
            try:
                f.unlink()
            except Exception:
                pass
    return out


def save_log(rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(CFG.log_path, index=False, encoding="utf-8-sig")


def login_earthdata() -> None:
    last_error = None
    for attempt in range(1, CFG.max_retries + 1):
        try:
            earthaccess.login(strategy=CFG.login_strategy, persist=True)
            return
        except Exception as exc:
            last_error = exc
            warnings.warn(f"Falha no login Earthdata {attempt}/{CFG.max_retries}: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Nao foi possivel autenticar no Earthdata: {last_error}")


def search_granules_for_day(day: date):
    temporal = (f"{day.isoformat()}T00:00:00Z", f"{day.isoformat()}T23:59:59Z")
    return earthaccess.search_data(
        short_name=CFG.short_name,
        version=CFG.version,
        temporal=temporal,
        bounding_box=CFG.bbox,
        cloud_hosted=CFG.cloud_hosted,
    )


def download_granules(results, out_dir: Path) -> list[Path]:
    if not results:
        return []
    try:
        downloaded = earthaccess.download(results, local_path=str(out_dir), threads=CFG.download_threads)
    except TypeError:
        downloaded = earthaccess.download(results, local_path=str(out_dir))
    paths = [Path(f) for f in downloaded]
    return sorted({p for p in paths if valid_file(p)})


def search_and_download_with_retry(day: date, out_dir: Path) -> tuple[list[Path], int | None, str, str]:
    last_error = ""
    for attempt in range(1, CFG.max_retries + 1):
        try:
            results = search_granules_for_day(day)
            n_found = len(results) if results is not None else 0
            if n_found == 0:
                return [], 0, "sem_granules", ""

            _ = download_granules(results, out_dir)
            nc_files = list_nc_files(out_dir)
            if nc_files:
                return nc_files, n_found, "download_ok", ""

            last_error = "Granules encontrados, mas nenhum NetCDF valido ficou na pasta."
        except Exception as exc:
            last_error = str(exc)
            warnings.warn(f"Tentativa {attempt}/{CFG.max_retries} falhou em {day}: {last_error}")
        time.sleep(5 * attempt)
    return [], None, "erro_download", last_error


def open_dataset_robust(nc_file: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(nc_file, engine="h5netcdf")
    except Exception:
        try:
            return xr.open_dataset(nc_file, engine="netcdf4")
        except Exception:
            return xr.open_dataset(nc_file)


def find_bt_variable(ds: xr.Dataset) -> str:
    for candidate in ["Tb", "tb", "TB", "brightness_temperature", "Brightness_Temperature", "IR"]:
        if candidate in ds.data_vars:
            return candidate
    for var in ds.data_vars:
        dims = set(ds[var].dims)
        if {"lat", "lon"}.issubset(dims) or {"latitude", "longitude"}.issubset(dims):
            return var
    raise ValueError(f"Nao encontrei variavel de BT. Variaveis: {list(ds.data_vars)}")


def normalize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    rename_dict = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon"), ("Latitude", "lat"), ("Longitude", "lon")]:
        if old in da.dims:
            rename_dict[old] = new
    if rename_dict:
        da = da.rename(rename_dict)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Dimensoes inesperadas: {da.dims}")
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180).sortby("lon")
    return da.sortby("lon").sortby("lat")


def subset_bbox(da: xr.DataArray) -> xr.DataArray:
    west, south, east, north = CFG.bbox
    return da.sel(lon=slice(west, east), lat=slice(south, north))


def open_bt_min_from_file(nc_file: Path) -> xr.DataArray:
    ds = open_dataset_robust(nc_file)
    try:
        var = find_bt_variable(ds)
        da = subset_bbox(normalize_lat_lon(ds[var]))
        da_min = da.min(dim="time", skipna=True) if "time" in da.dims else da
        da_min = da_min.astype("float32")
        da_min.name = "btmin_k"
        return da_min.load()
    finally:
        ds.close()


def daily_bt55_arrays(nc_files: list[Path]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    if not nc_files:
        return None, None, None, None, None, "sem_nc"

    bt_mins = []
    workers = max(1, min(CFG.read_workers, len(nc_files)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(open_bt_min_from_file, f): f for f in nc_files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                bt_mins.append(fut.result())
            except Exception as exc:
                warnings.warn(f"Falha lendo {f.name}: {exc}")

    if not bt_mins:
        return None, None, None, None, None, "sem_bt_valido"

    daily_min = xr.concat(bt_mins, dim="obs").min(dim="obs", skipna=True)
    daily_min = daily_min.transpose("lat", "lon").astype("float32")
    valid = daily_min.notnull().values.astype("uint8")
    mask = (daily_min <= CFG.bt_threshold_k).fillna(False).values.astype("uint8")
    btmin_k = daily_min.values.astype("float32")
    lats = daily_min["lat"].values.astype("float32")
    lons = daily_min["lon"].values.astype("float32")
    del daily_min, bt_mins
    gc.collect()
    return mask, valid, btmin_k, lats, lons, "processado"


def load_daily_cache(path: Path):
    with np.load(path) as z:
        btmin_k = z["btmin_k"] if "btmin_k" in z.files else None
        return z["mask"], z["valid"], btmin_k, z["lats"], z["lons"]


def save_daily_cache(path: Path, mask: np.ndarray, valid: np.ndarray, btmin_k: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, mask=mask, valid=valid, btmin_k=btmin_k, lats=lats, lons=lons)
    if path.exists():
        path.unlink()
    tmp.rename(path)


def process_day(day: date) -> dict:
    cache = daily_cache_file(day)
    if valid_file(cache, min_size=256) and not CFG.force_days:
        return {"data": day.isoformat(), "ano": day.year, "mes": day.month, "status": "cache_diario_ok", "erro": ""}

    temp_dir = day_temp_folder(day)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        clean_folder(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        nc_files, n_found, dl_status, dl_error = search_and_download_with_retry(day, temp_dir)
        if dl_status != "download_ok":
            return {
                "data": day.isoformat(),
                "ano": day.year,
                "mes": day.month,
                "status": dl_status,
                "granules_encontrados": n_found,
                "arquivos_nc": len(nc_files),
                "erro": dl_error,
            }

        mask, valid, btmin_k, lats, lons, proc_status = daily_bt55_arrays(nc_files)
        if proc_status != "processado":
            return {
                "data": day.isoformat(),
                "ano": day.year,
                "mes": day.month,
                "status": proc_status,
                "granules_encontrados": n_found,
                "arquivos_nc": len(nc_files),
                "erro": "",
            }

        save_daily_cache(cache, mask, valid, btmin_k, lats, lons)
        return {
            "data": day.isoformat(),
            "ano": day.year,
            "mes": day.month,
            "status": "processado_cache_diario",
            "granules_encontrados": n_found,
            "arquivos_nc": len(nc_files),
            "erro": "",
        }
    except Exception as exc:
        return {"data": day.isoformat(), "ano": day.year, "mes": day.month, "status": "erro_dia", "erro": str(exc)}
    finally:
        if not CFG.keep_raw_nc:
            clean_folder(temp_dir)
        time.sleep(CFG.sleep_seconds)


def assemble_month(month_start: date, month_end: date) -> tuple[Path, str]:
    year = month_start.year
    month = month_start.month
    out_file = pixel_month_file(year, month)
    if valid_file(out_file) and not CFG.force_month:
        return out_file, "ja_existia"

    days = list(daterange(month_start, month_end))
    caches = [daily_cache_file(day) for day in days]
    if not all(valid_file(p, min_size=256) for p in caches):
        missing = [str(p) for p in caches if not valid_file(p, min_size=256)]
        raise RuntimeError(f"Cache diario incompleto para {year}-{month:02d}: {missing[:5]}")

    loaded = [load_daily_cache(p) for p in caches]
    first_mask, first_valid, first_btmin, grid_lats, grid_lons = loaded[0]
    nlat, nlon = first_mask.shape
    print(
        f"Grid mensal {year}-{month:02d}: "
        f"{nlat} linhas lat x {nlon} colunas lon = {nlat * nlon:,} pixels"
    )

    lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)
    n_pixels = nlat * nlon
    df = pd.DataFrame(
        {
            "pixel_id": np.arange(n_pixels, dtype=np.int64),
            "row": np.repeat(np.arange(nlat), nlon),
            "col": np.tile(np.arange(nlon), nlat),
            "lon": lon2d.ravel().astype("float32"),
            "lat": lat2d.ravel().astype("float32"),
            "ano": year,
            "mes": month,
            "ano_mes": f"{year:04d}-{month:02d}",
        }
    )

    day_cols = []
    valid_cols = []
    threshold_cols = {50: [], 55: [], 60: [], 65: []}
    btmin_stack = []
    intensity_stack = []
    for day, (mask, valid, btmin_k, lats, lons) in zip(days, loaded):
        if mask.shape != (nlat, nlon):
            raise ValueError(f"Grade mudou no dia {day}")
        dcol = f"d{day.day:02d}"
        vcol = f"v{day.day:02d}"
        day_cols.append(dcol)
        valid_cols.append(vcol)
        df[dcol] = mask.ravel().astype("uint8")
        df[vcol] = valid.ravel().astype("uint8")
        if btmin_k is not None:
            btmin_stack.append(btmin_k)
            intensity_stack.append(np.where(valid, np.maximum(0.0, 218.15 - btmin_k), np.nan))
            for threshold_c in threshold_cols:
                col = f"b{threshold_c}_d{day.day:02d}"
                threshold_cols[threshold_c].append(col)
                threshold_k = 273.15 - threshold_c
                df[col] = ((btmin_k <= threshold_k) & valid.astype(bool)).ravel().astype("uint8")

    df["dias_bt55"] = df[day_cols].sum(axis=1).astype("uint8")
    df["dias_validos"] = df[valid_cols].sum(axis=1).astype("uint8")
    df["frac_dias_bt55"] = np.where(df["dias_validos"] > 0, df["dias_bt55"] / df["dias_validos"], np.nan).astype("float32")
    if btmin_stack:
        for threshold_c, cols in threshold_cols.items():
            df[f"dias_bt{threshold_c}"] = df[cols].sum(axis=1).astype("uint8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df["bt_min_k"] = np.nanmin(np.stack(btmin_stack), axis=0).ravel().astype("float32")
            df["bt_min_c"] = (df["bt_min_k"] - 273.15).astype("float32")
            df["intensidade_bt55_k_dia"] = np.nansum(np.stack(intensity_stack), axis=0).ravel().astype("float32")
    df.to_parquet(out_file, index=False)
    return out_file, "processado"


def main() -> None:
    ensure_dirs()
    print("Pasta base:")
    print(CFG.base_dir)
    print("Saida mensal por pixel:")
    print(CFG.dir_pixel_monthly)
    print("Cache diario:")
    print(CFG.dir_daily_cache)
    print(f"Periodo: {CFG.start_date} ate {CFG.end_date}")
    print(f"Busca cloud_hosted: {CFG.cloud_hosted}")
    print(f"Threads de download: {CFG.download_threads}")
    print(f"Leitura paralela de NetCDFs: {CFG.read_workers} workers")

    login_earthdata()
    print("Login Earthdata concluido.")

    rows = []
    if CFG.log_path.exists():
        rows = pd.read_csv(CFG.log_path).to_dict("records")
        print(f"Log anterior carregado: {len(rows)} linhas.")

    months = list(month_range(CFG.start_date, CFG.end_date))
    for month_start, month_end in months:
        year = month_start.year
        month = month_start.month
        out_file = pixel_month_file(year, month)
        if valid_file(out_file) and not CFG.force_month:
            print(f"Arquivo mensal ja existe. Pulando: {out_file.name}")
            rows.append({"data": f"{year:04d}-{month:02d}", "ano": year, "mes": month, "status": "mes_ja_existia", "arquivo_pixel_mensal": str(out_file), "erro": ""})
            save_log(rows)
            continue

        print("\n" + "#" * 90)
        print(f"Processando mes: {year}-{month:02d}")
        print("#" * 90)

        days_this_month = list(daterange(month_start, month_end))
        cached_days = sum(valid_file(daily_cache_file(day), min_size=256) for day in days_this_month)
        print(f"Cache diario existente: {cached_days}/{len(days_this_month)} dias")

        for day in tqdm(days_this_month, desc=f"{year}-{month:02d}"):
            row = process_day(day)
            rows.append(row)
            save_log(rows)

        try:
            out_file, status = assemble_month(month_start, month_end)
            rows.append({"data": f"{year:04d}-{month:02d}", "ano": year, "mes": month, "status": f"mes_{status}", "arquivo_pixel_mensal": str(out_file), "erro": ""})
            save_log(rows)
            print(f"Mes salvo: {out_file}")
        except Exception as exc:
            rows.append({"data": f"{year:04d}-{month:02d}", "ano": year, "mes": month, "status": "erro_mes", "arquivo_pixel_mensal": "", "erro": str(exc)})
            save_log(rows)
            warnings.warn(f"Erro no mes {year}-{month:02d}: {exc}")

    print("\nFinalizado.")
    print(CFG.dir_pixel_monthly)
    print(CFG.log_path)


if __name__ == "__main__":
    main()
