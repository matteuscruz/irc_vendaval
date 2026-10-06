"""Features ERA5 novas em teste — `dataset/raw/new_features/<região>/{sl,pl,static}`.

Qualquer região e qualquer variável colocada nessa árvore é lida
automaticamente, sem registrar nada no código:

    new_features/<região>/sl/<nome>_<AAAA>.nc      horário, (valid_time, lat, lon)
    new_features/<região>/pl/<nome>_<AAAA>.nc      horário, + pressure_level
    new_features/<região>/static/<nome>.nc         campo fixo (valid_time=1)

Regras (decididas com o usuário — sem imputação em nenhum ponto):
  - Dimensão espacial: a grade da região TEM de ser um recorte da grade do
    ERA5-Basin (mesmos nós de 0.25°); células fora da máscara da bacia viram
    NaN, então as features novas cobrem exatamente as mesmas células que a
    base antiga. Grade desalinhada é erro, não interpolação.
  - Agregação horária → diária por dia UTC (mesma convenção do ERA5-Basin,
    verificada: ws_max/ws_mean do Basin = max/mean horário por dia UTC).
  - Variável com período incompleto (dia sem as 24 horas, ano faltando ou NaN
    dentro da bacia) em relação às demais variáveis da região é DESCARTADA
    inteira, com aviso — nunca preenchida.
  - Toda feature sai com o prefixo `nf_` (grupo `new_features` em
    src/pipelines/common.py).

O resultado diário em grade fica em cache (`_daily_cache.nc`), refeito sempre
que a lista/tamanho/mtime dos arquivos muda.
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.interp import build_point_indexer, extract_points

NEW_FEATURES_DIRNAME = "new_features"
NEW_FEATURE_PREFIX = "nf_"
BASIN_FILENAME = "ERA5_Features_Basin_2000_2026.nc"
CACHE_FILENAME = "_daily_cache.nc"
GRAVITY = 9.80665

# Estatísticas diárias por variável NetCDF de nível único. Variável ausente
# daqui cai em DEFAULT_STATS.
SL_STATS: dict[str, tuple[str, ...]] = {
    "fg10": ("max", "mean"),
    "cape": ("max", "mean"),
    "fsr": ("mean",),
    "sp": ("mean", "min", "diff"),
}
DEFAULT_STATS = ("mean", "max")
PL_STATS = ("mean",)
# Pares (u, v) viram magnitude horária antes de agregar; os componentes
# isolados não entram como feature.
WIND_PAIRS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "ws10": ("u10", "v10", ("max", "mean", "std")),
    "ws100": ("u100", "v100", ("max", "mean")),
}
STATIC_RENAME = {"z": "orog_height", "lsm": "lsm"}

# Nomes finais (já com prefixo `nf_`) das features conhecidas como estáticas
# (relevo/superfície — 1 valor por célula, sem variação temporal). Usado para
# resolver os grupos `new_features_static`/`new_features_dynamic` em
# src/pipelines/common.py. Um arquivo novo em `static/` com nome diferente
# destes entra em `new_features` normalmente, mas precisa ser adicionado aqui
# para cair também em `new_features_static`.
KNOWN_STATIC_FEATURES = frozenset(
    f"{NEW_FEATURE_PREFIX}{name}"
    for name in ("orog_height", "lsm", "sdor", "isor", "anor", "slor", "sdfor")
)

_YEAR_RE = re.compile(r"^(?P<name>.+)_(?P<year>\d{4})$")
_DROP_COORDS = ("number", "expver", "step", "surface")


def new_features_root(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / NEW_FEATURES_DIRNAME


def discover_regions(raw_dir: str | Path) -> list[Path]:
    root = new_features_root(raw_dir)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and any((p / k).is_dir() for k in ("sl", "pl", "static"))
    )


# ── Leitura ──────────────────────────────────────────────────────────────────

def _open(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars([c for c in _DROP_COORDS if c in ds.coords or c in ds.data_vars])


def _single_var(ds: xr.Dataset, path: Path) -> str:
    names = list(ds.data_vars)
    if len(names) != 1:
        raise ValueError(f"{path}: esperada 1 variável por arquivo, encontradas {names}")
    return names[0]


def _yearly_files(folder: Path) -> dict[str, dict[int, Path]]:
    """{prefixo do arquivo: {ano: caminho}}."""
    out: dict[str, dict[int, Path]] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.nc")):
        m = _YEAR_RE.match(p.stem)
        if not m:
            print(f"[new_features] AVISO: {p} fora do padrão <nome>_<AAAA>.nc — ignorado")
            continue
        out.setdefault(m["name"], {})[int(m["year"])] = p
    return out


# ── Grade e máscara ──────────────────────────────────────────────────────────

def _snap_axis(values: np.ndarray, reference: np.ndarray, axis: str, region: str) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    idx = np.abs(ref[None, :] - np.asarray(values, dtype=float)[:, None]).argmin(axis=1)
    off = np.abs(ref[idx] - values) > 1e-6
    if off.any():
        raise ValueError(
            f"[new_features] {region}: {axis} {np.asarray(values)[off][:5]} não pertence à grade "
            "do ERA5-Basin — a feature nova precisa ter a mesma dimensão espacial da base antiga."
        )
    return ref[idx]


class _RegionGrid:
    """Coordenadas da região alinhadas à grade do Basin + máscara da bacia."""

    def __init__(self, region: Path, basin_path: Path, sample: xr.Dataset) -> None:
        if not basin_path.exists():
            raise FileNotFoundError(
                f"[new_features] {basin_path} é necessário para validar a grade de {region.name}"
            )
        with xr.open_dataset(basin_path) as basin:
            self.lat = _snap_axis(sample.latitude.values, basin.latitude.values, "latitude", region.name)
            self.lon = _snap_axis(sample.longitude.values, basin.longitude.values, "longitude", region.name)
            probe = next(iter(basin.data_vars))
            valid = basin[probe].isel(time=0).sel(latitude=self.lat, longitude=self.lon).notnull()
            self.mask = valid.load().drop_vars("time", errors="ignore")

    def align(self, da: xr.DataArray, source: Path) -> xr.DataArray:
        """Confere que o arquivo está na MESMA grade da região (todo arquivo,
        não só o primeiro) e aplica a máscara da bacia."""
        for axis, ref in (("latitude", self.lat), ("longitude", self.lon)):
            values = np.asarray(da[axis].values, dtype=float)
            if values.shape != ref.shape or np.abs(values - ref).max() > 1e-6:
                raise ValueError(
                    f"[new_features] {source}: {axis} difere da grade da região — a feature "
                    "nova precisa ter a mesma dimensão espacial da base antiga."
                )
        da = da.assign_coords(latitude=self.lat, longitude=self.lon)
        return da.where(self.mask)


# ── Agregação ────────────────────────────────────────────────────────────────

def _daily(da: xr.DataArray, stats: tuple[str, ...]) -> dict[str, xr.DataArray]:
    r = da.resample(time="1D")
    out: dict[str, xr.DataArray] = {}
    with warnings.catch_warnings():
        # Células fora da bacia são NaN o dia todo — o aviso de std vazio é ruído.
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        for s in stats:
            if s == "mean":
                out[s] = r.mean()
            elif s == "max":
                out[s] = r.max()
            elif s == "min":
                out[s] = r.min()
            elif s == "std":
                out[s] = r.std(ddof=0)
            elif s == "diff":
                mean = r.mean()
                out[s] = mean.diff("time").reindex(time=mean.time)
            else:
                raise ValueError(f"estatística desconhecida: {s}")
    return out


def _complete_days(da: xr.DataArray, mask: xr.DataArray) -> set[pd.Timestamp]:
    """Dias UTC com as 24 horas presentes e nenhum NaN dentro da bacia."""
    times = pd.DatetimeIndex(da["time"].values)
    counts = pd.Series(1, index=times).groupby(times.floor("D")).sum()
    bad = da.isnull().where(mask, False)
    bad = bad.any(dim=[d for d in bad.dims if d != "time"])
    bad_days = set(times[np.asarray(bad.values, dtype=bool)].floor("D"))
    return {d for d, n in counts.items() if n == 24 and d not in bad_days}


def _level_label(level) -> str:
    v = float(level)
    return str(int(v)) if v.is_integer() else str(v).replace(".", "p")


def _region_daily(region: Path, basin_path: Path) -> xr.Dataset | None:
    sl = _yearly_files(region / "sl")
    pl = _yearly_files(region / "pl")
    static_files = sorted((region / "static").glob("*.nc")) if (region / "static").is_dir() else []
    first = next((p for g in (*sl.values(), *pl.values()) for p in g.values()), None)
    first = first or (static_files[0] if static_files else None)
    if first is None:
        return None

    with _open(first) as sample:
        grid = _RegionGrid(region, basin_path, sample)
    print(f"[new_features] {region.name}: grade {len(grid.lat)}×{len(grid.lon)}, "
          f"{int(grid.mask.sum())} células dentro da bacia")

    # 1. Cobertura diária por variável dinâmica (sl + pl)
    sources: dict[str, dict[int, Path]] = {}
    var_of: dict[str, str] = {}
    for kind, files in (("sl", sl), ("pl", pl)):
        for prefix, years in files.items():
            with _open(next(iter(years.values()))) as ds:
                var = _single_var(ds, region / kind / prefix)
            key = f"{kind}:{var}"
            sources[key] = years
            var_of[key] = var

    days: dict[str, set[pd.Timestamp]] = {}
    for key, years in sources.items():
        ok: set[pd.Timestamp] = set()
        for path in years.values():
            with _open(path) as ds:
                ok |= _complete_days(grid.align(ds[var_of[key]].load(), path), grid.mask)
        days[key] = ok

    complete: list[str] = []
    if days:
        reference = set().union(*days.values())
        for key, ok in days.items():
            if ok == reference:
                complete.append(key)
            else:
                yrs = sorted(sources[key])
                print(f"[new_features] AVISO: {region.name}/{key} descartada — "
                      f"{len(ok)}/{len(reference)} dias completos (arquivos {yrs[0]}–{yrs[-1]}, "
                      f"{len(yrs)} anos); sem imputação, a variável fica fora.")
        ref_days = pd.DatetimeIndex(sorted(reference))
    else:
        ref_days = pd.DatetimeIndex([])

    # 2. Agregação diária, ano a ano
    sl_ok = {var_of[k] for k in complete if k.startswith("sl:")}
    pl_ok = {var_of[k] for k in complete if k.startswith("pl:")}
    paired = {c for u, v, _ in WIND_PAIRS.values() if {u, v} <= sl_ok for c in (u, v)}
    sl_path = {var_of[k]: sources[k] for k in complete if k.startswith("sl:")}
    pl_path = {var_of[k]: sources[k] for k in complete if k.startswith("pl:")}
    years = sorted({y for k in complete for y in sources[k]})

    per_year: list[xr.Dataset] = []
    for year in years:
        hourly: dict[str, xr.DataArray] = {}
        for var, yp in {**sl_path, **pl_path}.items():
            if year in yp:
                with _open(yp[year]) as ds:
                    hourly[var] = grid.align(ds[var].load(), yp[year])
        feats: dict[str, xr.DataArray] = {}
        for name, (u, v, stats) in WIND_PAIRS.items():
            if u in hourly and v in hourly:
                for s, da in _daily(np.hypot(hourly[u], hourly[v]), stats).items():
                    feats[f"{name}_{s}"] = da
        if all(c in hourly for c in ("u10", "v10", "u100", "v100")):
            shear = np.hypot(hourly["u100"] - hourly["u10"], hourly["v100"] - hourly["v10"])
            feats["shear_100_10_mean"] = _daily(shear, ("mean",))["mean"]
        for var in sorted(sl_ok - paired):
            if var in hourly:
                for s, da in _daily(hourly[var], SL_STATS.get(var, DEFAULT_STATS)).items():
                    feats[f"{var}_{s}"] = da
        for var in sorted(pl_ok):
            if var not in hourly:
                continue
            da = hourly[var]
            level_dim = next((d for d in da.dims if d not in ("time", "latitude", "longitude")), None)
            levels = da[level_dim].values if level_dim else [None]
            for lev in levels:
                sub = da.sel({level_dim: lev}) if level_dim else da
                tag = f"{var}_{_level_label(lev)}" if level_dim else var
                for s, d in _daily(sub.drop_vars(level_dim, errors="ignore"), PL_STATS).items():
                    feats[f"{tag}_{s}"] = d
        if feats:
            per_year.append(xr.Dataset(feats))

    parts: list[xr.Dataset] = []
    if per_year:
        dyn = xr.concat(per_year, dim="time").sortby("time")
        if "sp_diff" in dyn:  # diff atravessa a virada de ano
            dyn["sp_diff"] = dyn["sp_mean"].diff("time").reindex(time=dyn.time)
        parts.append(dyn.sel(time=dyn.time.isin(ref_days.values)))

    # 3. Estáticas
    static: dict[str, xr.DataArray] = {}
    for path in static_files:
        with _open(path) as ds:
            var = _single_var(ds, path)
            da = ds[var].load()
        da = grid.align(da.squeeze("time", drop=True) if "time" in da.dims else da, path)
        if bool(da.isnull().where(grid.mask, False).any()):
            print(f"[new_features] AVISO: {region.name}/static/{path.name} tem NaN dentro da bacia — descartada")
            continue
        if var == "z":
            da = da / GRAVITY
        static[STATIC_RENAME.get(var, var)] = da
    if static:
        parts.append(xr.Dataset(static))

    if not parts:
        return None
    out = xr.merge(parts, join="outer")
    out = out.drop_vars([c for c in out.coords if c not in ("time", "latitude", "longitude")])
    out = out.rename({v: f"{NEW_FEATURE_PREFIX}{v}" for v in out.data_vars})
    if "time" in out.dims:
        print(f"[new_features] {region.name}: {len(out.data_vars)} features, "
              f"{str(out.time.values[0])[:10]} → {str(out.time.values[-1])[:10]}")
    return out


# ── API ──────────────────────────────────────────────────────────────────────

def _fingerprint(raw_dir: Path, regions: list[Path]) -> str:
    h = hashlib.sha1()
    basin = raw_dir / BASIN_FILENAME
    for p in [basin] + sorted(f for r in regions for f in r.rglob("*.nc")):
        if p.exists():
            st = p.stat()
            h.update(f"{p.relative_to(raw_dir)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def load_new_features_grid(raw_dir: str | Path, use_cache: bool = True) -> xr.Dataset | None:
    """Grade diária (time, latitude, longitude) com todas as features novas de
    todas as regiões, nos nós da grade do ERA5-Basin. None se não houver
    nenhuma região em `raw_dir/new_features`."""
    raw_dir = Path(raw_dir)
    regions = discover_regions(raw_dir)
    if not regions:
        return None

    cache = new_features_root(raw_dir) / CACHE_FILENAME
    fp = _fingerprint(raw_dir, regions)
    if use_cache and cache.exists():
        with xr.open_dataset(cache) as cached:
            if cached.attrs.get("fingerprint") == fp:
                print(f"[new_features] Usando cache {cache.name}")
                return cached.load()
        print("[new_features] Arquivos mudaram — refazendo o cache diário")

    merged: xr.Dataset | None = None
    for region in regions:
        ds = _region_daily(region, raw_dir / BASIN_FILENAME)
        if ds is None:
            continue
        merged = ds if merged is None else merged.combine_first(ds)
    if merged is None:
        return None

    merged.attrs["fingerprint"] = fp
    # Nome temporário por processo: containers paralelos (fan-out por cluster
    # no Modal) podem refazer o cache ao mesmo tempo no mesmo volume.
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    try:
        merged.to_netcdf(tmp)
        tmp.replace(cache)
    except OSError as exc:
        print(f"[new_features] AVISO: cache não gravado ({exc}) — segue em memória")
        tmp.unlink(missing_ok=True)
    return merged


def new_features_to_points(
    grid: xr.Dataset, lats, lons, point_ids, point_dim: str, times, interp_method: str,
) -> xr.Dataset:
    """Extrai a grade nas coordenadas dos pontos (estações) com o mesmo método
    grade→ponto do ERA5-Basin; estáticas são repetidas em todos os dias de
    `times`. Pontos fora da grade ficam NaN."""
    indexer = build_point_indexer(grid.latitude.values, grid.longitude.values, lats, lons, interp_method)
    pts = extract_points(grid, indexer, point_dim, point_ids)
    times = pd.DatetimeIndex(times)
    if "time" in pts.dims:
        pts = pts.reindex(time=times)
    else:
        pts = pts.expand_dims(time=times)
    for v in pts.data_vars:
        if "time" not in pts[v].dims:
            pts[v] = pts[v].expand_dims(time=times)
    return pts.transpose("time", point_dim)


# ── Modo horário achatado (estudo RAW) ──────────────────────────────────────
#
# RAMO ADITIVO. O caminho diário acima é o default e não muda uma linha: todas
# as pipelines de produção (cluster_lazy/mlp/lstm) dependem dele.
#
# A diferença é ONDE a agregação deixa de existir. O caminho diário lê horário
# e reduz com `_daily` (mean/max/std). Aqui nada é reduzido: extraímos a grade
# nas estações ainda na resolução horária e entregamos as 24 horas do dia como
# 24 COLUNAS (`hfn_<var>_h<HH>`), para o estudo de features RAW. A motivação é
# a investigação de DJF: `mean`/`max` apagam o perfil intradiário, que é o que
# separa rajada convectiva de evento sinótico.

HOURLY_FLAT_CACHE = "_hourly_flat_cache.parquet"


def _region_hourly_flat(region: Path, basin_path: Path, stations: pd.DataFrame,
                        interp_method: str = "bilinear") -> pd.DataFrame | None:
    """Uma linha por (estação, dia) com 24 colunas por variável da região.

    Reusa integralmente a disciplina do caminho diário: `_RegionGrid` valida
    que a grade é recorte do Basin e mascara fora da bacia, `_complete_days`
    decide quais variáveis têm cobertura suficiente (e descarta inteira a que
    não tem, sem imputar), e as `WIND_PAIRS` viram magnitude ANTES de qualquer
    coisa — `hypot` calculado hora a hora, nunca a partir de médias diárias,
    que dariam outro número.
    """
    from src.feature_study.hourly_flat import HOURLY_FLAT_NEW_PREFIX, flatten_hourly

    sl = _yearly_files(region / "sl")
    first = next((p for years in sl.values() for p in years.values()), None)
    if first is None:
        return None
    with _open(first) as sample:
        grid = _RegionGrid(region, basin_path, sample)

    sources: dict[str, dict[int, Path]] = {}
    for prefix, years in sl.items():
        with _open(next(iter(years.values()))) as ds:
            sources[_single_var(ds, region / "sl" / prefix)] = years

    completas, ref_days = _usable_variables(sources, grid)
    if not completas:
        return None

    indexer = build_point_indexer(
        grid.lat, grid.lon,
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy(), interp_method,
    )
    ids = stations["estacao"].astype(str).to_numpy()
    pares = {c for u, v, _ in WIND_PAIRS.values() if {u, v} <= completas for c in (u, v)}

    partes: list[pd.DataFrame] = []
    for year in sorted({y for v in completas for y in sources[v]}):
        horario: dict[str, xr.DataArray] = {}
        for var in sorted(completas):
            caminho = sources[var].get(year)
            if caminho is not None:
                with _open(caminho) as ds:
                    horario[var] = grid.align(ds[var].load(), caminho)
        if not horario:
            continue
        campos = {v: da for v, da in horario.items() if v not in pares}
        for nome, (u, v, _) in WIND_PAIRS.items():
            if u in horario and v in horario:
                campos[nome] = np.hypot(horario[u], horario[v])
        if all(c in horario for c in ("u10", "v10", "u100", "v100")):
            campos["shear_100_10"] = np.hypot(
                horario["u100"] - horario["u10"], horario["v100"] - horario["v10"],
            )
        partes.append(_flatten_points(campos, grid, indexer, ids, flatten_hourly, HOURLY_FLAT_NEW_PREFIX))

    if not partes:
        return None
    out = pd.concat(partes, ignore_index=True)
    return out[out["time"].isin(ref_days)].reset_index(drop=True)


def _usable_variables(sources: dict[str, dict[int, Path]], grid: "_RegionGrid"):
    """(variáveis com cobertura completa, dias de referência). Mesma regra do
    caminho diário: quem não cobre os mesmos dias que as demais é descartada
    inteira, com aviso — nunca preenchida."""
    dias: dict[str, set[pd.Timestamp]] = {}
    for var, years in sources.items():
        ok: set[pd.Timestamp] = set()
        for path in years.values():
            with _open(path) as ds:
                ok |= _complete_days(grid.align(ds[var].load(), path), grid.mask)
        dias[var] = ok
    if not dias:
        return set(), pd.DatetimeIndex([])
    referencia = set().union(*dias.values())
    completas = set()
    for var, ok in dias.items():
        if ok == referencia:
            completas.add(var)
        else:
            print(f"[new_features] AVISO: {var} descartada do modo horário — "
                  f"{len(ok)}/{len(referencia)} dias completos; sem imputação.")
    return completas, pd.DatetimeIndex(sorted(referencia))


def _flatten_points(campos, grid, indexer, ids, flatten_hourly, prefix) -> pd.DataFrame:
    """Extrai os campos horários nas estações e achata as 24 horas em colunas."""
    from src.feature_study.hourly_source import _require_regular_hourly

    nomes = sorted(campos)
    pts = extract_points(xr.Dataset({k: campos[k] for k in nomes}), indexer, "estacao", ids)
    pts = pts.transpose("time", "estacao")
    times = pd.DatetimeIndex(pts.time.values)
    dias = pd.DatetimeIndex(_require_regular_hourly(times)[:, 0])
    valores = np.stack([np.asarray(pts[v].values, dtype="float64") for v in nomes])
    cols = flatten_hourly(valores, nomes, len(dias), len(ids), prefix)
    idx = pd.MultiIndex.from_product([dias, list(ids)], names=["time", "estacao"])
    return pd.DataFrame(cols, index=idx).reset_index()


def load_new_features_hourly_flat(
    raw_dir: str | Path, stations: pd.DataFrame, interp_method: str = "bilinear",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Features novas horárias achatadas, já nas estações de `stations`
    (colunas `estacao`, `latitude`, `longitude`).

    Cache próprio (`_hourly_flat_cache.parquet`), irmão do diário e com o mesmo
    fingerprint de arquivos — acrescentar um `.nc` novo invalida os dois.
    """
    raw_dir = Path(raw_dir)
    regions = discover_regions(raw_dir)
    if not regions:
        return None

    stations = stations.drop_duplicates("estacao").sort_values("estacao").reset_index(drop=True)
    cache = new_features_root(raw_dir) / HOURLY_FLAT_CACHE
    fp = _fingerprint(raw_dir, regions) + "|" + "|".join(stations["estacao"].astype(str)) + f"|{interp_method}"
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if cached.attrs.get("fingerprint") == fp:
            print(f"[new_features] Usando cache {cache.name}")
            return cached
        print("[new_features] Arquivos ou estações mudaram — refazendo o cache horário")

    merged: pd.DataFrame | None = None
    for region in regions:
        df = _region_hourly_flat(region, raw_dir / BASIN_FILENAME, stations, interp_method)
        if df is None:
            continue
        merged = df if merged is None else merged.merge(df, on=["estacao", "time"], how="outer")
    if merged is None:
        return None

    n_cols = len(merged.columns) - 2
    print(f"[new_features] horário achatado: {n_cols} colunas, {len(merged)} linhas, "
          f"{merged['estacao'].nunique()} estações")
    merged.attrs["fingerprint"] = fp
    try:
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(cache)
    except OSError as exc:
        print(f"[new_features] AVISO: cache horário não gravado ({exc}) — segue em memória")
    return merged


def merge_new_features_hourly_flat(
    df: pd.DataFrame, raw_dir: str | Path, interp_method: str = "bilinear",
    only_stations=None,
) -> pd.DataFrame:
    """Junta as colunas `hfn_*` a `df` por (estacao, time), usando as próprias
    coordenadas das estações do frame — nenhuma lista de estações duplicada em
    outro lugar do código.

    `only_stations` restringe a extração. Sem ele, o frame diário traz TODAS as
    estações de todos os clusters (237 na execução real) e a extração horária
    custa 26× mais do que o estudo usa — as demais são descartadas logo depois
    pela regra de cobertura. As estações de fora ficam NaN nas colunas `hfn_*`,
    que é exatamente o que a completude já faz com elas.
    """
    stations = df[["estacao", "latitude", "longitude"]].drop_duplicates("estacao")
    if only_stations is not None:
        stations = stations[stations["estacao"].astype(str).isin({str(e) for e in only_stations})]
        if stations.empty:
            raise ValueError("nenhuma das estações pedidas está no frame")
    flat = load_new_features_hourly_flat(raw_dir, stations, interp_method)
    if flat is None:
        return df
    df = df.copy()
    df["estacao"] = df["estacao"].astype(str)
    flat["estacao"] = flat["estacao"].astype(str)
    return df.merge(flat, on=["estacao", "time"], how="left")
