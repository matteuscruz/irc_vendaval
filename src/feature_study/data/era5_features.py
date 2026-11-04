"""As colunas selecionadas (`config/features_maping.json`) calculadas direto do ERA5 BRUTO, para qualquer cluster.

Os parquets do cluster 3 vieram prontos de um pipeline externo. Para aplicar o modelo a outro cluster (ex.: 12),
este módulo refaz as mesmas colunas a partir de `dataset/raw/new_features/cluster<N>/{sl,static}`, já no desenho
do estudo: uma linha por (célula, dia UTC) na hora de maior `gust10fg` do dia.

Regras, conferidas contra `test_inference_cluster_3.parquet` (valores idênticos):
  - vizinhança: células da grade BRUTA inteira a até 75/250 km (haversine, R = 6371 km) do centro da célula;
    mediana, máximo e desvio-padrão populacional (ddof=0);
  - `c75_250` = mediana(75) − mediana(250); `l75_250` = máximo(75) − mediana(250);
  - `fracpos` = fração das células em 75 km com valor > 0;
  - defasagens na própria célula: `lag1h` = x[t−1], `max_prev3h` = máx(x[t−3..t−1]),
    `std_prev3h` = desvio amostral (ddof=1) de x[t−3..t−1], `delta3h` = x[t] − x[t−3];
  - `hora_solar_sin` = sen(2π·h/24), h = (hora UTC + lon/15) mod 24;
  - estáticos (`anor`): valor da célula e média/desvio/mín/máx em 75 e 250 km;
  - células de saída = as de centro dentro do polígono do cluster (a regra que dá as 74 do cluster 3).
NÃO conferidas (o bruto do cluster 3 não tem `mcpr` nem `msl` aqui): `fracpos` e `l75_250` do `mcpr` seguem a mesma
regra do `cape`; `grad_mslp_hpa_100km` é o módulo do gradiente de `msl` por diferença central, em hPa/100 km, e
`grad_mslp_r250_median` a mediana dele em 250 km.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

R_TERRA = 6371.0
RAIOS = (75.0, 250.0)
LAGGED = ("gust10fg", "w10", "cape", "mcpr")
KM_GRAU = 111.32


def haversine(la1, lo1, la2, lo2):
    p = np.pi / 180
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * R_TERRA * np.arcsin(np.sqrt(a))


def _open(path) -> xr.DataArray:
    d = xr.open_dataset(path)
    v = d[list(d.data_vars)[0]]
    if "valid_time" in v.dims:
        v = v.rename(valid_time="time")
    return v


def load_year(raw_dir, var: str, year: int, tail_hours: int = 3) -> xr.DataArray:
    """(time, latitude, longitude) do ano, com as últimas `tail_hours` horas do ano anterior na frente (para as
    defasagens de 1º de janeiro); sem o ano anterior, essas horas ficam NaN."""
    sl = Path(raw_dir) / "sl"
    cur = _open(sl / f"{var}_{year}.nc").astype("float32")
    prev_p = sl / f"{var}_{year - 1}.nc"
    if prev_p.exists():
        prev = _open(prev_p).astype("float32").isel(time=slice(-tail_hours, None))
    else:
        t0 = pd.to_datetime(cur.time.values[0])
        prev = xr.full_like(cur.isel(time=slice(0, tail_hours)), np.nan)
        prev = prev.assign_coords(time=pd.date_range(t0 - pd.Timedelta(hours=tail_hours), periods=tail_hours, freq="h"))
    out = xr.concat([prev, cur], dim="time")
    dt = np.diff(out.time.values).astype("timedelta64[h]").astype(int)
    if not (dt == 1).all():
        raise ValueError(f"{var} {year}: série horária irregular")
    return out


def grad_mslp(msl_pa: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Módulo do gradiente de msl em hPa/100 km, por diferença central. `msl_pa` (..., lat, lon) em Pa."""
    dlat = np.gradient(lat) * KM_GRAU                              # km por passo de latitude
    dlon = np.gradient(lon)[None, :] * KM_GRAU * np.cos(np.deg2rad(lat))[:, None]
    hpa = msl_pa / 100.0
    gy = np.gradient(hpa, axis=-2) / dlat[:, None]
    gx = np.gradient(hpa, axis=-1) / dlon
    if np.isnan(hpa).any():                                          # grade com buracos: lado disponível
        gy, gx = _gradiente_com_buracos(hpa, -2) / dlat[:, None], _gradiente_com_buracos(hpa, -1) / dlon
    return np.sqrt(gx ** 2 + gy ** 2) * 100.0


def grid_of(raw_dir) -> tuple[np.ndarray, np.ndarray]:
    """(latitudes, longitudes) da grade bruta: do estático `static/anor.nc` se houver, senão do primeiro
    `sl/gust10fg_<ano>.nc` (downloads como o de RS/SC não trazem `static/`)."""
    raw_dir = Path(raw_dir)
    if raw_dir.is_file():                                            # arquivo por pontos
        ds = xr.open_dataset(raw_dir)
        return (np.sort(np.unique(np.round(ds.lat.values.astype(float), 3)))[::-1],
                np.sort(np.unique(np.round(ds.lon.values.astype(float), 3))))
    st = raw_dir / "static" / "anor.nc"
    src = st if st.exists() else sorted((raw_dir / "sl").glob("gust10fg_*.nc"))[0]
    v = _open(src)
    return v.latitude.values, v.longitude.values


def has_static(raw_dir, name: str = "anor") -> bool:
    return (Path(raw_dir) / "static" / f"{name}.nc").exists()


def static_features(raw_dir, cells, name: str = "anor") -> pd.DataFrame:
    v = _open(Path(raw_dir) / "static" / f"{name}.nc").squeeze(drop=True)
    la2, lo2 = np.meshgrid(v.latitude.values, v.longitude.values, indexing="ij")
    vals = v.values
    linhas = []
    for la, lo in cells:
        d = haversine(la, lo, la2, lo2)
        row = {"lat": la, "lon": lo, f"{name}_ponto": float(v.sel(latitude=la, longitude=lo, method="nearest"))}
        for r in RAIOS:
            x = vals[d <= r]
            k = int(r)
            row.update({f"{name}_r{k}_mean": x.mean(), f"{name}_r{k}_std": x.std(), f"{name}_r{k}_min": x.min(),
                        f"{name}_r{k}_max": x.max()})
        linhas.append(row)
    return pd.DataFrame(linhas)


def _neigh(field_t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """field_t (n_dias, lat, lon) → (n_dias, k) valores da vizinhança."""
    return field_t[:, mask]


VARIAVEIS = ("gust10fg", "u10", "v10", "u100", "v100", "cape", "mcpr", "msl")


def _gradiente_com_buracos(x: np.ndarray, eixo: int) -> np.ndarray:
    """Diferença central onde há os dois vizinhos, unilateral onde há um, NaN onde não há nenhum (por passo)."""
    fr = np.full_like(x, np.nan)
    tr = np.full_like(x, np.nan)
    a = np.moveaxis(x, eixo, 0)
    np.moveaxis(fr, eixo, 0)[:-1] = a[1:] - a[:-1]
    np.moveaxis(tr, eixo, 0)[1:] = a[1:] - a[:-1]
    n = np.isfinite(fr).astype("float32") + np.isfinite(tr).astype("float32")
    soma = np.nan_to_num(fr) + np.nan_to_num(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        return soma / n


def _load_points_file(path, year: int, variables, tail_hours: int):
    """Arquivo por pontos (`point`, `time`, com `lat`, `lon`) como campos retangulares, NaN fora dos pontos."""
    ds = xr.open_dataset(path)
    t = pd.to_datetime(ds.time.values)
    fim = np.searchsorted(t, np.datetime64(f"{year + 1}-01-01"))
    ini = np.searchsorted(t, np.datetime64(f"{year}-01-01"))
    la_p, lo_p = np.round(ds.lat.values.astype(float), 3), np.round(ds.lon.values.astype(float), 3)
    lat, lon = np.sort(np.unique(la_p))[::-1], np.sort(np.unique(lo_p))
    ii, jj = {x: k for k, x in enumerate(lat)}, {x: k for k, x in enumerate(lon)}
    linhas, colunas = np.array([ii[x] for x in la_p]), np.array([jj[x] for x in lo_p])
    arr = {}
    for v in variables:
        vals = ds[v].isel(time=slice(max(ini - tail_hours, 0), fim)).values.astype("float32")       # (point, tempo)
        falta = tail_hours - min(ini, tail_hours)                    # antes do primeiro ano: NaN
        c = np.full((falta + vals.shape[1], len(lat), len(lon)), np.nan, dtype="float32")
        c[falta:, linhas, colunas] = vals.T
        arr[v] = c
    times = pd.date_range(pd.Timestamp(f"{year}-01-01") - pd.Timedelta(hours=tail_hours),
                          periods=arr["gust10fg"].shape[0], freq="h")
    return arr, lat, lon, times


def _load_arrays(raw_dir, year: int, variables, tail_hours: int):
    """Campos do ano (com `tail_hours` horas do ano anterior na frente), já com `w10`/`w100`."""
    raw_dir = Path(raw_dir)
    if raw_dir.is_file():
        arr, lat, lon, times = _load_points_file(raw_dir, year, variables, tail_hours)
        arr["w10"] = np.sqrt(arr["u10"] ** 2 + arr["v10"] ** 2)
        arr["w100"] = np.sqrt(arr["u100"] ** 2 + arr["v100"] ** 2)
        return arr, lat, lon, times
    disp = [v for v in variables if (raw_dir / "sl" / f"{v}_{year}.nc").exists()]
    if "gust10fg" not in disp:
        raise FileNotFoundError(f"gust10fg_{year}.nc ausente em {raw_dir / 'sl'}")
    f = {v: load_year(raw_dir, v, year, tail_hours=tail_hours) for v in disp}
    ref = f["gust10fg"]
    arr = {v: f[v].values for v in disp}
    if {"u10", "v10"} <= set(disp):
        arr["w10"] = np.sqrt(arr["u10"] ** 2 + arr["v10"] ** 2)
    if {"u100", "v100"} <= set(disp):
        arr["w100"] = np.sqrt(arr["u100"] ** 2 + arr["v100"] ** 2)
    return arr, ref.latitude.values, ref.longitude.values, pd.to_datetime(ref.time.values)


def _peak_index(arr, i, j, off: int):
    """(índice da hora do pico em `times`, índice do dia) por dia com as 24 horas de `gust10fg`."""
    n_h = arr["gust10fg"].shape[0] - off
    if n_h % 24:
        raise ValueError(f"{n_h} horas, não é múltiplo de 24")
    n_d = n_h // 24
    g = arr["gust10fg"][off:, i, j].reshape(n_d, 24)
    ok = ~np.isnan(g).any(axis=1)
    h = np.nanargmax(np.where(np.isnan(g), -np.inf, g), axis=1)
    t = off + np.arange(n_d) * 24 + h
    return t[ok], np.flatnonzero(ok)


def _cell_features(arr, t, i, j, lo, m75, m250, times, lat, lon) -> dict:
    """As colunas da célula (i, j) nos instantes `t` (índices em `times`). Regras no cabeçalho do módulo."""
    out = {}
    for v in ("gust10fg", "v10", "v100", "w10", "w100", "cape", "mcpr"):
        if v in arr:
            out[v] = arr[v][t, i, j]
    hs = (times[t].hour + lo / 15.0) % 24
    out["hora_solar_sin"] = np.sin(2 * np.pi * np.asarray(hs) / 24)
    for v in LAGGED:
        if v not in arr:
            continue
        s = arr[v][:, i, j]
        prev3 = np.stack([s[t - 3], s[t - 2], s[t - 1]], axis=1)
        out[f"{v}_lag1h"] = s[t - 1]
        out[f"{v}_max_prev3h"] = prev3.max(axis=1)
        out[f"{v}_std_prev3h"] = prev3.std(axis=1, ddof=1)
        out[f"{v}_delta3h"] = s[t] - s[t - 3]

    def viz(v, m):
        return _neigh(arr[v][t], m)

    g75, g250 = viz("gust10fg", m75), viz("gust10fg", m250)
    out.update({"gust_r75_median": np.median(g75, 1), "gust_r75_max": g75.max(1), "gust_r75_std": g75.std(1),
                "gust_r250_median": np.median(g250, 1), "gust_r250_max": g250.max(1),
                "gust_r250_std": g250.std(1)})
    if "w10" in arr:
        w75, w250 = viz("w10", m75), viz("w10", m250)
        out.update({"w10_r75_median": np.median(w75, 1), "w10_r75_max": w75.max(1), "w10_r75_std": w75.std(1),
                    "w10_r250_median": np.median(w250, 1), "w10_r250_std": w250.std(1)})
    for v in ("cape", "mcpr"):
        if v not in arr:
            continue
        a75, a250 = viz(v, m75), viz(v, m250)
        med75, med250 = np.median(a75, 1), np.median(a250, 1)
        out.update({f"{v}_r75_median": med75, f"{v}_r75_max": a75.max(1), f"{v}_r75_std": a75.std(1),
                    f"{v}_r75_fracpos": (a75 > 0).mean(1), f"{v}_r250_median": med250,
                    f"{v}_r250_max": a250.max(1), f"{v}_c75_250": med75 - med250,
                    f"{v}_l75_250": a75.max(1) - med250})
    if "msl" in arr:
        gm = grad_mslp(arr["msl"][t], lat, lon)
        out["grad_mslp_hpa_100km"] = gm[:, i, j]
        g250m = gm[:, m250]
        out["grad_mslp_r250_median"] = np.nanmedian(g250m, 1) if np.isnan(g250m).any() else np.median(g250m, 1)
    return out


def _cell_index(lat, lon):
    return ({round(float(x), 4): i for i, x in enumerate(lat)}, {round(float(x), 4): j for j, x in enumerate(lon)})


def daily_peak_year(raw_dir, cells, year: int, variables=VARIAVEIS) -> pd.DataFrame:
    """Linhas (célula, dia) do ano na hora do pico de `gust10fg`, com as colunas que as `variables` disponíveis
    permitem. Dia com alguma hora de `gust10fg` faltando é descartado (nada é imputado)."""
    off = 3                                                         # horas do ano anterior (defasagens)
    arr, lat, lon, times = _load_arrays(raw_dir, year, variables, off)
    dias = times[off::24]
    la2, lo2 = np.meshgrid(lat, lon, indexing="ij")
    ilat, ilon = _cell_index(lat, lon)
    existe = np.isfinite(arr["gust10fg"]).any(axis=0)                # vizinhança: só pontos com dado
    partes = []
    for la, lo in cells:
        i, j = ilat[round(la, 4)], ilon[round(lo, 4)]
        t, dd = _peak_index(arr, i, j, off)
        d = haversine(la, lo, la2, lo2)
        out = {"lat": np.full(len(t), la), "lon": np.full(len(t), lo), "time": dias[dd], "hora_pico": times[t],
               **_cell_features(arr, t, i, j, lo, (d <= RAIOS[0]) & existe, (d <= RAIOS[1]) & existe, times, lat, lon)}
        partes.append(pd.DataFrame(out))
    return pd.concat(partes, ignore_index=True)


def peak_windows_year(raw_dir, cells, year: int, columns, window: int = 24,
                      variables=VARIAVEIS) -> tuple[pd.DataFrame, np.ndarray]:
    """Janelas para a LSTM: para cada (célula, dia) do ano, as `columns` em cada uma das `window` horas que
    terminam na hora do pico (a última posição é o próprio pico, igual à linha de `daily_peak_year`).

    Devolve `(meta, X)`: `meta` com `lat, lon, time, hora_pico` e `X` float32 `(n, window, len(columns))`. Hora
    sem dado (ex.: antes de 2000) vira NaN; nada é imputado. As regras de cada coluna são as mesmas da linha
    no pico, só que aplicadas em t−(window−1) … t."""
    columns = list(columns)
    off = window + 3                                                # janela + 3 h das defasagens
    arr, lat, lon, times = _load_arrays(raw_dir, year, variables, off)
    dias = times[off::24]
    la2, lo2 = np.meshgrid(lat, lon, indexing="ij")
    ilat, ilon = _cell_index(lat, lon)
    existe = np.isfinite(arr["gust10fg"]).any(axis=0)                # vizinhança: só pontos com dado
    metas, xs = [], []
    for la, lo in cells:
        i, j = ilat[round(la, 4)], ilon[round(lo, 4)]
        t, dd = _peak_index(arr, i, j, off)
        d = haversine(la, lo, la2, lo2)
        m75, m250 = (d <= RAIOS[0]) & existe, (d <= RAIOS[1]) & existe
        x = np.full((len(t), window, len(columns)), np.nan, dtype="float32")
        for k in range(window):
            f = _cell_features(arr, t - (window - 1 - k), i, j, lo, m75, m250, times, lat, lon)
            for c, col in enumerate(columns):
                x[:, k, c] = f[col]
        metas.append(pd.DataFrame({"lat": np.full(len(t), la), "lon": np.full(len(t), lo), "time": dias[dd],
                                   "hora_pico": times[t]}))
        xs.append(x)
    return pd.concat(metas, ignore_index=True), np.concatenate(xs)


def points_of_cluster(path, cluster_id: int) -> list[tuple[float, float]]:
    """Células de um cluster num arquivo por pontos (a coluna `cluster` do próprio arquivo)."""
    ds = xr.open_dataset(path)
    m = ds.cluster.values == cluster_id
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(ds.lat.values[m], ds.lon.values[m])]


def cells_in_cluster(shp_path, cluster_id: int, lat, lon) -> list[tuple[float, float]]:
    """Células da grade com centro dentro do polígono do cluster (a regra das 74 células do cluster 3)."""
    import geopandas as gpd
    from shapely.geometry import Point

    s = gpd.read_file(shp_path).to_crs(4326)
    poly = s[s["cluster"].astype(int) == cluster_id].geometry.union_all()
    return [(float(a), float(b)) for a in lat for b in lon if poly.contains(Point(float(b), float(a)))]


def build_daily_peak(raw_dir, cells, years, columns=None, static=("anor",)) -> pd.DataFrame:
    """Todos os anos, com os estáticos juntados por célula e a estação do ano. `columns` restringe a saída."""
    st = None
    for name in static:
        s = static_features(raw_dir, cells, name)
        st = s if st is None else st.merge(s, on=["lat", "lon"])
    partes = []
    for y in years:
        d = daily_peak_year(raw_dir, cells, y)
        if st is not None:
            d = d.merge(st, on=["lat", "lon"], how="left")
        if columns is not None:
            d = d[["lat", "lon", "time", "hora_pico", *[c for c in columns if c in d.columns]]]
        partes.append(d)
    out = pd.concat(partes, ignore_index=True)
    from src.feature_study.diagnostics.inference import SEASON_OF_MONTH
    out["season"] = out["time"].dt.month.map(SEASON_OF_MONTH)
    return out.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
