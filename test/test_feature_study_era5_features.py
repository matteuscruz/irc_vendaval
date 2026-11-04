"""Colunas selecionadas a partir do ERA5 bruto (`data/era5_features.py`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.data import era5_features as ef

LAT = np.array([-30.0, -30.25, -30.5])
LON = np.array([-51.5, -51.25, -51.0])


def _grava(raw, var, year, valores):
    t = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    da = xr.DataArray(valores(len(t)).astype("float32"), dims=("valid_time", "latitude", "longitude"),
                      coords={"valid_time": t, "latitude": LAT, "longitude": LON}, name=var)
    (raw / "sl").mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(raw / "sl" / f"{var}_{year}.nc")


def _raw(tmp_path):
    raw = tmp_path / "c"
    r = np.random.default_rng(0)
    for y in (2011, 2012):
        def gust(n):
            base = r.uniform(5, 8, (n, 3, 3))
            h = np.arange(n) % 24
            base[h == 15] += 10                                   # pico às 15 UTC em toda a grade
            return base
        _grava(raw, "gust10fg", y, gust)
        _grava(raw, "u10", y, lambda n: np.full((n, 3, 3), 3.0))
        _grava(raw, "v10", y, lambda n: np.full((n, 3, 3), 4.0))
        _grava(raw, "cape", y, lambda n: r.uniform(0, 100, (n, 3, 3)))
    (raw / "static").mkdir()
    xr.DataArray(np.arange(9, dtype="float32").reshape(3, 3), dims=("latitude", "longitude"),
                 coords={"latitude": LAT, "longitude": LON}, name="anor").to_dataset().to_netcdf(raw / "static/anor.nc")
    return raw


def test_peak_hour_lags_and_neighbourhood_follow_the_rules(tmp_path):
    raw = _raw(tmp_path)
    d = ef.build_daily_peak(raw, [(-30.25, -51.25)], [2012])
    assert len(d) == 366 and (d.hora_pico.dt.hour == 15).all()
    assert np.allclose(d.w10, 5.0) and np.allclose(d.w10_r75_median, 5.0)
    g = ef.load_year(raw, "gust10fg", 2012).sel(latitude=-30.25, longitude=-51.25).to_series()
    r0 = d.iloc[0]
    i = g.index.get_loc(r0.hora_pico)
    assert np.isclose(r0.gust10fg_lag1h, g.iloc[i - 1])
    assert np.isclose(r0.gust10fg_max_prev3h, g.iloc[i - 3:i].max())
    assert np.isclose(r0.gust10fg_std_prev3h, g.iloc[i - 3:i].std(ddof=1))
    assert np.isclose(r0.gust10fg_delta3h, g.iloc[i] - g.iloc[i - 3])
    assert np.isclose(r0.cape_c75_250, r0.cape_r75_median - r0.cape_r250_median)
    assert np.isclose(r0.cape_l75_250, r0.cape_r75_max - r0.cape_r250_median)
    assert r0.anor_ponto == 4.0 and r0.anor_r250_max == 8.0
    hs = (15 + (-51.25) / 15) % 24
    assert np.isclose(r0.hora_solar_sin, np.sin(2 * np.pi * hs / 24))


def test_first_year_without_previous_file_has_nan_lags_only_on_the_first_day(tmp_path):
    raw = _raw(tmp_path)
    (raw / "sl" / "gust10fg_2011.nc").unlink()
    d = ef.build_daily_peak(raw, [(-30.25, -51.25)], [2012])
    assert d.gust10fg_lag1h.isna().sum() == 0             # pico às 15h: as 3 horas anteriores são do mesmo dia


def test_grad_mslp_of_a_uniform_meridional_gradient():
    lat, lon = LAT, LON
    msl = np.repeat((101000 + (lat[:, None] - lat[0]) / 0.25 * 100 * np.ones((1, 3)))[None], 2, 0)  # 1 hPa por 0,25°
    g = ef.grad_mslp(msl, lat, lon)
    assert np.allclose(g, 1.0 / (0.25 * ef.KM_GRAU) * 100, rtol=1e-6)


def test_window_last_step_equals_the_peak_row_and_earlier_steps_are_earlier_hours(tmp_path):
    raw = _raw(tmp_path)
    cols = ["gust10fg", "gust10fg_lag1h", "w10_r75_median", "cape_c75_250", "hora_solar_sin"]
    linha = ef.daily_peak_year(raw, [(-30.25, -51.25)], 2012)
    meta, x = ef.peak_windows_year(raw, [(-30.25, -51.25)], 2012, cols, window=24)
    assert x.shape == (len(linha), 24, len(cols)) and (meta.hora_pico.values == linha.hora_pico.values).all()
    np.testing.assert_allclose(x[:, -1, :], linha[cols].to_numpy(), rtol=1e-6)
    g = ef.load_year(raw, "gust10fg", 2012, tail_hours=27).sel(latitude=-30.25, longitude=-51.25).to_series()
    r = meta.iloc[5]
    np.testing.assert_allclose(x[5, 0, 0], g.loc[r.hora_pico - pd.Timedelta(hours=23)], rtol=1e-6)
    np.testing.assert_allclose(x[5, 0, 1], g.loc[r.hora_pico - pd.Timedelta(hours=24)], rtol=1e-6)


def test_grid_of_reads_the_grid_from_sl_when_there_is_no_static(tmp_path):
    raw = _raw(tmp_path)
    (raw / "static" / "anor.nc").unlink()
    lat, lon = ef.grid_of(raw)
    assert not ef.has_static(raw) and list(lat) == list(LAT) and list(lon) == list(LON)


def test_arquivo_por_pontos_usa_so_vizinhos_disponiveis(tmp_path):
    t = pd.date_range("2010-01-01", "2010-01-02 23:00", freq="h")
    la, lo = np.array([-25.0, -25.0, -25.25, -30.0]), np.array([-50.0, -49.75, -50.0, -55.0])
    rng = np.random.default_rng(0)
    vs = {v: (("point", "time"), rng.random((4, len(t))).astype("float32") + 1) for v in ef.VARIAVEIS}
    p = tmp_path / "pontos.nc"
    xr.Dataset({**vs, "lat": ("point", la.astype("float32")), "lon": ("point", lo.astype("float32")),
                "cluster": ("point", np.array([5, 5, 5, 4], "int16"))}, coords={"time": t}).to_netcdf(p)
    assert ef.points_of_cluster(p, 5) == [(-25.0, -50.0), (-25.0, -49.75), (-25.25, -50.0)]
    d = ef.daily_peak_year(p, [(-25.0, -50.0)], 2010)
    assert len(d) == 2 and d.drop(columns=["time", "hora_pico"]).notna().all().all()
    g = vs["gust10fg"][1]
    hora = d.hora_pico.iloc[0].hour
    assert d.gust10fg.iloc[0] == pytest.approx(g[0, hora])
    assert d.gust_r75_max.iloc[0] == pytest.approx(g[:3, hora].max())    # o ponto distante não entra
