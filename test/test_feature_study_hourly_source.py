"""Segunda fonte de features novas — arquivo horário por estação
(`src/feature_study/hourly_source.py`). Sintético, pequeno, sem tocar no
arquivo real: cada teste sabe a resposta certa de antemão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.feature_study.hourly_source import (
    HOURLY_STATS, load_hourly_daily, merge_hourly_features,
)


def _write_hourly_nc(path, n_days=3, estacoes=("A801", "B807"), seed=0):
    """2 dias "bons" (24h completas) + o resto com alguma variação, por
    estação. `ws_h` tem um salto conhecido no dia 0 para testar `jump`."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    n = len(times)
    ws_h = rng.normal(5, 1, (n, len(estacoes))).astype("float32")
    ws_h[11, 0] = ws_h[10, 0] + 15.0   # salto grande no dia 0, estação 0 (hora 11)
    msl_h = rng.normal(1013, 2, (n, len(estacoes))).astype("float32")
    t2m_h = rng.normal(20, 3, (n, len(estacoes))).astype("float32")
    rh_h = rng.normal(70, 5, (n, len(estacoes))).astype("float32")
    td_dep_h = rng.normal(4, 1, (n, len(estacoes))).astype("float32")
    tp_h = np.abs(rng.normal(0.2, 0.1, (n, len(estacoes)))).astype("float32")
    tp_roll24h = tp_h.copy()
    sin_dir_h = rng.uniform(-1, 1, (n, len(estacoes))).astype("float32")
    cos_dir_h = rng.uniform(-1, 1, (n, len(estacoes))).astype("float32")

    ds = xr.Dataset(
        {
            "ws_h": (("time", "estacao"), ws_h), "msl_h": (("time", "estacao"), msl_h),
            "t2m_h": (("time", "estacao"), t2m_h), "rh_h": (("time", "estacao"), rh_h),
            "td_dep_h": (("time", "estacao"), td_dep_h), "tp_h": (("time", "estacao"), tp_h),
            "tp_roll24h": (("time", "estacao"), tp_roll24h),
            "sin_dir_h": (("time", "estacao"), sin_dir_h), "cos_dir_h": (("time", "estacao"), cos_dir_h),
        },
        coords={"time": times, "estacao": list(estacoes)},
    )
    ds.to_netcdf(path)
    return ds


def test_load_hourly_daily_produces_one_row_per_station_per_day(tmp_path):
    _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=3, estacoes=("A801", "B807"))
    df = load_hourly_daily(tmp_path)

    assert len(df) == 3 * 2
    assert set(df["estacao"]) == {"A801", "B807"}
    assert sorted(df.columns) == sorted(
        ["time", "estacao"] + [f"nf_h_{v.removesuffix('_h')}_{s}" for v, ss in HOURLY_STATS.items() for s in ss]
    )


def test_jump_catches_the_hourly_spike_that_mean_and_max_alone_would_dilute(tmp_path):
    """É o caso-chave: a mesma hora de salto tem que aparecer em `jump` mesmo
    quando `mean`/`std` do dia não destacam nada de especial."""
    _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=3, estacoes=("A801", "B807"))
    df = load_hourly_daily(tmp_path).set_index(["time", "estacao"])

    dia0_a801 = df.loc[(pd.Timestamp("2020-01-01"), "A801")]
    dia0_b807 = df.loc[(pd.Timestamp("2020-01-01"), "B807")]
    assert dia0_a801["nf_h_ws_jump"] > 10          # pegou o salto de +15
    assert dia0_b807["nf_h_ws_jump"] < 5            # sem salto, ruído normal


def test_range_is_max_minus_min_within_the_day(tmp_path):
    ds = _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=2, estacoes=("A801",))
    df = load_hourly_daily(tmp_path).set_index(["time", "estacao"])
    dia0 = ds["t2m_h"].isel(time=slice(0, 24), estacao=0).values
    row = df.loc[(pd.Timestamp("2020-01-01"), "A801")]
    assert row["nf_h_t2m_range"] == pytest.approx(dia0.max() - dia0.min(), abs=1e-4)


def test_a_day_missing_even_one_hour_is_dropped_whole_not_imputed(tmp_path):
    """Sem imputação: falta 1 das 24 horas ⇒ a estatística daquele dia some,
    não vira uma média sobre 23 horas disfarçada de dado completo."""
    path = tmp_path / "test_cluster_3_hourly.nc"
    ds = _write_hourly_nc(path, n_days=2, estacoes=("A801",))   # já escrito e fechado
    ds["ws_h"][5, 0] = np.nan   # 1 hora faltando no dia 0
    ds.to_netcdf(path)

    df = load_hourly_daily(tmp_path).set_index(["time", "estacao"])
    assert np.isnan(df.loc[(pd.Timestamp("2020-01-01"), "A801"), "nf_h_ws_mean"])
    assert not np.isnan(df.loc[(pd.Timestamp("2020-01-02"), "A801"), "nf_h_ws_mean"])  # dia intacto, intocado


def test_irregular_hourly_series_is_rejected_not_silently_misaligned(tmp_path):
    """A forma vetorizada (reshape em blocos de 24) exige série 1h regular
    começando à meia-noite — furo ou passo diferente tem de falhar alto, não
    desalinhar dia/hora em silêncio."""
    path = tmp_path / "test_cluster_3_hourly.nc"
    ds = _write_hourly_nc(path, n_days=2, estacoes=("A801",))
    ds = ds.isel(time=[i for i in range(len(ds.time)) if i != 5])   # remove 1 hora: quebra o múltiplo de 24
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="múltiplo de 24h|horária regular"):
        load_hourly_daily(tmp_path)


def test_missing_file_returns_none_and_merge_leaves_the_frame_unchanged(tmp_path):
    assert load_hourly_daily(tmp_path) is None
    df = pd.DataFrame({"estacao": ["A801"], "time": [pd.Timestamp("2020-01-01")], "x": [1.0]})
    merged = merge_hourly_features(df, tmp_path)
    pd.testing.assert_frame_equal(merged, df)


def test_merge_matches_by_station_and_day_and_leaves_other_stations_as_nan(tmp_path):
    _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=2, estacoes=("A801",))
    df = pd.DataFrame({
        "estacao": ["A801", "A801", "OUTRA"],
        "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01")],
        "y": [1.0, 2.0, 3.0],
    })
    out = merge_hourly_features(df, tmp_path)

    assert len(out) == len(df)                                  # left join: não duplica nem perde linha
    assert out.loc[out.estacao == "OUTRA", "nf_h_ws_mean"].isna().all()
    assert out.loc[out.estacao == "A801", "nf_h_ws_mean"].notna().all()
