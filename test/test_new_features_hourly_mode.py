"""Modo horário achatado de `src/data/new_features.py` — o ramo ADITIVO que o
estudo RAW usa, e a garantia de que ele não move o caminho diário.

O caminho diário é do qual TODAS as pipelines de produção (cluster_lazy, mlp,
lstm) dependem. Ele é o default e tem de continuar dando exatamente o mesmo
resultado, tenha o modo horário rodado antes ou não.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.new_features import (
    load_new_features_grid, load_new_features_hourly_flat,
    merge_new_features_hourly_flat,
)

# As 3 estações do cluster 9 do conftest, dentro do recorte NF_LATS/NF_LONS.
ESTACOES = pd.DataFrame({
    "estacao": ["A504", "A509", "A515"],
    "latitude": [-21.4500, -22.8617, -21.5664],
    "longitude": [-45.9500, -46.0433, -45.4042],
})


def test_daily_path_is_unchanged_after_the_hourly_mode_runs(synthetic_nf_raw_dir):
    """A regressão que importa: o modo horário compartilha `_RegionGrid`,
    `_complete_days` e o diretório de cache com o diário. Se ele corromper
    qualquer um dos três, toda a produção sai errada sem nenhum erro."""
    antes = load_new_features_grid(synthetic_nf_raw_dir)
    assert antes is not None

    load_new_features_hourly_flat(synthetic_nf_raw_dir, ESTACOES)

    depois = load_new_features_grid(synthetic_nf_raw_dir)
    assert sorted(depois.data_vars) == sorted(antes.data_vars)
    xr.testing.assert_identical(depois, antes)


def test_hourly_mode_returns_twenty_four_columns_per_variable_per_day(synthetic_nf_raw_dir):
    df = load_new_features_hourly_flat(synthetic_nf_raw_dir, ESTACOES)
    assert df is not None

    variaveis = sorted({c.rsplit("_h", 1)[0] for c in df.columns if c.startswith("hfn_")})
    for v in variaveis:
        horas = sorted(c for c in df.columns if c.startswith(f"{v}_h"))
        assert len(horas) == 24, v
        assert horas[0].endswith("_h00") and horas[-1].endswith("_h23")

    assert set(df["estacao"]) == set(ESTACOES["estacao"])
    assert (df.groupby("estacao")["time"].nunique() > 300).all()   # dias, não horas


def test_wind_magnitude_pairs_are_computed_hourly_before_flattening(synthetic_nf_raw_dir):
    """`sqrt(u²+v²)` hora a hora ≠ `sqrt(u²+v²)` de médias diárias — a
    desigualdade de Jensen garante que o segundo subestima. Como o alvo do
    projeto são EXTREMOS de vento, agregar antes de compor a magnitude seria um
    erro que não apareceria em lugar nenhum do resultado final.
    """
    df = load_new_features_hourly_flat(synthetic_nf_raw_dir, ESTACOES).set_index(["time", "estacao"])
    assert any(c.startswith("hfn_ws10_h") for c in df.columns)
    # os componentes crus NÃO entram como feature: só a magnitude
    assert not any(c.startswith(("hfn_u10_h", "hfn_v10_h")) for c in df.columns)

    ws = df[[f"hfn_ws10_h{h:02d}" for h in range(24)]].to_numpy()
    assert np.nanmin(ws) >= 0.0                                   # magnitude nunca é negativa
    # perfil intradiário preservado: as 24 horas variam dentro do dia
    assert np.nanstd(ws, axis=1).mean() > 0


def test_variable_with_incomplete_year_coverage_is_dropped_whole_without_imputation(
    synthetic_nf_raw_dir, capsys,
):
    """`u100` só existe no último ano da fixture. A disciplina do projeto é
    descartar a variável inteira, com aviso — nunca preencher os anos que
    faltam."""
    df = load_new_features_hourly_flat(synthetic_nf_raw_dir, ESTACOES)
    assert not any("u100" in c or "ws100" in c for c in df.columns)
    assert "descartada" in capsys.readouterr().out


def test_merge_takes_station_coordinates_from_the_frame_itself(synthetic_nf_raw_dir):
    """Nenhuma lista de estações duplicada: as coordenadas vêm do próprio
    frame, então acrescentar uma estação ao estudo não exige tocar aqui."""
    dias = pd.date_range("2018-03-01", periods=3, freq="D")
    df = pd.DataFrame({
        "estacao": np.repeat(ESTACOES["estacao"].to_numpy(), len(dias)),
        "latitude": np.repeat(ESTACOES["latitude"].to_numpy(), len(dias)),
        "longitude": np.repeat(ESTACOES["longitude"].to_numpy(), len(dias)),
        "time": np.tile(dias, len(ESTACOES)),
    })
    out = merge_new_features_hourly_flat(df, synthetic_nf_raw_dir)

    assert len(out) == len(df)                                    # left join: não duplica linha
    assert any(c.startswith("hfn_") for c in out.columns)
    assert out[[c for c in out.columns if c.startswith("hfn_")]].notna().all().all()


def test_no_regions_returns_none_and_merge_is_a_noop(tmp_path):
    assert load_new_features_hourly_flat(tmp_path, ESTACOES) is None
    df = pd.DataFrame({"estacao": ["A504"], "latitude": [-21.45], "longitude": [-45.95],
                       "time": [pd.Timestamp("2018-03-01")]})
    pd.testing.assert_frame_equal(merge_new_features_hourly_flat(df, tmp_path), df)


@pytest.mark.parametrize("metodo", ["nearest", "bilinear"])
def test_both_interpolation_methods_produce_the_same_shape(synthetic_nf_raw_dir, metodo):
    """O método grade→estação é o mesmo do ERA5-Basin; trocar de método muda os
    valores, nunca o formato."""
    df = load_new_features_hourly_flat(synthetic_nf_raw_dir, ESTACOES, metodo)
    assert len([c for c in df.columns if c.startswith("hfn_")]) % 24 == 0
