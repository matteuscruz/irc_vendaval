"""Achatamento horário RAW (`src/feature_study/hourly_flat.py`): as 24 horas do
dia viram 24 colunas, sem nenhuma redução.

Sintético e pequeno — cada teste sabe a resposta certa de antemão, sem tocar no
arquivo real de 232 MB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.feature_study.hourly_flat import (
    BASE_HOURLY_VARS, base_variable_of, flat_name, hourly_flat_columns,
    load_hourly_flat, merge_hourly_flat, purge_days,
)

VARS_TESTE = ("ws_h", "msl_h", "tp_roll72h")


def _write_hourly_nc(path, n_days=3, estacoes=("A801", "B807")):
    """`ws_h[dia, hora, estacao] = hora` — assim a hora fica legível no valor e
    qualquer troca de eixo no `reshape` aparece na hora."""
    times = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    n = len(times)
    horas = np.tile(np.arange(24, dtype="float32"), n_days)
    ws_h = np.repeat(horas[:, None], len(estacoes), axis=1)
    msl_h = ws_h + 1000.0
    tp_roll72h = ws_h / 10.0

    ds = xr.Dataset(
        {
            "ws_h": (("time", "estacao"), ws_h),
            "msl_h": (("time", "estacao"), msl_h),
            "tp_roll72h": (("time", "estacao"), tp_roll72h),
            # direção em graus: existe no arquivo real e tem de ser ignorada
            "wd_h": (("time", "estacao"), np.full((n, len(estacoes)), 180.0, dtype="float32")),
        },
        coords={"time": times, "estacao": list(estacoes)},
    )
    ds.to_netcdf(path)
    return ds


def test_build_hourly_flat_keeps_all_24_hours_as_separate_columns_in_order(tmp_path):
    """O ponto do módulo: nada é colapsado. `mean`/`max` apagariam o perfil
    intradiário, que é exatamente o sinal que o estudo quer entregar ao modelo.
    """
    _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=3, estacoes=("A801", "B807"))
    df = load_hourly_flat(tmp_path, variables=VARS_TESTE)

    assert len(df) == 3 * 2                                    # uma linha por (dia, estação)
    assert sorted(df.columns) == sorted(["time", "estacao"] + hourly_flat_columns(VARS_TESTE))
    # ws_h[hora] == hora, em todas as linhas: confirma o alinhamento dia×hora×estação
    for h in (0, 7, 13, 23):
        assert (df[f"hf_ws_h{h:02d}"] == h).all()


def test_column_order_preserves_the_intraday_sequence(tmp_path):
    """A sequência intradiária vive na ORDEM das colunas — é o que dá ao modelo
    a chance de ver "subiu à tarde" em vez de 24 números soltos."""
    cols = hourly_flat_columns(("ws_h", "msl_h"))
    assert cols[:3] == ["hf_ws_h00", "hf_ws_h01", "hf_ws_h02"]
    assert cols[24] == "hf_msl_h00"
    assert len(cols) == 2 * 24


def test_day_with_any_missing_hour_is_dropped_entirely_without_imputation(tmp_path):
    """Sem imputação: falta 1 das 24 horas e o dia inteiro daquela variável sai.
    Zerar só a hora faltante daria ao modelo um perfil fisicamente impossível.
    """
    path = tmp_path / "test_cluster_3_hourly.nc"
    ds = _write_hourly_nc(path, n_days=2, estacoes=("A801",))
    ds["msl_h"][13, 0] = np.nan          # 1 hora faltando no dia 0
    ds.to_netcdf(path)

    df = load_hourly_flat(tmp_path, variables=VARS_TESTE).set_index(["time", "estacao"])
    dia0 = df.loc[(pd.Timestamp("2020-01-01"), "A801")]
    dia1 = df.loc[(pd.Timestamp("2020-01-02"), "A801")]

    assert dia0[[f"hf_msl_h{h:02d}" for h in range(24)]].isna().all()   # as 24, não só a hora 13
    assert dia0[[f"hf_ws_h{h:02d}" for h in range(24)]].notna().all()   # outra variável, intacta
    assert dia1[[f"hf_msl_h{h:02d}" for h in range(24)]].notna().all()  # outro dia, intacto


def test_irregular_hourly_series_raises_instead_of_silently_reshaping(tmp_path):
    """O achatamento é um `reshape` em blocos de 24: um furo na série
    desalinharia dia e hora em silêncio, e o erro só apareceria no resultado."""
    path = tmp_path / "test_cluster_3_hourly.nc"
    ds = _write_hourly_nc(path, n_days=2, estacoes=("A801",))
    ds = ds.isel(time=[i for i in range(len(ds.time)) if i != 5])
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="múltiplo de 24h|horária regular"):
        load_hourly_flat(tmp_path, variables=VARS_TESTE)


def test_hourly_flat_columns_excludes_wind_direction_degrees():
    """`wd_h` existe no arquivo mas não pode entrar: graus não interpolam em
    0°/360°, e 359 vs 1 viraria uma diferença de 358. `sin_dir_h`/`cos_dir_h`
    cobrem a direção de forma utilizável."""
    assert "wd_h" not in BASE_HOURLY_VARS
    assert not any("wd" in c for c in hourly_flat_columns())


def test_base_variable_of_inverts_flat_name_for_both_prefixes():
    """A purga e o roteamento dos grupos temáticos dependem de recuperar a
    variável a partir do nome da coluna."""
    assert base_variable_of(flat_name("ws_h", 7)) == "ws_h"
    assert base_variable_of(flat_name("tp_roll72h", 0)) == "tp_roll72h"
    assert base_variable_of(flat_name("blh", 13, "hfn_")) == "blh"
    assert base_variable_of("latitude") == "latitude"      # não achatada: passa direto


def test_purge_days_is_derived_from_the_longest_rolling_window_not_hardcoded():
    """O gap do split tem de acompanhar o conjunto de features. Fixá-lo em 3
    deixaria um vazamento silencioso no dia em que uma feature com lag maior
    entrasse."""
    assert purge_days(hourly_flat_columns(("ws_h", "msl_h"))) == 0
    assert purge_days(hourly_flat_columns(("ws_h", "tp_roll24h"))) == 1
    assert purge_days(hourly_flat_columns(("ws_h", "tp_roll48h"))) == 2
    assert purge_days(hourly_flat_columns(BASE_HOURLY_VARS)) == 3      # tp_roll72h manda


def test_missing_file_returns_none_and_merge_leaves_the_frame_unchanged(tmp_path):
    assert load_hourly_flat(tmp_path) is None
    df = pd.DataFrame({"estacao": ["A801"], "time": [pd.Timestamp("2020-01-01")], "x": [1.0]})
    pd.testing.assert_frame_equal(merge_hourly_flat(df, tmp_path), df)


def test_merge_matches_by_station_and_day_and_leaves_other_stations_as_nan(tmp_path):
    _write_hourly_nc(tmp_path / "test_cluster_3_hourly.nc", n_days=2, estacoes=("A801",))
    df = pd.DataFrame({
        "estacao": ["A801", "A801", "OUTRA"],
        "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01")],
        "y": [1.0, 2.0, 3.0],
    })
    out = merge_hourly_flat(df, tmp_path, variables=VARS_TESTE)

    assert len(out) == len(df)                              # left join: não duplica nem perde linha
    assert out.loc[out.estacao == "OUTRA", "hf_ws_h00"].isna().all()
    assert out.loc[out.estacao == "A801", "hf_ws_h00"].notna().all()


# ── `prepare` de ponta a ponta no modo `hourly_raw` ─────────────────────────

def _raw_dir_completo(tmp_path):
    """Diário (alvo INMET + ERA5) + arquivo horário por estação + features
    novas em grade — as três fontes que o modo RAW junta."""
    from test.conftest import (
        _synthetic_daily_dates, build_synthetic_hourly_file,
        build_synthetic_new_features, build_synthetic_raw_dir,
    )

    datas = _synthetic_daily_dates()
    raw = build_synthetic_raw_dir(tmp_path / "raw", datas)
    build_synthetic_hourly_file(raw)
    build_synthetic_new_features(raw, range(datas[0].year, datas[-1].year + 1))
    return raw


def test_prepare_in_hourly_raw_mode_builds_a_population_with_no_daily_features(
    tmp_path, synthetic_shp_dir,
):
    """O contrato do estudo RAW: nada de diário em lugar nenhum. A base é o
    perfil horário cru (`hf_`), as novas são igualmente horárias (`hfn_`), e as
    estáticas entram como estão — são geografia, não agregados."""
    from src.feature_study.prepare import prepare

    raw = _raw_dir_completo(tmp_path)
    meta = prepare(str(raw), str(synthetic_shp_dir), tmp_path / "estudo",
                   cluster_id=9, arm_sets=("anchors", "groups", "controls", "hourly_raw"))

    assert all(f.startswith("hf_") or f in ("latitude", "longitude") for f in meta["base_features"])
    assert all(f.startswith(("hfn_", "nf_")) for f in meta["new_features"])
    # nenhuma feature diária da base antiga sobreviveu
    assert not any(f in meta["base_features"] for f in ("ws_max", "ws_mean", "msl_mean"))
    assert meta["n_arms"] > 2 and meta["n_test_rows"] > 0


def test_prepare_records_the_purge_gap_it_derived_and_applied(tmp_path, synthetic_shp_dir):
    """O gap vai para o `meta.json` porque é uma decisão do desenho, não um
    detalhe: quem lê o estudo depois precisa saber quantos dias de fronteira
    foram descartados, e por qual feature."""
    from src.feature_study.prepare import prepare

    raw = _raw_dir_completo(tmp_path)
    meta = prepare(str(raw), str(synthetic_shp_dir), tmp_path / "estudo",
                   cluster_id=9, arm_sets=("anchors", "groups", "controls", "hourly_raw"))

    assert meta["purge_days"] == 3                    # tp_roll72h olha 72h para trás
    assert meta["rows_purged"] > 0


def test_prepare_keeps_station_coordinates_for_the_spatial_map(tmp_path, synthetic_shp_dir):
    """Sem lat/lon no `test.parquet` não há mapa espacial nenhum, e a perda
    seria silenciosa — o mlp v4 já removeu lat/lon do conjunto de features no
    caminho dele."""
    from src.feature_study.prepare import prepare

    raw = _raw_dir_completo(tmp_path)
    prepare(str(raw), str(synthetic_shp_dir), tmp_path / "estudo",
            cluster_id=9, arm_sets=("anchors", "groups", "controls", "hourly_raw"))

    teste = pd.read_parquet(tmp_path / "estudo" / "data" / "test.parquet")
    assert {"latitude", "longitude", "estacao", "time"} <= set(teste.columns)
    assert teste[["latitude", "longitude"]].notna().all().all()
    assert set(teste["month"]) <= {1, 4, 7, 10}       # split por blocos de mês, intacto


def test_prepare_keeps_the_era5_proxy_as_a_reference_column_not_a_feature(
    tmp_path, synthetic_shp_dir,
):
    """`wind_mag_max` é o proxy de rajada do ERA5 — é contra ele que o mapa
    espacial compara o modelo. No estudo diário ele vinha de graça, por ser uma
    das 40 features base; no modo RAW a base virou `hf_*` e ele sumiu do frame.

    A perda era SILENCIOSA: nada falhava, só o `era5_proxy` das predições saía
    todo NaN e o mapa de correlação ficava vazio. Ele é preservado como coluna
    de REFERÊNCIA — está no parquet, não está em arm nenhum.
    """
    import json

    from src.feature_study.prepare import prepare
    from src.pipelines.common import ERA5_GUST_PROXY

    raw = _raw_dir_completo(tmp_path)
    prepare(str(raw), str(synthetic_shp_dir), tmp_path / "estudo",
            cluster_id=9, arm_sets=("anchors", "groups", "controls", "hourly_raw"))
    data = tmp_path / "estudo" / "data"

    teste = pd.read_parquet(data / "test.parquet")
    assert ERA5_GUST_PROXY in teste.columns
    assert teste[ERA5_GUST_PROXY].notna().all()

    # e não vazou para dentro de nenhum arm — é referência, não entrada
    arms = json.loads((data / "arms.json").read_text())
    assert not any(ERA5_GUST_PROXY in a["features"] for a in arms)
