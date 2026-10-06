"""Estrutura nova por grupo (`src/feature_study/groups_source.py`) e o modo
`era5_groups` do `prepare` e dos arms.

Parquets sintéticos pequenos, com a forma dos reais: longos por (estação, hora)
para os grupos 1 a 3, uma linha por estação para o grupo 4. Cada teste sabe a
resposta certa de antemão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.data import groups_source as gs
from src.feature_study.core.arms import build_arms, extra_comparisons
from src.feature_study.data.purge import lookback_hours, purge_days

ESTACOES = ["A504", "A509", "A515"]          # as 3 do cluster 9 da fixture sintética
DIAS = pd.date_range("2016-01-01", "2019-12-31", freq="D")


def _escreve(raw_dir, *, hora_pico=lambda i: i % 24, nulo_em_cin=0.8, falta=None):
    """Grava os quatro parquets em `raw_dir/feature_study_cluster_3/`.

    `hora_pico(i)` é a hora (0–23) do máximo de `gust10fg` no dia `i`.
    `falta=(estacao, dia_idx, hora)` apaga uma hora de rajada, para o teste da
    regra de dia completo.
    """
    pasta = raw_dir / gs.SUBDIR
    pasta.mkdir(parents=True, exist_ok=True)
    horas = pd.date_range(DIAS[0], DIAS[-1] + pd.Timedelta(hours=23), freq="h")
    dia_idx = ((horas.normalize() - DIAS[0]).days).to_numpy()
    rng = np.random.default_rng(0)
    pico = np.array([hora_pico(i) for i in range(len(DIAS))])[dia_idx]

    g1, g2, g3 = [], [], []
    for k, e in enumerate(ESTACOES):
        gust = 5.0 + 10.0 * (horas.hour.to_numpy() == pico) + rng.uniform(0, 0.5, len(horas))
        if falta and falta[0] == e:
            gust[(dia_idx == falta[1]) & (horas.hour.to_numpy() == falta[2])] = np.nan
        t = horas.astype("datetime64[us]")
        g1.append(pd.DataFrame({
            gs.STATION_SRC: e, gs.TIME_SRC: t, "gust10fg": gust,
            "w10": horas.hour.to_numpy() / 24.0 + k,
            "hora_solar_sin": np.sin(2 * np.pi * horas.hour.to_numpy() / 24),
            "gust10fg_lag1h": pd.Series(gust).shift(1).to_numpy(),
            "gust10fg_max_prev3h": pd.Series(gust).rolling(3).max().shift(1).to_numpy(),
        }))
        cin = rng.normal(size=len(horas))
        cin[rng.uniform(size=len(horas)) < nulo_em_cin] = np.nan
        g2.append(pd.DataFrame({gs.STATION_SRC: e, gs.TIME_SRC: t, "cape": gust * 2.0,
                                "cin": cin, "mcpr": rng.uniform(size=len(horas))}))
        g3.append(pd.DataFrame({gs.STATION_SRC: e, gs.TIME_SRC: t,
                                "mslp_hpa": 1013 + rng.normal(size=len(horas)),
                                "w850": rng.uniform(size=len(horas))}))
    for nome, frames in (("grupo1", g1), ("grupo2", g2), ("grupo3", g3)):
        pd.concat(frames, ignore_index=True).to_parquet(
            pasta / gs.GROUP_FILES[nome], index=False, row_group_size=40_000)

    pd.DataFrame({
        gs.STATION_SRC: ESTACOES,
        "latitude": [-21.45, -22.86, -21.57], "longitude": [-45.95, -46.04, -45.40],
        "grid_lat": [-21.5, -23.0, -21.5], "grid_lon": [-46.0, -46.0, -45.5],
        "dist_celula_km": [3.1, 4.2, 5.3],
        "z_surface_ponto": [900.0, 1100.0, 950.0], "z_surface_r75_mean": [910.0, 1090.0, 960.0],
        "z_surface_r75_n": np.array([9, 9, 9], dtype="int64"),
        "sdor_ponto": [30.0, 60.0, 45.0],
    }).to_parquet(pasta / gs.GROUP_FILES["grupo4"], index=False)
    return raw_dir


@pytest.fixture
def raw(tmp_path):
    return _escreve(tmp_path / "raw")


# ── Escolha da hora do pico ─────────────────────────────────────────────────

def test_the_row_of_each_day_is_the_hour_of_the_peak_era5_gust(raw):
    """O desenho é um ponto por dia na hora do pico de `gust10fg`. A hora vem do
    ERA5, nunca do INMET — o que evita vazar o alvo — e nessa hora a rajada é,
    por construção, o máximo diário: a linha de base honesta."""
    daily, _ = gs.build_daily_table(raw, ESTACOES)
    daily = daily.sort_values([gs.STATION, gs.TIME])

    esperado = ((daily[gs.TIME] - DIAS[0]).dt.days % 24).to_numpy()
    assert (daily[gs.PEAK_HOUR_COLUMN].to_numpy() == esperado).all()
    assert (daily[gs.REFERENCE_COLUMN] == daily["gust10fg"]).all()
    assert (daily["gust10fg"] > 14.5).all()          # 5 + 10: o pico, não o fundo


def test_features_of_other_groups_come_from_the_same_peak_hour(raw):
    """`cape` foi montada como 2 × `gust10fg` da MESMA hora. Se o grupo 2 fosse
    lido em outra hora, a relação se desfaria."""
    daily, _ = gs.build_daily_table(raw, ESTACOES)
    assert np.allclose(daily["cape"], 2.0 * daily["gust10fg"], atol=1e-4)


def test_a_day_missing_any_gust_hour_is_dropped_whole_without_imputation(tmp_path):
    """Sem a hora, o pico não é identificável, e escolher entre as 23 restantes
    seria imputar justamente a informação que falta."""
    raw = _escreve(tmp_path / "raw", falta=("A509", 100, 7))
    daily, _ = gs.build_daily_table(raw, ESTACOES)
    dia = DIAS[100]

    assert not ((daily[gs.STATION] == "A509") & (daily[gs.TIME] == dia)).any()
    assert ((daily[gs.STATION] == "A504") & (daily[gs.TIME] == dia)).any()   # outra estação, intacta
    assert len(daily) == len(DIAS) * 3 - 1


# ── Colunas e grupos ────────────────────────────────────────────────────────

def test_a_column_with_too_many_nulls_is_dropped_and_reported_not_filled(raw):
    """É o caso real de `cin` (75 % nulo nas 9 estações). O projeto não imputa:
    a coluna sai, com aviso, e fica registrada."""
    daily, info = gs.build_daily_table(raw, ESTACOES)
    assert "cin" not in daily.columns and "cin" not in info["group_of"]
    assert info["dropped_null_columns"]["cin"] > 0.5


def test_group_of_records_the_source_of_every_feature_column(raw):
    daily, info = gs.build_daily_table(raw, ESTACOES)
    g = info["group_of"]
    assert g["gust10fg"] == "grupo1" and g["cape"] == "grupo2" and g["mslp_hpa"] == "grupo3"
    assert g["z_surface_ponto"] == "grupo4"
    # o spec põe lat/lon no grupo 1
    assert g["latitude"] == "grupo1" and g["longitude"] == "grupo1"
    # toda coluna registrada existe na tabela, e vice-versa (fora metadados)
    meta = {gs.STATION, gs.TIME, gs.PEAK_HOUR_COLUMN, gs.REFERENCE_COLUMN}
    assert set(g) == set(daily.columns) - meta


def test_static_identifiers_and_cell_counts_are_not_features(raw):
    """`*_n` (células no raio) e `dist_celula_km` só identificariam a estação, e
    `grid_lat/lon` são a posição da célula, não relevo."""
    cols = gs.static_columns(raw)
    assert "z_surface_r75_n" not in cols and "dist_celula_km" not in cols
    assert "grid_lat" not in cols and "latitude" not in cols
    assert {"z_surface_ponto", "z_surface_r75_mean", "sdor_ponto"} <= set(cols)


def test_era5_gust_reference_is_kept_out_of_every_arm_by_construction(raw):
    """`era5_gust_max` é a referência contra a qual se mede; somá-la a um arm
    seria dar ao modelo a própria resposta do ERA5."""
    _, info = gs.build_daily_table(raw, ESTACOES)
    assert gs.REFERENCE_COLUMN not in info["group_of"]
    assert gs.PEAK_HOUR_COLUMN not in info["group_of"]


# ── Cache e memória ─────────────────────────────────────────────────────────

def test_cache_is_reused_and_invalidated_when_stations_change(raw, tmp_path, capsys):
    cache = tmp_path / "cache"
    a, _ = gs.build_daily_table(raw, ESTACOES, cache_dir=cache)
    capsys.readouterr()
    b, _ = gs.build_daily_table(raw, ESTACOES, cache_dir=cache)
    assert "usando cache" in capsys.readouterr().out
    pd.testing.assert_frame_equal(a, b)

    c, _ = gs.build_daily_table(raw, ESTACOES[:2], cache_dir=cache)
    assert "refazendo" in capsys.readouterr().out
    assert set(c[gs.STATION]) == set(ESTACOES[:2])


def test_reading_in_small_batches_gives_the_same_table(raw, monkeypatch):
    """O limite de memória da máquina local exige lotes pequenos; isso não pode
    mudar um único valor."""
    grande, _ = gs.build_daily_table(raw, ESTACOES)
    monkeypatch.setattr(gs, "BATCH_ROWS", 997)
    pequeno, _ = gs.build_daily_table(raw, ESTACOES)
    pd.testing.assert_frame_equal(
        grande.sort_values([gs.STATION, gs.TIME]).reset_index(drop=True),
        pequeno.sort_values([gs.STATION, gs.TIME]).reset_index(drop=True))


# ── Alvo e split ────────────────────────────────────────────────────────────

def test_target_axis_stops_at_2024_so_the_split_matches_the_previous_study(tmp_path):
    """Com os dias de 2025 o sorteio dos blocos de validação muda (medido: 15 dos
    32 trocam). Truncar o eixo em 2024 é o que torna este estudo comparável ao
    anterior."""
    import xarray as xr

    dias = pd.date_range("2022-01-01", "2025-06-30", freq="D")
    xr.Dataset({gs.TARGET_VAR: (("time", "estacao"), np.full((len(dias), 2), 9.0))},
               coords={"time": dias, "estacao": ["A504", "A509"]}).to_netcdf(tmp_path / gs.TARGET_FILE)
    alvo, eixo = gs.load_target(tmp_path)
    assert eixo.max() == pd.Timestamp("2024-12-31") and alvo[gs.TIME].max() <= pd.Timestamp("2024-12-31")


# ── Arms do modo novo ───────────────────────────────────────────────────────

def _arms(raw):
    _, info = gs.build_daily_table(raw, ESTACOES)
    g = info["group_of"]
    base = sorted(c for c, x in g.items() if x == "grupo1")
    estaticas = sorted(c for c, x in g.items() if x == "grupo4")
    new = sorted(c for c, x in g.items() if x in ("grupo2", "grupo3")) + estaticas
    return build_arms(base, new, estaticas, ("anchors", "groups", "controls", "era5_groups"),
                      group_of=g), g


def test_arms_measure_whole_groups_against_a_base_made_of_group_one(raw):
    arms, g = _arms(raw)
    por_nome = {a.name: a for a in arms}

    assert set(por_nome) == {
        "base", "full", "ctrl__noise", "ctrl__perm_static",
        "add_grp__grupo2", "add_grp__grupo3", "add_grp__grupo4",
        "drop_grp__grupo1", "drop_grp__grupo2", "drop_grp__grupo3", "drop_grp__grupo4",
    }
    assert {g[c] for c in por_nome["base"].features} == {"grupo1"}
    assert {g[c] for c in por_nome["add_grp__grupo2"].features} == {"grupo1", "grupo2"}
    assert not any(g[c] == "grupo2" for c in por_nome["drop_grp__grupo2"].features)


def test_group_one_is_only_ever_removed_never_added_to_its_own_base(raw):
    """O grupo 1 já é a base: somá-lo à base daria um arm idêntico a ela."""
    arms, _ = _arms(raw)
    nomes = {a.name for a in arms}
    assert "drop_grp__grupo1" in nomes and "add_grp__grupo1" not in nomes


def test_real_versus_permuted_statics_is_compared_under_the_new_group_name(raw):
    """O diagnóstico existia com o grupo chamado `static`; no modo novo ele se
    chama `grupo4`, e sem o ajuste o controle deixaria de ser comparado."""
    arms, _ = _arms(raw)
    nomes = {c[0] for c in extra_comparisons(arms)}
    assert {"full_vs_base", "real_vs_perm_static"} <= nomes


def test_a_column_without_a_source_group_raises_instead_of_escaping_every_drop(raw):
    _, info = gs.build_daily_table(raw, ESTACOES)
    g = dict(info["group_of"])
    base = sorted(c for c, x in g.items() if x == "grupo1")
    new = ["cape", "coluna_orfa"]
    with pytest.raises(ValueError, match="sem grupo de origem"):
        build_arms(base, new, [], ("anchors", "groups", "era5_groups"), group_of=g)




# ── Purga pelas defasagens do spec ──────────────────────────────────────────

def test_lag_suffixes_set_the_lookback_and_three_hours_means_one_purged_day():
    """A hora do pico pode ser 00 UTC (10 % dos dias): `max_prev3h` então olha
    para 21–23 h do dia ANTERIOR, que pode ser de outro split. Três horas de
    alcance pedem 1 dia de purga."""
    assert lookback_hours("gust10fg_lag1h") == 1
    assert lookback_hours("w10_max_prev3h") == 3 and lookback_hours("blh_delta3h") == 3
    assert lookback_hours("mslp_tend_3h") == 3
    assert lookback_hours("gust_r75_median") == 0 and lookback_hours("cape") == 0
    assert purge_days(["cape", "gust10fg_max_prev3h"]) == 1
    assert purge_days(["cape", "w10"]) == 0


# ── prepare de ponta a ponta ────────────────────────────────────────────────

def test_prepare_builds_the_study_without_the_old_era5_files(tmp_path):
    """O ponto do modo: roda só com os quatro parquets e o INMET do alvo, sem
    `ERA5_Stratified`, `ERA5_Features_Basin` ou `era5_merged_cache` — que saíram
    de `dataset/raw`."""
    import json

    from src.feature_study.core.prepare import prepare
    from test.conftest import build_synthetic_raw_dir

    raw = build_synthetic_raw_dir(tmp_path / "raw", DIAS)
    for antigo in ("ERA5_Stratified.nc", "ERA5_Features_Basin_2000_2026.nc"):
        (raw / antigo).unlink(missing_ok=True)
    _escreve(raw)

    meta = prepare(str(raw), tmp_path / "estudo", cluster_id=9,
                   arm_sets=("anchors", "groups", "controls", "era5_groups"))
    data = tmp_path / "estudo" / "data"

    assert meta["n_arms"] == 11 and meta["purge_days"] == 1
    assert meta["row_design"] == "peak_hour"
    teste = pd.read_parquet(data / "test.parquet")
    assert set(teste["month"]) <= {1, 4, 7, 10} and len(teste) > 0
    assert gs.REFERENCE_COLUMN in teste.columns                 # a rajada ERA5 vai junto
    assert teste[["latitude", "longitude"]].notna().all().all()
    arms = json.loads((data / "arms.json").read_text())
    assert not any(gs.REFERENCE_COLUMN in a["features"] for a in arms)
    assert "cin" in meta["dropped_null_columns"]


def test_the_static_group_file_is_found_under_its_old_or_new_name(tmp_path):
    """O arquivo do grupo 4 foi renomeado sem aviso (`..._estatico_cluster3` →
    `..._cluster3`), e a rodada passou a falhar por "arquivo não encontrado".
    Os dois nomes valem, para que uma renomeação não derrube um estudo pago."""
    raw = _escreve(tmp_path / "raw")
    novo = gs.group_path(raw, "grupo4")
    assert novo.name == "features_grupo4_cluster3.parquet"

    antigo = novo.with_name("features_grupo4_estatico_cluster3.parquet")
    novo.rename(antigo)
    assert gs.group_path(raw, "grupo4") == antigo
    assert gs.is_available(raw)
    assert set(gs.static_columns(raw)) >= {"z_surface_ponto", "sdor_ponto"}


def test_a_missing_group_file_is_reported_by_its_expected_name(tmp_path):
    raw = _escreve(tmp_path / "raw")
    gs.group_path(raw, "grupo2").unlink()
    assert not gs.is_available(raw)
    assert gs.group_path(raw, "grupo2").name == "features_grupo2_cluster3.parquet"


def test_the_study_runs_from_prepare_to_aggregate_and_measures_group_effects(tmp_path):
    """Ponta a ponta na estrutura por grupo: prepare → fit (Ridge) em todos os trimestres →
    aggregate. Cobre o elo que os testes por módulo não pegam: os arms que o prepare grava
    são os que o worker ajusta e os que a análise compara."""
    from sklearn.linear_model import Ridge

    from src.feature_study.core.analysis import load_arms, run_aggregate
    from src.feature_study.core.prepare import prepare
    from src.feature_study.core.worker import run_unit
    from test.conftest import build_synthetic_raw_dir

    raw = build_synthetic_raw_dir(tmp_path / "raw", DIAS)
    _escreve(raw)
    estudo = tmp_path / "estudo"
    prepare(str(raw), estudo, cluster_id=9, arm_sets=("anchors", "groups", "controls"))
    arms = [a for a in load_arms(estudo / "data") if a.name in ("base", "full", "drop_grp__grupo2", "ctrl__noise")]
    assert len(arms) == 4
    for season in ("DJF", "MAM", "JJA", "SON"):
        run_unit(estudo / "data", estudo, "full", season, arms, regressors=[Ridge], seed=42, out_tag="full")

    res = run_aggregate(estudo, estudo / "data", ["full"], label="t", n_boot=50)
    eff = res["effects"]
    comparacoes = set(eff["comparison"])
    assert {"full_vs_base", "drop_grp__grupo2", "ctrl__noise"} <= comparacoes
    assert (estudo / "summary" / "t" / "effects.csv").exists()
