"""Análise de efeitos (src/feature_study/analysis.py).

Cada teste gera resíduos sintéticos com uma verdade CONHECIDA e confere que a
análise a recupera. O caso-chave é a redundância: duas cópias da mesma feature
têm que dar ganho isolado > 0 e perda-ao-remover ≈ 0 — exatamente a leitura que
motivou o estudo.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_study.core.analysis import (
    benjamini_hochberg, compare_units, compute_effects, effects_by_season,
    make_blocks, ranking_features, run_aggregate,
)
from src.feature_study.core.arms import build_arms
from src.feature_study.core.config import REFERENCE_MODELS
from src.pipelines.common import TARGET_VAR

SEASON_OF_MONTH = {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"}


def _test_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2000, 2020):
        for month in (1, 4, 7, 10):
            n = 15
            rows.append(pd.DataFrame({
                "estacao": rng.choice(list("ABCDE"), n),
                "time": pd.Timestamp(year, month, 1) + pd.to_timedelta(rng.integers(0, 28, n), unit="D"),
                "year": year, "month": month, "season": SEASON_OF_MONTH[month],
                TARGET_VAR: rng.gamma(4.0, 2.0, n),
            }))
    df = pd.concat(rows, ignore_index=True)
    df["row_id"] = np.arange(len(df))
    return df


# sd do resíduo de cada arm. a e b são cópias uma da outra; c é único.
SCALES = {
    "base": 2.0, "full": 1.4,
    "add__nf_a": 1.7, "add__nf_b": 1.7, "add__nf_c": 1.7,
    "drop__nf_a": 1.4, "drop__nf_b": 1.4, "drop__nf_c": 1.7,
    "ctrl__noise": 2.0,
}


def _write(tmp_path, scales=SCALES, tags=("r0", "r1", "r2"), seed=0, identical=()):
    """Escreve data_dir + units com resíduos = escala × (0,9·u + 0,44·v): `u`
    compartilhado entre arms (correlação alta, como arms reais), `v` próprio."""
    rng = np.random.default_rng(seed)
    test = _test_frame()
    data = tmp_path / "data"
    data.mkdir()
    test.to_parquet(data / "test.parquet", index=False)
    grupo = {"x": "grupo1", "nf_a": "grupo2", "nf_b": "grupo2", "nf_c": "grupo3"}
    arms = build_arms(("x",), ("nf_a", "nf_b", "nf_c"), (), ("core", "controls"), group_of=grupo)
    (data / "arms.json").write_text(json.dumps([a.to_dict() for a in arms]))

    for tag in tags:
        (tmp_path / "units" / tag).mkdir(parents=True)
        shared = {m: rng.standard_normal(len(test)) for m in REFERENCE_MODELS}
        base_e = {m: scales["base"] * (0.9 * shared[m] + 0.44 * rng.standard_normal(len(test)))
                  for m in REFERENCE_MODELS}
        for arm, s in scales.items():
            for season in test["season"].unique():
                mask = (test["season"] == season).to_numpy()
                cols = {}
                for m in REFERENCE_MODELS:
                    # `base` e os arms marcados como idênticos usam o MESMO vetor.
                    e = base_e[m] if (arm == "base" or arm in identical) else s * (0.9 * shared[m] + 0.44 * rng.standard_normal(len(test)))
                    cols[m] = e[mask].astype("float32")
                df = pd.DataFrame({"row_id": test.loc[mask, "row_id"].to_numpy(), **cols})
                df.to_parquet(tmp_path / "units" / tag / f"resid__{season}__{arm}.parquet", index=False)
    return tmp_path, data, arms


def _effect(eff, comparison, metric="rmse", family="all"):
    row = eff[(eff["comparison"] == comparison) & (eff["metric"] == metric) & (eff["family"] == family)]
    assert len(row) == 1, (comparison, metric, family)
    return row.iloc[0]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out, data, arms = _write(tmp_path_factory.mktemp("study"))
    return compute_effects(out, data, ["r0", "r1", "r2"], arms, n_boot=400), out, data, arms


# ── O caso-chave ────────────────────────────────────────────────────────────

def test_a_useful_feature_is_recovered_with_the_right_sign(result):
    eff = result[0]
    add_c = _effect(eff, "add__nf_c")

    assert add_c["effect"] == pytest.approx(0.3, abs=0.06)     # 2,0 → 1,7
    assert add_c["ci_lo"] > 0
    assert add_c["verdict"] == "acrescenta"


def test_redundant_copies_have_gain_alone_but_no_loss_when_removed(result):
    """a e b carregam o mesmo sinal: cada uma sozinha ajuda (A > 0), mas tirar
    uma das duas não piora nada porque a outra cobre (L ≈ 0). É a leitura que
    o SHAP × experimento controlado deixou em aberto."""
    eff = result[0]
    a_gain = _effect(eff, "add__nf_a")["effect"]
    a_loss = _effect(eff, "drop__nf_a")
    c_loss = _effect(eff, "drop__nf_c")

    assert a_gain > 0.2
    assert abs(a_loss["effect"]) < 0.05
    assert c_loss["effect"] > 0.2 and c_loss["ci_lo"] > 0     # c é única: removê-la custa


def test_negative_control_covers_zero(result):
    ctrl = _effect(result[0], "ctrl__noise")
    assert abs(ctrl["effect"]) < 0.05
    assert ctrl["ci_lo"] <= 0 <= ctrl["ci_hi"]


def test_full_vs_base_is_the_sum_of_what_the_features_bring(result):
    assert _effect(result[0], "full_vs_base")["effect"] == pytest.approx(0.6, abs=0.06)


# ── Convenção de sinal e casos degenerados ──────────────────────────────────

def test_a_worse_add_arm_has_negative_effect(tmp_path):
    scales = {**SCALES, "add__nf_a": 2.6}                       # adicionar PIOROU
    out, data, arms = _write(tmp_path, scales)
    eff = compute_effects(out, data, ["r0", "r1", "r2"], arms, n_boot=200)
    a = _effect(eff, "add__nf_a")

    assert a["effect"] < -0.3
    assert a["verdict"] == "prejudica"


def test_identical_arms_have_exactly_zero_effect(tmp_path):
    out, data, arms = _write(tmp_path, identical=("ctrl__noise",))
    eff = compute_effects(out, data, ["r0", "r1", "r2"], arms, n_boot=200)
    ctrl = _effect(eff, "ctrl__noise")

    assert ctrl["effect"] == 0.0
    assert ctrl["ci_lo"] <= 0 <= ctrl["ci_hi"]
    assert ctrl["verdict"] == "sem informação"


def test_missing_arm_drops_only_its_own_comparisons(tmp_path):
    out, data, arms = _write(tmp_path)
    for p in (out / "units").rglob("resid__*__add__nf_b.parquet"):
        p.unlink()
    eff = compute_effects(out, data, ["r0", "r1", "r2"], arms, n_boot=100)

    assert "add__nf_b" not in set(eff["comparison"])
    assert {"add__nf_a", "drop__nf_b"} <= set(eff["comparison"])


def test_a_model_missing_from_one_arm_is_dropped_only_for_that_comparison(tmp_path):
    """Um regressor que divergiu num arm não tem resíduo lá. A comparação segue
    com os demais e `n_models` mostra que o conjunto foi menor — sem NaN
    preenchido."""
    out, data, arms = _write(tmp_path)
    for p in (out / "units").rglob("resid__*__add__nf_c.parquet"):
        pd.read_parquet(p).drop(columns=["Ridge"]).to_parquet(p, index=False)
    eff = compute_effects(out, data, ["r0", "r1", "r2"], arms, n_boot=100)

    c = _effect(eff, "add__nf_c")
    a = _effect(eff, "add__nf_a")
    assert c["n_models"] == len(REFERENCE_MODELS) - 1
    assert a["n_models"] == len(REFERENCE_MODELS)
    assert c["effect"] == pytest.approx(0.3, abs=0.08)
    assert "linear" not in set(eff[eff["comparison"] == "add__nf_c"]["family"])


def test_leaderboard_exposes_nonfinite_counts(tmp_path):
    from src.feature_study.core.analysis import models_leaderboard

    d = tmp_path / "units" / "t"
    d.mkdir(parents=True)
    base = {"arm": "base", "season": "DJF", "split": "test", "n": 10, "R2": 0.5, "RMSE": 1.0,
            "Bias": 0.0, "Bias_P90": -1.0, "RMSE_P90": 2.0, "fit_seconds": 1.0}
    pd.DataFrame([{**base, "model": "Ridge", "n_nonfinite": 0},
                  {**base, "model": "XGBRegressor", "RMSE": np.nan, "n_nonfinite": 7}]
                 ).to_parquet(d / "metrics__DJF__base.parquet")

    lb = models_leaderboard(tmp_path, ["t"]).set_index("model")
    assert lb.loc["XGBRegressor", "n_nonfinite"] == 7 and np.isnan(lb.loc["XGBRegressor", "RMSE"])
    assert lb.loc["Ridge", "n_nonfinite"] == 0


def test_no_residuals_at_all_is_an_error(tmp_path):
    out, data, arms = _write(tmp_path)
    for p in (out / "units").rglob("resid__*.parquet"):
        p.unlink()
    with pytest.raises(ValueError, match="nenhuma comparação"):
        compute_effects(out, data, ["r0"], arms, n_boot=50)


# ── Estrutura da saída ──────────────────────────────────────────────────────

def test_effects_are_reported_per_family_and_metric(result):
    eff = result[0]
    assert set(eff["family"]) == {"boosting", "bagging", "linear", "all"}
    assert set(eff["metric"]) == {"rmse", "rmse_p90", "r2", "bias_p90"}
    assert (eff["n_replicates"] == 3).all()
    assert (eff.loc[eff["family"] == "all", "n_models"] == len(REFERENCE_MODELS)).all()


def test_uncertainty_combines_block_bootstrap_and_replicate_spread(result):
    eff = result[0]
    row = _effect(eff, "add__nf_c")
    assert row["se"] == pytest.approx(np.hypot(row["se_boot"], row["se_rep"]))
    assert row["se_boot"] > 0
    assert row["ci_hi"] - row["ci_lo"] == pytest.approx(2 * 1.96 * row["se"])


def test_single_training_set_has_only_block_bootstrap_uncertainty(tmp_path):
    """Treino completo + seeds fixas ⇒ um único conjunto (R = 1): não há variância
    entre réplicas a estimar, e a incerteza é só a do bootstrap em blocos. O erro
    a evitar é inventar uma variância entre réplicas que não existe."""
    out, data, arms = _write(tmp_path, tags=("full",))
    eff = compute_effects(out, data, ["full"], arms, n_boot=200)
    row = _effect(eff, "add__nf_c")

    assert row["n_replicates"] == 1
    assert row["se_rep"] == 0.0
    assert row["se"] == pytest.approx(row["se_boot"])
    assert row["effect"] == pytest.approx(0.3, abs=0.08) and row["ci_lo"] > 0


def test_analysis_is_reproducible(tmp_path):
    out, data, arms = _write(tmp_path)
    a = compute_effects(out, data, ["r0", "r1"], arms, n_boot=150)
    b = compute_effects(out, data, ["r0", "r1"], arms, n_boot=150)
    pd.testing.assert_frame_equal(a, b)


def test_ranking_pairs_add_and_drop_per_feature(result):
    eff, _, _, arms = result
    rank = ranking_features(eff, arms).set_index("feature")

    assert set(rank.index) == {"nf_a", "nf_b", "nf_c"}
    assert rank.loc["nf_a", "A_gain_rmse"] > 0.2 and abs(rank.loc["nf_a", "L_loss_rmse"]) < 0.05
    assert rank.loc["nf_c", "L_loss_rmse"] > 0.2


def test_effects_by_season_has_one_row_per_comparison_and_season(result):
    eff, out, data, arms = result
    by_season = effects_by_season(out, data, ["r0", "r1", "r2"], arms)
    assert set(by_season["season"]) == {"DJF", "MAM", "JJA", "SON"}
    assert (by_season[by_season["comparison"] == "add__nf_c"]["gain_rmse"] > 0.1).all()


def test_run_aggregate_writes_every_summary_file(tmp_path):
    out, data, arms = _write(tmp_path)
    run_aggregate(out, data, ["r0", "r1", "r2"], label="t", n_boot=100)
    dest = out / "summary" / "t"
    for f in ("effects.csv", "ranking_features.csv", "ranking_groups.csv",
              "effects_by_season.csv", "ranking_features.png", "meta.json"):
        assert (dest / f).exists(), f


# ── Blocos e BH ─────────────────────────────────────────────────────────────

def test_blocks_are_year_month_and_tail_is_top_decile():
    test = _test_frame()
    b = make_blocks(test)
    assert b.nb == 20 * 4
    assert b.n_b.sum() == len(test)
    assert b.tail.mean() == pytest.approx(0.10, abs=0.02)


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q == pytest.approx([0.04, 0.05333333, 0.05333333, 0.5])
    assert (benjamini_hochberg(np.array([0.2])) == 0.2).all()


def test_compare_units_detects_nondeterminism(tmp_path):
    def unit(tag, r2):
        d = tmp_path / "units" / tag
        d.mkdir(parents=True)
        pd.DataFrame({"arm": "base", "season": "DJF", "model": "Ridge", "split": "test",
                      "R2": r2, "RMSE": 1.0, "Bias": 0.0, "Bias_P90": -1.0, "RMSE_P90": 2.0},
                     index=[0]).to_parquet(d / "metrics__DJF__base.parquet")

    unit("a", 0.5)
    unit("b", 0.5)
    unit("c", 0.5001)
    assert compare_units(tmp_path, "a", "b") == 0.0
    assert compare_units(tmp_path, "a", "c") == pytest.approx(1e-4)


def test_compare_units_only_looks_at_the_repeated_files(tmp_path):
    """O run principal guarda muito mais unidades que a repetição do piloto;
    comparar tudo daria FileNotFoundError em vez de comparar o que se repetiu."""
    row = {"arm": "base", "season": "DJF", "model": "Ridge", "split": "test",
           "R2": 0.5, "RMSE": 1.0, "Bias": 0.0, "Bias_P90": -1.0, "RMSE_P90": 2.0}
    for tag, files in (("main", ["base", "full", "add__x"]), ("repeat", ["base"])):
        d = tmp_path / "units" / tag
        d.mkdir(parents=True)
        for arm in files:
            pd.DataFrame({**row, "arm": arm}, index=[0]).to_parquet(d / f"metrics__DJF__{arm}.parquet")

    assert compare_units(tmp_path, "main", "repeat") == 0.0
    (tmp_path / "units" / "main" / "metrics__DJF__base.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        compare_units(tmp_path, "main", "repeat")
    with pytest.raises(FileNotFoundError, match="nenhuma unidade"):
        compare_units(tmp_path, "main", "vazio")


# ── Variabilidade entre seeds ───────────────────────────────────────────────

def _write_metrics(out, tag, seed, et_rmse):
    """Um arquivo de métricas: Ridge não depende da seed, ExtraTrees sim."""
    rows = []
    for split in ("val", "test"):
        for model, rmse in (("Ridge", 2.0), ("ExtraTreesRegressor", et_rmse)):
            rows.append({"arm": "base", "season": "DJF", "tag": tag, "model": model, "split": split,
                         "n": 10, "R2": 0.4, "RMSE": rmse, "Bias": 0.0, "Bias_P90": -3.0, "RMSE_P90": 5.0,
                         "n_nonfinite": 0, "lazypredict_R2_val": 0.4, "fit_seconds": 1.0,
                         "model_seed": seed, "models_mode": "all"})
    d = out / "units" / tag
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "metrics__DJF__base.parquet", index=False)


def test_seed_sensitivity_flags_only_the_models_the_seed_changes(tmp_path):
    from src.feature_study.core.analysis import seed_sensitivity

    for tag, seed, et in (("s42", 42, 2.10), ("s43", 43, 2.16), ("s44", 44, 2.04)):
        _write_metrics(tmp_path, tag, seed, et)
    sens = seed_sensitivity(tmp_path, ["s42", "s43", "s44"]).set_index("model")

    assert not sens.loc["Ridge", "estocastico"] and sens.loc["Ridge", "sd_rmse_max"] == 0.0
    assert sens.loc["ExtraTreesRegressor", "estocastico"]
    assert sens.loc["ExtraTreesRegressor", "sd_rmse_mean"] == pytest.approx(0.06)
    assert (sens["n_seeds"] == 3).all()


def test_seed_sensitivity_needs_at_least_two_seeds(tmp_path):
    from src.feature_study.core.analysis import seed_sensitivity

    _write_metrics(tmp_path, "s42", 42, 2.1)
    assert seed_sensitivity(tmp_path, ["s42"]).empty


def test_aggregate_writes_per_seed_tables_only_when_there_are_several_seeds(tmp_path):
    (tmp_path / "multi").mkdir()
    out, data, arms = _write(tmp_path / "multi", tags=("r0", "r1"))
    for tag, seed in (("r0", 42), ("r1", 43)):
        _write_metrics(out, tag, seed, 2.0 + 0.05 * seed % 3)
    run_aggregate(out, data, ["r0", "r1"], label="t", n_boot=50)
    dest = out / "summary" / "t"
    assert (dest / "models_leaderboard_by_seed.csv").exists() and (dest / "seed_sensitivity.csv").exists()
    assert set(pd.read_csv(dest / "models_leaderboard_by_seed.csv")["tag"]) == {"r0", "r1"}

    (tmp_path / "single").mkdir()
    out1, data1, _ = _write(tmp_path / "single", tags=("r0",))
    _write_metrics(out1, "r0", 42, 2.0)
    run_aggregate(out1, data1, ["r0"], label="t", n_boot=50)
    assert not (out1 / "summary" / "t" / "seed_sensitivity.csv").exists()


# ── Eixo de perda ───────────────────────────────────────────────────────────











def test_champion_rule_with_r2_slack_buys_the_tail_when_the_r2_costs_little(tmp_path):
    """A seleção por R² prefere o modelo mais COMPRIMIDO: maximizar R² é
    minimizar erro quadrático, cujo minimizador é a média condicional. A folga
    deixa trocar um R² irrelevante por cauda — mas só até o limite, senão se
    elege um modelo ruim (medido no DJF: a melhor cauda era o RANSAC, R²=−0,349).
    """
    from src.feature_study.core.analysis import champion_rules

    units = tmp_path / "units" / "r0"
    units.mkdir(parents=True)
    linhas = []
    for split in ("val", "test"):
        for modelo, r2, p90 in (("Comprimido", 0.80, 5.0), ("Cauda", 0.77, 4.2),
                                ("Degenerado", -0.35, 3.1)):
            linhas.append({"arm": "base", "season": "DJF", "model": modelo, "split": split,
                           "R2": r2, "RMSE": 2.0, "Bias": 0.0, "Bias_P90": -3.0,
                           "RMSE_P90": p90, "fit_seconds": 1.0, "n_nonfinite": 0,
                           "model_seed": 42, "models_mode": "all", "loss": "mse", "tag": "r0"})
    pd.DataFrame(linhas).to_parquet(units / "metrics__DJF__base.parquet", index=False)

    picks, resumo = champion_rules(tmp_path, ["r0"], slack=0.05)
    por_regra = picks.set_index("rule")["model"]

    assert por_regra["r2"] == "Comprimido"
    assert por_regra["rmse_p90"] == "Degenerado"          # sem trava, elege o R² negativo
    assert por_regra["r2_slack_then_rmse_p90"] == "Cauda"
    assert set(resumo["rule"]) == set(por_regra.index)

    apertada = champion_rules(tmp_path, ["r0"], slack=0.01)[0].set_index("rule")["model"]
    assert apertada["r2_slack_then_rmse_p90"] == "Comprimido"




# ── Cobertura parcial: cada modelo só existe nos trimestres em que foi eleito ──

def test_a_model_covering_only_some_seasons_is_measured_not_discarded(tmp_path):
    """Com a triagem top-5, um modelo só tem resíduo nos trimestres em que foi
    eleito. Exigir cobertura de TODAS as linhas do teste deixava só a interseção
    dos quatro trimestres — na execução real, um único modelo — e o efeito
    pareado passava a repousar nele, com incerteza subestimada."""
    from src.feature_study.core.analysis import _masked_draws

    test = pd.DataFrame({
        "year": [2000] * 6 + [2001] * 6, "month": [1] * 6 + [4] * 6,
        "daily_wind_gust_max": np.arange(12, dtype=float) + 1.0,
    })
    e = np.full(12, np.nan)
    e[:6] = 1.0                                   # o modelo só cobre o primeiro bloco
    r = _masked_draws(np.ones((1, 2)), e, test, {})
    assert r is not None
    draws, mask = r
    assert mask.sum() == 6
    assert np.isclose(draws["rmse"][0], 1.0)      # RMSE só sobre as linhas que existem


def test_masked_metrics_equal_the_full_metrics_when_coverage_is_complete():
    """A guarda de regressão: um estudo com todos os modelos em todas as linhas
    (o desenho antigo) tem de dar exatamente o mesmo resultado de antes."""
    from src.feature_study.core.analysis import _draw_metrics, _masked_draws, make_blocks

    rng = np.random.default_rng(3)
    test = pd.DataFrame({"year": np.repeat([2000, 2001, 2002], 40), "month": np.tile([1, 4], 60),
                         "daily_wind_gust_max": rng.gamma(4, 2, 120)})
    e = rng.normal(0, 2, 120)
    counts = np.vstack([np.ones(6), rng.multinomial(6, np.full(6, 1 / 6), size=5)]).astype(float)
    antigo = _draw_metrics(counts, e, make_blocks(test))
    novo, _ = _masked_draws(counts, e, test, {})
    for k in antigo:
        assert np.allclose(antigo[k], novo[k], equal_nan=True), k


def test_a_pair_covering_different_rows_is_skipped_instead_of_mixing_seasons(tmp_path):
    """Se o arm pior e o melhor cobrem linhas diferentes (uma unidade incompleta),
    a diferença misturaria trimestres. O par é descartado em silêncio seguro, não
    comparado."""
    from src.feature_study.core.analysis import _masked_draws

    test = pd.DataFrame({"year": [2000] * 4 + [2001] * 4, "month": [1] * 4 + [4] * 4,
                         "daily_wind_gust_max": np.arange(8, dtype=float) + 1})
    a = np.array([1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan])
    b = np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1])
    ra, rb = _masked_draws(np.ones((1, 2)), a, test, {}), _masked_draws(np.ones((1, 2)), b, test, {})
    assert not np.array_equal(ra[1], rb[1])


def test_discover_models_returns_every_model_that_has_residuals(tmp_path):
    """O modo top-5 mede os modelos que a triagem elegeu, e não só os 7 de
    referência fixados a priori."""
    from src.feature_study.core.analysis import discover_models

    pasta = tmp_path / "units" / "full"
    pasta.mkdir(parents=True)
    pd.DataFrame({"row_id": [0], "MLPRegressor": [1.0], "Ridge": [1.0]}).to_parquet(pasta / "resid__DJF__base.parquet")
    pd.DataFrame({"row_id": [1], "GradientBoostingRegressor": [1.0], "Ridge": [1.0]}).to_parquet(
        pasta / "resid__JJA__base.parquet")
    assert discover_models(tmp_path, ["full"]) == ["GradientBoostingRegressor", "MLPRegressor", "Ridge"]
