"""Amostragem estratificada representativa (src/feature_study/sampling.py).

O que estes testes travam: a alocação soma exatamente n; a amostra NÃO depende
das features (idêntica entre arms); não há vazamento para o teste; e os
diagnósticos realmente reprovam uma amostra enviesada — um gate que nunca
falha não protege nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.sampling import (
    KEY, allocate_proportional, assert_no_gross_failure, build_replicate, dkw_epsilon,
    dkw_sample_size, draw_stratified, group_sizes, representativeness_report,
    row_hash, structural_violations, target_edges,
)
from src.pipelines.common import TARGET_VAR


def _pop(n_days=700, stations=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2005-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "estacao": np.repeat(stations, len(days)),
        "time": np.tile(days, len(stations)),
    })
    st_effect = df["estacao"].map({s: 1.0 + 0.4 * i for i, s in enumerate(stations)})
    df[TARGET_VAR] = rng.gamma(4.0, 1.0, len(df)) * st_effect.to_numpy()
    df["f1"] = df[TARGET_VAR] * 0.5 + rng.normal(0, 1, len(df))
    df["f2"] = rng.normal(size=len(df))
    df["season"] = "DJF"
    df["_split"] = "train"
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    return df.sort_values(KEY).reset_index(drop=True)


# ── Tamanho ─────────────────────────────────────────────────────────────────

def test_dkw_sizes_match_the_documented_values():
    # ceil, não round: n = 4611,1 não garante a tolerância, 4612 sim.
    assert dkw_sample_size(0.02) == 4612
    assert dkw_sample_size(0.015) == 8198
    assert dkw_sample_size(0.01) == 18445


def test_dkw_epsilon_inverts_the_size_and_bonferroni_widens_it():
    n = dkw_sample_size(0.02)
    assert dkw_epsilon(n) == pytest.approx(0.02, abs=1e-4)
    assert dkw_epsilon(n, m=50) > dkw_epsilon(n, m=1)


# ── Alocação ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 7, 100, 999])
def test_allocation_sums_to_n_and_stays_within_one_row_of_expected(n):
    counts = pd.Series([500, 120, 33, 7, 340, 1, 900], index=list("abcdefg"))
    alloc = allocate_proportional(counts, n)
    expected = counts * n / counts.sum()

    assert int(alloc.sum()) == n
    assert ((alloc - expected).abs() < 1.0 + 1e-9).all()
    assert (alloc <= counts).all()


def test_allocation_returns_everything_when_n_covers_the_population():
    counts = pd.Series([5, 3, 2])
    assert allocate_proportional(counts, 10).tolist() == [5, 3, 2]
    assert allocate_proportional(counts, 99).tolist() == [5, 3, 2]


# ── Sorteio ─────────────────────────────────────────────────────────────────

def test_draw_returns_exactly_n_rows_and_is_reproducible():
    pop = _pop()
    edges = target_edges(pop[TARGET_VAR].to_numpy())
    a = draw_stratified(pop, 300, 42, edges)
    b = draw_stratified(pop, 300, 42, edges)
    c = draw_stratified(pop, 300, 43, edges)

    assert len(a) == 300 and len(set(a)) == 300
    assert (a == b).all()
    assert set(a) != set(c)                    # outra seed, outra amostra


def test_sample_does_not_depend_on_feature_columns():
    """Requisito de todo o estudo: as MESMAS linhas em todos os arms. Trocar
    valores de feature — inclusive por NaN — não pode mudar o sorteio."""
    pop = _pop()
    edges = target_edges(pop[TARGET_VAR].to_numpy())
    original = draw_stratified(pop, 300, 42, edges)

    alterada = pop.copy()
    alterada["f1"] = np.nan
    alterada["f2"] = 12345.0
    alterada["coluna_nova"] = 1.0

    assert (draw_stratified(alterada, 300, 42, edges) == original).all()


def test_row_hash_is_deterministic_and_seed_sensitive():
    pop = _pop(n_days=50)
    assert (row_hash(pop, 42) == row_hash(pop, 42)).all()
    assert (row_hash(pop, 42) != row_hash(pop, 43)).any()


def test_sample_preserves_composition_and_the_target_distribution():
    pop = _pop(n_days=1500)
    edges = target_edges(pop[TARGET_VAR].to_numpy())
    s = pop.iloc[draw_stratified(pop, 800, 42, edges)]

    share_p = pop["estacao"].value_counts(normalize=True)
    share_s = s["estacao"].value_counts(normalize=True).reindex(share_p.index)
    assert (share_s - share_p).abs().max() < 0.01
    # Cauda: proporção de >= P90 preservada (alocação proporcional, não sobre-amostragem).
    p90 = np.quantile(pop[TARGET_VAR], 0.9)
    assert abs((s[TARGET_VAR] >= p90).mean() - 0.10) < 0.01


# ── Réplica ─────────────────────────────────────────────────────────────────

def _trainval(seed=0):
    tv = pd.concat([_pop(seed=seed), _pop(seed=seed + 1).assign(_split="val")], ignore_index=True)
    tv["time"] = tv["time"] + pd.to_timedelta(np.where(tv["_split"] == "val", 2000, 0), unit="D")
    tv["year"], tv["month"] = tv["time"].dt.year, tv["time"].dt.month
    return tv


def test_replicate_sizes_follow_the_rule_and_small_groups_enter_whole():
    tv = _trainval()
    s = build_replicate(tv, 42, n_target=300)
    sizes = s.groupby(["_split", "season"]).size().to_dict()

    assert sizes[("train", "DJF")] == 300
    assert sizes[("val", "DJF")] == 300                # 2800 disponíveis, n=300 também amostra
    s_all = build_replicate(tv, 42, n_target=10 ** 6)
    assert len(s_all) == len(tv)                       # n >= N ⇒ integral


def test_val_bands_come_from_the_training_population():
    """A validação não define nada: as faixas do alvo vêm do treino."""
    tv = _trainval()
    tv.loc[tv["_split"] == "val", TARGET_VAR] += 1000.0     # val totalmente fora do treino
    s = build_replicate(tv, 42, n_target=300)
    assert (s["_split"] == "val").sum() == 300              # não quebra nem some


# ── Diagnósticos e gates ────────────────────────────────────────────────────

def test_report_passes_a_proportional_sample():
    pop = _pop(n_days=1500)
    edges = target_edges(pop[TARGET_VAR].to_numpy())
    s = pop.iloc[draw_stratified(pop, 800, 42, edges)]
    rep = representativeness_report(pop, s, ["f1", "f2"])

    assert not (rep["flag"] == "fail").any()
    assert_no_gross_failure(rep)                            # não levanta


def test_report_rejects_a_deliberately_biased_sample():
    """Um diagnóstico que nunca reprova não protege nada: uma amostra só com os
    dias mais fortes tem de acender."""
    pop = _pop(n_days=1500)
    biased = pop.nlargest(800, TARGET_VAR)
    rep = representativeness_report(pop, biased, ["f1", "f2"])

    assert (rep["flag"] == "fail").any()
    with pytest.raises(ValueError, match="falha GROSSEIRA"):
        assert_no_gross_failure(rep)


def test_integral_sample_is_reported_as_such():
    pop = _pop(n_days=100)
    rep = representativeness_report(pop, pop, ["f1"])
    assert rep["check"].tolist() == ["integral"]


def test_structural_gates_catch_leakage_duplicates_and_nan():
    pop = _pop(n_days=200)
    test = pop.iloc[:40].assign(_split="test")
    pop_no_test = pop.iloc[40:].reset_index(drop=True)
    good = pop_no_test.iloc[:100]
    assert structural_violations(good, pop_no_test, test, ["f1", "f2"]) == []

    leak = pd.concat([good, test.iloc[:3].assign(_split="train")], ignore_index=True)
    assert any("vazamento" in v for v in structural_violations(leak, pop, test, ["f1"]))

    dup = pd.concat([good, good.iloc[:2]], ignore_index=True)
    assert any("duplicadas" in v for v in structural_violations(dup, pop_no_test, test, ["f1"]))

    nan = good.copy()
    nan.loc[nan.index[0], "f1"] = np.nan
    assert any("NaN" in v for v in structural_violations(nan, pop_no_test, test, ["f1"]))


def test_group_sizes_helper_matches_min_of_size_and_target():
    tv = _trainval()
    assert group_sizes(tv, 300) == {("train", "DJF"): 300, ("val", "DJF"): 300}
    assert group_sizes(tv, 10 ** 6)[("train", "DJF")] == int((tv["_split"] == "train").sum())
