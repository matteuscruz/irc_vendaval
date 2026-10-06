"""Robustez do estudo (`src/feature_study/robustness.py`): métricas de cauda,
esquemas de reamostragem, Diebold–Mariano, calibração e excedência.

Sintético e leve: cada teste conhece a resposta certa de antemão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study import robustness as rb
from src.feature_study.analysis import _draw_metrics, make_blocks


def _teste(n_dias=60, estacoes=("A", "B", "C"), seed=0):
    """Dois meses (jan/abr de 2000) separados por buraco, com 3 estações."""
    rng = np.random.default_rng(seed)
    dias = list(pd.date_range("2000-01-01", periods=n_dias // 2)) + \
        list(pd.date_range("2000-04-01", periods=n_dias // 2))
    linhas = [{"estacao": e, "time": d, "year": d.year, "month": d.month,
               TARGET: float(rng.gamma(4, 2.5))} for d in dias for e in estacoes]
    return pd.DataFrame(linhas)


TARGET = "daily_wind_gust_max"


# ── Reprodução do estudo ────────────────────────────────────────────────────

def test_month_scheme_reproduces_the_study_metrics_draw_for_draw():
    """A âncora de tudo: com blocos (ano, mês) e o mesmo gerador, as métricas por
    sorteio têm de ser IDÊNTICAS às de `analysis._draw_metrics`. Sem isso,
    comparar esquemas mediria a diferença entre dois códigos, não entre dois
    desenhos de bloco."""
    test = _teste()
    rng = np.random.default_rng(3)
    e = rng.normal(0, 2, len(test))
    y = test[TARGET].to_numpy(float)
    dias = rb.make_days(test)

    counts = rb.counts_by_blocks(rb.month_labels(test, dias), 50)
    novo = rb.metrics_from_totals(counts @ rb.day_sums(e, y, np.ones(len(e), bool), dias))

    blocks = make_blocks(test)
    por_bloco = np.vstack([np.ones(blocks.nb),
                           np.random.default_rng(42).multinomial(blocks.nb, np.full(blocks.nb, 1 / blocks.nb),
                                                                 size=50).astype(float)])
    velho = _draw_metrics(por_bloco, e, blocks)
    assert np.allclose(novo["rmse"], velho["rmse"])
    assert np.allclose(novo["rmse_p90"], velho["rmse_p90"])
    assert np.allclose(novo["bias_p90"], velho["bias_p90"])


# ── Métricas de cauda ───────────────────────────────────────────────────────

def test_tail_metrics_match_a_direct_computation():
    test = _teste()
    rng = np.random.default_rng(1)
    y = test[TARGET].to_numpy(float)
    e = rng.normal(0, 1.5, len(y))
    dias = rb.make_days(test)
    m = rb.metrics_from_totals(np.ones((1, dias.n_days)) @ rb.day_sums(e, y, np.ones(len(y), bool), dias))

    for q, tag in ((0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
        cauda = y >= np.quantile(y, q)
        assert np.isclose(m[f"rmse_{tag}"][0], np.sqrt(np.mean(e[cauda] ** 2)))
        assert np.isclose(m[f"bias_{tag}"][0], np.mean(e[cauda]))
    assert np.isclose(m["rmse"][0], np.sqrt(np.mean(e ** 2)))


def test_pinball_loss_is_asymmetric_and_penalises_underprediction_at_high_tau():
    """Com τ=0,99, errar para BAIXO custa 99× mais que para cima: é a perda que
    pune justamente a subestimação de extremos."""
    test = _teste()
    y = test[TARGET].to_numpy(float)
    dias = rb.make_days(test)
    ones = np.ones((1, dias.n_days))
    todos = np.ones(len(y), bool)
    sub = rb.metrics_from_totals(ones @ rb.day_sums(np.full(len(y), -1.0), y, todos, dias))   # prevê 1 a menos
    sup = rb.metrics_from_totals(ones @ rb.day_sums(np.full(len(y), +1.0), y, todos, dias))   # prevê 1 a mais
    assert np.isclose(sub["pinball_99"][0], 0.99)
    assert np.isclose(sup["pinball_99"][0], 0.01)
    assert np.isclose(sub["pinball_90"][0], 0.90) and np.isclose(sup["pinball_90"][0], 0.10)


def test_rows_outside_the_mask_never_contribute_to_any_metric():
    """Cada modelo só existe nos trimestres em que foi eleito; as linhas sem
    resíduo não podem entrar nem como zero."""
    test = _teste()
    y = test[TARGET].to_numpy(float)
    dias = rb.make_days(test)
    e = np.full(len(y), 1.0)
    mask = np.zeros(len(y), bool)
    mask[: len(y) // 2] = True
    m = rb.metrics_from_totals(np.ones((1, dias.n_days)) @ rb.day_sums(np.where(mask, e, np.nan), y, mask, dias))
    assert np.isclose(m["rmse"][0], 1.0)


# ── Eventos e esquemas de reamostragem ──────────────────────────────────────

def _calendario(estacoes=("A", "B", "C"), exceder=()):
    """Janeiro de 2000 (31 dias). `exceder` = dias com rajada alta em 2+ estações."""
    linhas = []
    for d in pd.date_range("2000-01-01", periods=31):
        alto = d.day in exceder
        for k, e in enumerate(estacoes):
            linhas.append({"estacao": e, "time": d, "year": 2000, "month": 1,
                           TARGET: 30.0 if alto and k < 2 else 5.0 + 0.01 * d.day})
    return pd.DataFrame(linhas)


def test_consecutive_active_days_form_one_event_and_a_one_day_gap_is_bridged():
    test = _calendario(exceder={5, 6, 7, 9, 20})
    dias = rb.make_days(test)
    rot, ev = rb.event_labels(test, dias, k=2, gap_days=1)
    assert len(ev) == 2                                         # {5..9} e {20}
    assert ev.iloc[0]["dias"] == 5 and ev.iloc[1]["dias"] == 1  # o dia 8 calmo é ponte
    assert len(set(rot[4:9])) == 1                              # 5..9 no mesmo bloco


def test_every_day_belongs_to_exactly_one_block_and_quiet_blocks_are_short():
    """Sem rótulo `-1` sobrando, e nenhum trecho calmo do tamanho de um mês: um
    bloco calmo gigante reintroduziria o problema dos blocos ano-mês."""
    test = _calendario(exceder={10, 11})
    dias = rb.make_days(test)
    rot, ev = rb.event_labels(test, dias, k=2, quiet_chunk=7)
    assert (rot >= 0).all()
    tam = pd.Series(rot).value_counts()
    eventos = set(ev["evento"])
    assert all(tam[b] <= 7 for b in tam.index if b not in eventos)


def test_event_blocks_never_cross_a_calendar_gap():
    """Jan e abr não são adjacentes: nenhum bloco pode juntar o fim de um com o
    começo do outro."""
    test = _teste(n_dias=20)
    test.loc[test["time"].isin([pd.Timestamp("2000-01-10"), pd.Timestamp("2000-04-01")]), TARGET] = 99.0
    dias = rb.make_days(test)
    rot, _ = rb.event_labels(test, dias, k=1)
    for b in np.unique(rot):
        idx = np.flatnonzero(rot == b)
        assert len(set(dias.run_id[idx])) == 1


def test_moving_block_counts_keep_the_original_row_and_the_sample_size():
    test = _teste()
    dias = rb.make_days(test)
    c = rb.counts_moving_block(dias, 5, 200)
    assert (c[0] == 1).all()
    tot = c[1:].sum(axis=1)
    assert tot.min() >= dias.n_days and tot.max() <= dias.n_days + 5      # ceil(N/L)·L


def test_moving_block_refuses_a_length_with_no_contiguous_window():
    """Com sequências de 2 dias e blocos de 3, qualquer janela atravessaria um
    buraco — melhor falhar do que correlacionar dias não adjacentes."""
    test = _teste(n_dias=4)                                     # dois meses de 2 dias
    with pytest.raises(ValueError, match="contígua"):
        rb.counts_moving_block(rb.make_days(test), 3, 10)


def test_stationary_counts_are_nonnegative_and_cover_about_the_sample():
    test = _teste()
    dias = rb.make_days(test)
    c = rb.counts_stationary(dias, 5, 100)
    assert (c >= 0).all() and (c[0] == 1).all()
    assert c[1:].sum(axis=1).min() >= dias.n_days


# ── Efeitos pareados ────────────────────────────────────────────────────────

def _resid_par(test, melhor_menor=True, seed=0):
    rng = np.random.default_rng(seed)
    n = len(test)
    pior = rng.normal(0, 3.0, n)
    melhor = rng.normal(0, 1.0 if melhor_menor else 3.0, n)
    return {("t", "base", "m"): pior, ("t", "full", "m"): melhor}


def test_paired_effect_is_positive_when_the_better_arm_has_smaller_error():
    test = _teste()
    dias = rb.make_days(test)
    resid = _resid_par(test)
    counts = rb.counts_by_blocks(rb.month_labels(test, dias), 200)
    ef = rb.paired_effects(resid, test, dias, [("full_vs_base", "base", "full")], ["t"], ["m"], counts,
                           scheme="mes", metrics=("rmse", "pinball_95"))
    assert (ef["effect"] > 0).all() and (ef["ci_lo"] > 0).all()


def test_bias_effect_is_the_signed_shift_not_a_gain():
    """Como no estudo: para o viés, o efeito é `melhor − pior` COM SINAL (o viés é
    negativo e pode cruzar zero), e não uma redução de erro."""
    test = _teste()
    dias = rb.make_days(test)
    n = len(test)
    resid = {("t", "base", "m"): np.full(n, -3.0), ("t", "full", "m"): np.full(n, -1.0)}
    counts = rb.counts_by_blocks(rb.month_labels(test, dias), 50)
    ef = rb.paired_effects(resid, test, dias, [("c", "base", "full")], ["t"], ["m"], counts,
                           scheme="mes", metrics=("bias_p90",))
    assert np.isclose(ef["effect"].iloc[0], 2.0)               # de −3 para −1: +2


def test_a_pair_with_different_coverage_is_skipped_not_compared():
    test = _teste()
    dias = rb.make_days(test)
    n = len(test)
    a = np.full(n, 1.0)
    b = np.full(n, 1.0)
    b[: n // 2] = np.nan
    counts = rb.counts_by_blocks(rb.month_labels(test, dias), 20)
    ef = rb.paired_effects({("t", "base", "m"): a, ("t", "full", "m"): b}, test, dias,
                           [("c", "base", "full")], ["t"], ["m"], counts, scheme="mes", metrics=("rmse",))
    assert ef.empty


# ── Diebold–Mariano ─────────────────────────────────────────────────────────

def test_dm_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(0)
    run = np.zeros(400, dtype=int)
    real = rb.diebold_mariano(1.0 + rng.normal(0, 1, 400), run)
    nulo = rb.diebold_mariano(rng.normal(0, 1, 400), run)
    assert real["p"] < 1e-6 and real["dm"] > 0
    assert nulo["p"] > 0.01


def test_dm_hac_widens_the_test_under_autocorrelation():
    """Um diferencial fortemente autocorrelacionado tem bem menos informação do que
    o número de dias sugere; o HAC tem de refletir isso, e o t ingênuo não."""
    rng = np.random.default_rng(1)
    x = np.zeros(500)
    for i in range(1, 500):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    x += 0.15
    run = np.zeros(500, dtype=int)
    hac = rb.diebold_mariano(x, run, max_lag=15)
    ing = rb.diebold_mariano(x, np.arange(500), max_lag=15)    # cada dia numa sequência: sem autocovariância
    assert abs(hac["dm"]) < abs(ing["dm"])


def test_dm_only_pairs_days_within_the_same_contiguous_run():
    """Os meses de teste não são contíguos: correlacionar o último dia de um mês
    com o primeiro do seguinte, sem nada entre eles, seria correlacionar o que não
    é adjacente."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.3, 1, 200)
    run_um = np.zeros(200, dtype=int)
    run_cada = np.arange(200)
    com = rb.diebold_mariano(d, run_um, max_lag=5)
    sem = rb.diebold_mariano(d, run_cada, max_lag=5)
    assert np.isfinite(com["dm"]) and np.isfinite(sem["dm"])
    # com sequências unitárias não há par válido: a variância cai para γ0
    z = d - d.mean()
    esperado = d.mean() / np.sqrt((z @ z / 200) / 200) * np.sqrt(199 / 200)
    assert np.isclose(sem["dm"], esperado)


def test_dm_returns_nan_on_too_few_days():
    assert np.isnan(rb.diebold_mariano(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int))["dm"])


# ── Calibração e excedência ─────────────────────────────────────────────────

def test_bias_given_the_observation_is_negative_even_for_a_calibrated_forecast():
    """O motivo de `reliability` existir. Um previsor PERFEITAMENTE calibrado
    (obs = prev + ruído de média zero) tem viés NEGATIVO se os dias forem
    escolhidos pelo OBSERVADO alto — regressão à média —, mas viés ZERO se forem
    escolhidos pela PREVISÃO alta. Afirmar "o modelo subestima a cauda" com a
    primeira medida confunde esse artefato com falta de habilidade."""
    rng = np.random.default_rng(0)
    pred = rng.gamma(4, 2.5, 200_000)
    y = pred + rng.normal(0, 2.0, pred.size)

    top_obs = y >= np.quantile(y, 0.99)
    assert (pred[top_obs] - y[top_obs]).mean() < -1.0           # parece subestimar muito

    r = rb.reliability(y, pred)
    topo = r.iloc[-1]                                           # faixa 0,99–1,00 da PREVISÃO
    assert abs(topo["obs_media"] - topo["prev_media"]) < 0.15   # e está calibrado


def test_reliability_exposes_a_forecast_that_overshoots_at_the_top():
    rng = np.random.default_rng(1)
    y = rng.gamma(4, 2.5, 100_000)
    pred = y.mean() + 1.4 * (y - y.mean()) + rng.normal(0, 1.0, y.size)   # exagera a amplitude
    topo = rb.reliability(y, pred).iloc[-1]
    assert topo["obs_media"] < topo["prev_media"] - 1.0


def test_contingency_scores_on_a_hand_made_table():
    y = np.array([20, 20, 20, 5, 5, 5, 5, 5.0])
    pred = np.array([20, 20, 5, 20, 5, 5, 5, 5.0])
    c = rb.contingency(y, pred, 15.0)
    assert (c["n_obs_exc"], c["n_prev_exc"]) == (3, 3)
    assert np.isclose(c["pod"], 2 / 3) and np.isclose(c["far"], 1 / 3)
    assert np.isclose(c["csi"], 2 / 4) and np.isclose(c["freq_bias"], 1.0)
