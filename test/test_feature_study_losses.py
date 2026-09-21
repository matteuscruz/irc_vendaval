"""Eixo de perda (src/feature_study/losses.py).

Dado sintético pequeno e poucos estimadores: o objetivo aqui é a MECÂNICA da
perda (sinal, monotonicidade, sobrevivência da parametrização), não desempenho.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.feature_study import losses as losses_mod
from src.feature_study.config import LOSS_MODELS
from src.feature_study.losses import (
    EXPECTILE_TAUS, HUBER_DELTA, expectile_objective, huber_objective,
    loss_keys, loss_spec, make_regressors,
)

# Só o que as TRÊS bibliotecas aceitam (o CatBoost usa `thread_count`, não `n_jobs`).
FIT_KW = dict(n_estimators=40, random_state=0)


def _data(n=400, seed=0):
    """Alvo com cauda à direita, na escala de rajada (m/s)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 9.0 + 2.0 * x[:, 0] + rng.gamma(2.0, 1.5, n)
    return x, y


def _study_lgbm(tau):
    """O LGBM COMO O ESTUDO O CONSTRÓI — subclasse com a perda fixa e a
    centralização do alvo. Testar um `LGBMRegressor` cru aqui não exercitaria
    nada do que o eixo de perda realmente usa."""
    return losses_mod._lightgbm_class({"kind": "expectile", "param": tau})(**FIT_KW)


def _plain_lgbm(objective, **kw):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(objective=objective, verbose=-1, **FIT_KW, **kw)


# ── O invariante que sustenta o eixo inteiro ────────────────────────────────

def test_expectile_at_tau_half_reproduces_the_mse_fit():
    """Uma expectila de nível 0,5 É a média condicional: o passo de Newton
    (grad/hess) vale `e` nos dois casos. Se o gradiente customizado estiver
    errado — sinal trocado, peso invertido — este é o teste que quebra, e sem
    ele todo o eixo de perda mediria outra coisa que não a assimetria."""
    x, y = _data()
    meio = _study_lgbm(0.5).fit(x, y).predict(x)
    mse = _plain_lgbm("regression").fit(x, y).predict(x)

    assert np.allclose(meio, mse, rtol=5e-3, atol=5e-3)


def test_a_custom_objective_does_not_lose_the_average_it_boosts_from():
    """Objetivo customizado desliga o `boost_from_average` do LightGBM: sem
    centralizar o alvo, o boosting parte de 0 e fica devendo um deslocamento
    CONSTANTE para baixo (medido: −0,18 m/s com 40 árvores) — exatamente a
    direção que o estudo mede, o que viraria "efeito da perda"."""
    x, y = _data()
    mse = _plain_lgbm("regression").fit(x, y).predict(x)
    sem_centrar = _plain_lgbm(expectile_objective(0.5)).fit(x, y).predict(x)
    com_centrar = _study_lgbm(0.5).fit(x, y).predict(x)

    assert (mse - sem_centrar).mean() > 0.1          # o artefato existe
    assert abs((mse - com_centrar).mean()) < 0.01    # e a subclasse o remove


def test_higher_tau_raises_the_prediction_monotonically():
    """`tau > 0,5` encarece SUBESTIMAR, então a predição sobe. É a direção toda
    do eixo: se estivesse invertida, o estudo mediria uma perda que agrava a
    compressão da cauda em vez de corrigi-la."""
    x, y = _data()
    medias = [_study_lgbm(t).fit(x, y).predict(x).mean() for t in (0.5, 0.7, 0.9)]

    assert medias[0] < medias[1] < medias[2]


def test_every_loss_arm_class_fits_and_predicts_on_the_gust_scale():
    """Cada perda × cada modelo tem de treinar e devolver predição na escala do
    alvo. Pega divergência (o `reg:pseudohubererror` do XGBoost dava média 209
    para alvo de média 17) e congelamento no `base_score` (média 0,5)."""
    x, y = _data()
    for key in loss_keys():
        for cls in make_regressors(key):
            pred = cls(**FIT_KW).fit(x, y).predict(x)
            assert np.isfinite(pred).all(), (key, cls.__name__)
            assert abs(pred.mean() - y.mean()) < 0.3 * y.mean(), (key, cls.__name__)


# ── O contrato com o LazyPredict ────────────────────────────────────────────

def test_loss_subclasses_keep_the_original_model_name():
    """`LazyRegressor._get_estimator_list` rotula por `cls.__name__`, e é esse
    rótulo que casa com REFERENCE_MODELS/MODEL_FAMILY e vira coluna do
    `resid__*.parquet`. Um nome de subclasse vazando (`_CatBoost`) tiraria o
    modelo de toda a análise sem erro nenhum."""
    for key in loss_keys():
        assert [c.__name__ for c in make_regressors(key)] == list(LOSS_MODELS)


def test_catboost_subclass_is_not_seeded_so_it_matches_the_mse_arm():
    """O `get_params()` do CatBoost devolve só o que foi setado e descarta
    `None`, então o LazyPredict NUNCA o semeia — nem no arm MSE contra o qual
    estes braços são comparados. Semear só o lado da perda tornaria a diferença
    medida parte seed, parte perda."""
    catboost = make_regressors("exp70")[0]

    assert "random_state" not in catboost().get_params()


def test_the_subclasses_do_not_rely_on_lazypredict_to_silence_them():
    """O LazyPredict só injeta `verbose`/`verbosity` quando `cls.__module__`
    contém o nome da biblioteca; o módulo destas subclasses é
    `src.feature_study.losses`, então o silenciamento não dispara e o log do
    Modal levaria um treino inteiro de CatBoost por modelo, arm e trimestre."""
    catboost, lgbm, xgb = make_regressors("exp70")

    assert catboost().get_params()["verbose"] == 0
    assert lgbm().get_params()["verbose"] == -1
    assert xgb().get_params()["verbosity"] == 0


def test_loss_survives_a_sklearn_clone():
    """Nenhum caminho do LazyPredict clona hoje, mas a perda vive no `__init__`
    e não em `get_params()` — então uma reconstrução futura a REAPLICA em vez de
    devolver silenciosamente um modelo MSE."""
    from sklearn.base import clone

    lgbm = make_regressors("exp80")[1]
    clonado = clone(lgbm())

    assert callable(clonado.get_params()["objective"])


# ── A armadilha do XGBoost ──────────────────────────────────────────────────

def test_xgboost_huber_learns_instead_of_staying_at_the_base_score():
    """A hessiana exata da Huber é ~0 fora da região quadrática; com ela o
    `min_child_weight=1` do XGBoost poda toda folha e o modelo congela no
    `base_score` (medido: média 0,5 para um alvo de média 17). Daí a
    pseudo-hessiana de Gauss-Newton."""
    from xgboost import XGBRegressor

    x, y = _data()
    pred = XGBRegressor(objective=huber_objective(HUBER_DELTA), n_estimators=40,
                        random_state=0, verbosity=0, n_jobs=1).fit(x, y).predict(x)

    assert abs(pred.mean() - y.mean()) < 0.2 * y.mean()


def test_huber_gradient_is_capped_but_expectile_grows_with_the_error():
    """A diferença que motiva o estudo, no gradiente: a Huber para de puxar
    acima de delta (é o que ignora os extremos), a expectila continua crescendo."""
    e_pequeno, e_grande = 1.0, 40.0
    y = np.zeros(2)
    pred = np.array([e_pequeno, e_grande])

    g_huber, _ = huber_objective(HUBER_DELTA)(y, pred)
    g_exp, _ = expectile_objective(0.8)(y, pred)

    assert g_huber[1] == pytest.approx(HUBER_DELTA)      # travado
    assert g_exp[1] > g_exp[0] * 10                       # proporcional ao erro


# ── Chaves ──────────────────────────────────────────────────────────────────

def test_loss_keys_cover_every_tau_and_the_huber_diagnostic():
    keys = loss_keys()

    assert len(keys) == len(EXPECTILE_TAUS) + 1
    assert keys[:len(EXPECTILE_TAUS)] == ("exp60", "exp70", "exp80", "exp90")
    assert loss_spec(keys[-1])["kind"] == "huber"
    assert [loss_spec(k)["param"] for k in keys[:-1]] == list(EXPECTILE_TAUS)


def test_an_unknown_loss_key_is_an_error_not_a_silent_mse():
    """Um token errado no nome do arm significaria um experimento diferente do
    que o nome promete — tem de falhar alto, não cair no default."""
    with pytest.raises(ValueError, match="perda desconhecida"):
        loss_spec("exp75")

    assert make_regressors("") == []
