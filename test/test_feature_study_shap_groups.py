"""SHAP por grupo (`src/feature_study/shap_groups.py`).

Dado sintético com dependência conhecida, para que a resposta certa seja sabida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from src.feature_study import shap_groups as sg


def _dados(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a1": rng.normal(size=n), "a2": rng.normal(size=n),   # grupo 1, carregam o sinal
        "b1": rng.normal(size=n),                              # grupo 2, ruído puro
        "ctrl_noise": rng.normal(size=n),                      # controle
    })
    y = 3.0 * X.a1 + 1.5 * X.a2 + rng.normal(0, 0.3, n)
    return X, y.to_numpy()


GRUPOS = {"a1": "grupo1", "a2": "grupo1", "b1": "grupo2"}


def _modelo(X, y):
    return LGBMRegressor(n_estimators=80, random_state=0, verbose=-1, n_jobs=1).fit(X, y)


def test_shap_values_are_additive_with_the_model_prediction():
    """Sem aditividade o explicador não é o do modelo ajustado e nenhuma atribuição
    vale. É a verificação de sanidade de tudo o mais."""
    X, y = _dados()
    m = _modelo(X, y)
    sv, base = sg.shap_values(m, X)
    assert sg.check_additivity(m, X, sv, base) < 1e-6


def test_group_shap_is_the_row_wise_sum_of_its_columns():
    """O SHAP do grupo é a soma das colunas, linha a linha: é isso que torna o
    resultado imune à divisão de crédito entre colunas colineares."""
    X, y = _dados()
    m = _modelo(X, y)
    sv, _ = sg.shap_values(m, X)
    gs = sg.group_shap(sv, X.columns, GRUPOS)
    i1, i2 = list(X.columns).index("a1"), list(X.columns).index("a2")
    assert np.allclose(gs["grupo1"].to_numpy(), sv[:, i1] + sv[:, i2])
    # a soma entre grupos reproduz a soma total
    assert np.allclose(gs.sum(axis=1).to_numpy(), sv.sum(axis=1))


def test_the_informative_group_dominates_and_the_noise_control_is_the_floor():
    X, y = _dados()
    m = _modelo(X, y)
    sv, _ = sg.shap_values(m, X)
    fatia = sg.group_share(sg.group_shap(sv, X.columns, GRUPOS))
    assert fatia["grupo1"] > 85
    assert fatia[sg.CONTROL_NOISE] < 5 and fatia["grupo2"] < 5
    assert np.isclose(fatia.sum(), 100.0)


def test_collinear_columns_split_credit_but_the_group_total_does_not_change():
    """Duas cópias da mesma variável dividem o SHAP entre si, e cada uma parece
    metade tão importante. O grupo, que as soma, não muda — por isso a análise é
    por grupo."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=800)
    y = 2.0 * a + rng.normal(0, 0.2, 800)
    um = pd.DataFrame({"a": a, "r": rng.normal(size=800)})
    dois = pd.DataFrame({"a": a, "a_copia": a + rng.normal(0, 1e-3, 800), "r": rng.normal(size=800)})
    f1 = sg.group_share(sg.group_shap(sg.shap_values(_modelo(um, y), um)[0], um.columns, {"a": "G", "r": "R"}))
    f2 = sg.group_share(sg.group_shap(sg.shap_values(_modelo(dois, y), dois)[0], dois.columns,
                                      {"a": "G", "a_copia": "G", "r": "R"}))
    assert abs(f1["G"] - f2["G"]) < 5


def test_controls_are_labelled_apart_and_never_added_to_a_real_group():
    g = {"cape": "grupo2"}
    assert sg.group_of_column("ctrl_noise", g) == sg.CONTROL_NOISE
    assert sg.group_of_column("perm_sdor_ponto", g) == sg.CONTROL_PERM
    assert sg.group_of_column("cape", g) == "grupo2"
    assert sg.group_of_column("coluna_orfa", g) == sg.UNKNOWN


def test_an_unsupported_model_raises_instead_of_returning_a_wrong_explanation():
    """O HistGradientBoosting e as redes não passam pelo TreeExplainer; falhar alto é
    melhor que devolver uma atribuição de outro objeto."""
    X, y = _dados(200)
    mlp = MLPRegressor(hidden_layer_sizes=(4,), max_iter=50, random_state=0).fit(X, y)
    with pytest.raises(TypeError, match="não é suportado"):
        sg.shap_values(mlp, X)
