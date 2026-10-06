"""SHAP por GRUPO de features, com os controles negativos como régua.

O que o SHAP responde, e o que não: ele mede quanto o MODELO usa cada variável
para chegar à previsão (atribuição), não quanto cada variável melhora a previsão
fora da amostra (contribuição). Uma variável pode receber muito crédito e não
acrescentar nada para generalizar — foi o que a sondagem mostrou: as estáticas
PERMUTADAS entre estações, sem física alguma, receberam ~23 % do SHAP, o mesmo
que o grupo 4 verdadeiro, porque o modelo as usa só para identificar a estação.

Por isso toda leitura de SHAP aqui vem com os controles do estudo ao lado
(`ctrl__noise`, `ctrl__perm_static`): eles dão o que é "uso sem informação".

O SHAP de um grupo é a SOMA dos SHAP das suas colunas, linha a linha. A
aditividade é exata, e isso contorna a colinearidade: duas colunas quase iguais
dividem o crédito entre si, mas a soma do grupo não muda.

Só modelos de árvore suportados pelo `TreeExplainer` (LightGBM, CatBoost,
RandomForest, ExtraTrees, GradientBoosting, XGBoost). O HistGradientBoosting do
scikit-learn NÃO é suportado, e MLP, Bagging e lineares não são árvores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTROL_NOISE = "ruído (controle)"
CONTROL_PERM = "estáticas permutadas (controle)"
UNKNOWN = "sem grupo"

TREE_SHAP_MODELS = (
    "LGBMRegressor", "CatBoostRegressor", "RandomForestRegressor",
    "ExtraTreesRegressor", "GradientBoostingRegressor", "XGBRegressor",
)


def group_of_column(col: str, group_of: dict[str, str]) -> str:
    """Grupo de uma coluna. Os controles saem rotulados à parte para que ruído e
    estáticas trocadas nunca sejam somados a um grupo real."""
    if col == "ctrl_noise":
        return CONTROL_NOISE
    if col.startswith("perm_"):
        return CONTROL_PERM
    return group_of.get(col, UNKNOWN)


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Valores SHAP exatos (TreeSHAP) e valor-base. Levanta para modelo não suportado."""
    import shap

    nome = type(model).__name__
    if nome not in TREE_SHAP_MODELS:
        raise TypeError(
            f"{nome} não é suportado pelo TreeExplainer (suportados: {', '.join(TREE_SHAP_MODELS)})"
        )
    ex = shap.TreeExplainer(model)
    sv = np.asarray(ex.shap_values(X), dtype=float)
    base = float(np.ravel(ex.expected_value)[0])
    return sv, base


def group_shap(sv: np.ndarray, columns, group_of: dict[str, str]) -> pd.DataFrame:
    """(linhas × grupos): soma dos SHAP das colunas de cada grupo, por linha."""
    rotulos = [group_of_column(c, group_of) for c in columns]
    return pd.DataFrame(sv, columns=list(columns)).T.groupby(rotulos).sum().T


def group_share(gs: pd.DataFrame) -> pd.Series:
    """Fatia (%) de cada grupo no |SHAP| médio total. Soma 100 dentro de um modelo."""
    m = gs.abs().mean()
    return 100.0 * m / m.sum()


def check_additivity(model, X: pd.DataFrame, sv: np.ndarray, base: float) -> float:
    """Maior desvio entre (base + Σ SHAP) e a previsão do modelo. Tem de ser ~0:
    se não for, o explicador não é o do modelo ajustado."""
    return float(np.abs(base + sv.sum(axis=1) - np.asarray(model.predict(X), dtype=float)).max())
