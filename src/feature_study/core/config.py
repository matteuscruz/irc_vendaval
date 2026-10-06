"""Constantes do estudo. FIXAS por decisão: somente o cluster 3 e seeds fixas.

Nada aqui é configurável por CLI — mudar um valor muda o experimento, e o
experimento anterior deixa de ser comparável. Todos os valores são gravados em
`meta.json` pelo estágio `prepare`.
"""
from __future__ import annotations

from src.pipelines.common import RANDOM_STATE

CLUSTER_ID = 3

# Seeds FIXAS do modelo. Uma seed só congela um sorteio da aleatoriedade de treino
# sem dizer o quão grande ela é; com estas 5 (todas fixas, gravadas no meta.json)
# cada arm é treinado 5 vezes e a variância entre seeds entra no erro padrão do
# efeito (`se_rep`). A 1ª é a seed das pipelines de produção. Rodar de novo
# reproduz cada seed bit a bit.
MODEL_SEEDS: tuple[int, ...] = (RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2,
                                RANDOM_STATE + 3, RANDOM_STATE + 4)
MODEL_SEED = MODEL_SEEDS[0]
BOOTSTRAP_SEED = 42


def seed_tag(seed: int) -> str:
    """Pasta de saída (`units/<tag>/`) de uma seed. A seed base mantém o nome
    `full` — é onde já está o run anterior, que continua valendo como réplica 1."""
    return "full" if seed == MODEL_SEED else f"full_s{seed}"


TRIAGE_SUFFIX = "_triage"


def triage_tag(seed: int) -> str:
    """Pasta da TRIAGEM (`units/<tag>_triage/`): os 39 modelos no arm `base`.

    Precisa ser separada da pasta do estudo. Triagem e estudo gravam o MESMO
    nome de arquivo para o arm `base` (`metrics__<trimestre>__base.parquet`), e o
    estudo o refaz com só 5 modelos: na mesma pasta, ele sobrescrevia a
    leaderboard de 39 modelos e a triagem deixava de ser auditável. Pior: rodar
    `screen` DEPOIS do estudo escolheria entre os 5 que já tinham sido escolhidos
    (medido: um top-5 encolheu para 4)."""
    return seed_tag(seed) + TRIAGE_SUFFIX

MAIN_TAGS = tuple(seed_tag(s) for s in MODEL_SEEDS)   # uma réplica de treino por seed

SEASONS = ("DJF", "MAM", "JJA", "SON")

# Efeito mínimo relevante: 1% do RMSE do arm base.
SESOI_REL = 0.01
BOOTSTRAP_DRAWS = 2000

# Faixa física, a mesma da inferência (src/inference/grid_direct_predict.py).
CLIP_RANGE = (0.0, 80.0)

# Conjunto de referência FIXADO A PRIORI (não escolhido nos dados: escolher o
# top-K pelo arm base geraria regressão à média contra as features).
REFERENCE_MODELS = (
    "CatBoostRegressor", "LGBMRegressor", "XGBRegressor",
    "HistGradientBoostingRegressor", "ExtraTreesRegressor",
    "RandomForestRegressor", "Ridge",
)
FAST_MODELS = ("CatBoostRegressor", "LGBMRegressor", "HistGradientBoostingRegressor")

# Métrica primária por eixo de comparação (menor erro = melhor; o veredito usa o sinal do ganho).
PRIMARY_METRIC = {"features": "rmse", "selection": "rmse"}

# Folga de R² da regra de campeão "R² com folga, depois RMSE_P90": entre os
# modelos a até CHAMPION_SLACK do melhor R², escolhe-se o de menor RMSE_P90.
# Sem a folga, a melhor cauda pode ser um modelo com R² negativo (medido: no
# DJF o melhor Bias_P90 era o RANSAC, com R² = −0,349).
CHAMPION_SLACK = 0.05

MODEL_FAMILY = {
    "CatBoostRegressor": "boosting",
    "LGBMRegressor": "boosting",
    "XGBRegressor": "boosting",
    "HistGradientBoostingRegressor": "boosting",
    "ExtraTreesRegressor": "bagging",
    "RandomForestRegressor": "bagging",
    "Ridge": "linear",
}

# Colunas de controle negativo (materializadas pelo `prepare`).
NOISE_COLUMN = "ctrl_noise"
PERM_PREFIX = "perm_"
