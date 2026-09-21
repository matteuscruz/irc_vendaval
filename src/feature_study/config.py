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

# AMOSTRAGEM DESLIGADA (decisão: treinar no treino COMPLETO). O cluster 3 tem só
# 24.814 linhas de treino (~6 mil por trimestre) — a estimativa inicial de ~29 mil
# por trimestre vinha do `cluster_lazy`, que soma o cluster vizinho ao treino, e
# este estudo não usa vizinhos. Com a população tão pequena, amostrar quase não
# economiza (uma unidade de 39 modelos leva ~25 s) e as réplicas se sobreporiam
# em ~75%. Com seeds fixas e dados completos não há réplica a fazer: a única
# incerteza é o bootstrap em blocos do teste. O código de amostragem
# (`sampling.py`) continua testado e disponível: passe `seeds`/`pilot_sizes` a
# `prepare()` para reativá-lo.
SAMPLE_SEEDS: tuple[int, ...] = ()
MAIN_TAGS = tuple(seed_tag(s) for s in MODEL_SEEDS)   # uma réplica de treino por seed

SEASONS = ("DJF", "MAM", "JJA", "SON")

# Tamanho da amostra: limite DKW n = ln(2/alpha) / (2 eps^2). Heurística de
# dimensionamento (assume iid; o n efetivo é menor por dependência entre
# estações do mesmo dia) — quem valida o tamanho é a curva de aprendizado.
EPS = 0.02
ALPHA = 0.05
PILOT_SIZES: tuple[int, ...] = ()

# Efeito mínimo relevante: 1% do RMSE do arm base.
SESOI_REL = 0.01
BOOTSTRAP_DRAWS = 2000

# Faixa física, a mesma da inferência (src/inference/grid_direct_predict.py).
CLIP_RANGE = (0.0, 80.0)

# Idênticas a ws_max / ws_mean do ERA5-Bacia (diferença medida: 0,0000 m/s).
EXCLUDED_DUPLICATES = ("nf_ws10_max", "nf_ws10_mean")

# Conjunto de referência FIXADO A PRIORI (não escolhido nos dados: escolher o
# top-K pelo arm base geraria regressão à média contra as features).
REFERENCE_MODELS = (
    "CatBoostRegressor", "LGBMRegressor", "XGBRegressor",
    "HistGradientBoostingRegressor", "ExtraTreesRegressor",
    "RandomForestRegressor", "Ridge",
)
FAST_MODELS = ("CatBoostRegressor", "LGBMRegressor", "HistGradientBoostingRegressor")

# Modelos do EIXO DE PERDA — subconjunto de REFERENCE_MODELS que aceita trocar a
# função de perda. HistGB não tem expectila (só squared_error/absolute_error/
# poisson/quantile/gamma); ExtraTrees/RandomForest (só `criterion`) e Ridge
# também não. Incluir o HistGB apenas no braço Huber deixaria o conjunto de
# modelos irregular ENTRE perdas e confundiria o contraste Huber×expectila com a
# troca de modelos — por isso os mesmos 3 em todos os braços de perda.
LOSS_MODELS = ("CatBoostRegressor", "LGBMRegressor", "XGBRegressor")

# Métrica primária por eixo. RMSE é a métrica ERRADA para o eixo de perda: um
# modelo expectila perde em RMSE por construção (deixa de estimar a média
# condicional). `rmse_p90` continua sendo erro (menor é melhor), então a
# convenção de sinal e o veredito valem sem mudança.
PRIMARY_METRIC = {"features": "rmse", "loss": "rmse_p90"}

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

# Subconjunto de nf_* de orografia de sub-grade (sem a máscara terra-mar).
TOPOGRAPHY = ("nf_orog_height", "nf_sdor", "nf_isor", "nf_anor", "nf_slor", "nf_sdfor")
