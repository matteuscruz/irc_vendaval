"""Eixo de PERDA do estudo: mesmo conjunto de features, funções de perda diferentes.

MSE e Huber são simétricas e puxam a predição para a média condicional — é daí
que vem a subestimação da cauda superior das rajadas. A expectila de nível `tau`
pondera o resíduo por `tau` quando o modelo SUBESTIMA e por `1 - tau` quando
superestima: `tau > 0,5` encarece prever abaixo e a predição sobe. Em `tau = 0,5`
a expectila É o MSE — é o invariante que o teste-portão verifica.

O Huber entra como DIAGNÓSTICO, não como candidato: acima de `delta` ele troca
MSE por MAE, limitando o gradiente justamente nos erros grandes. Medido no
cluster 3, 27,9% dos dias acima do P90 caem nessa zona linear. A LSTM do projeto
usa `Huber(delta=1.0)` sobre alvo escalonado por RobustScaler (1 IQR = 4,4 m/s),
e este braço mede quanto do viés de cauda dela vem da perda.

A perda entra por SUBCLASSE pré-parametrizada, não por `model_kwargs`: o
LazyPredict instancia `model(**kwargs)` e só injeta `random_state`, não há por
onde passar a perda. Três consequências desse caminho, todas verificadas:

1. `LazyRegressor._get_estimator_list` rotula por `cls.__name__`, e é esse rótulo
   que casa com REFERENCE_MODELS/MODEL_FAMILY e vira nome de coluna no
   `resid__*.parquet`. Por isso `__name__`/`__qualname__` são reescritos.
2. O LazyPredict só injeta `verbose`/`verbosity` quando `cls.__module__` contém
   "catboost"/"lightgbm"/"xgboost". O módulo da subclasse é ESTE arquivo, então o
   silenciamento não dispara sozinho e cada subclasse o passa à mão.
3. `random_state=None` no default é DELIBERADO para o CatBoost: o `get_params()`
   dele devolve só o que foi setado e descarta `None`, então
   `"random_state" in model().get_params()` é `False` e o LazyPredict não o
   semeia — nem aqui, nem no arm MSE contra o qual ele é comparado. Fixar a seed
   só deste lado enviesaria a comparação pareada.

XGBoost e a Huber: `reg:pseudohubererror` DIVERGE (média medida 209 onde a
verdadeira é 17) e a hessiana exata da Huber (≈0 fora da região quadrática) faz
`min_child_weight=1` podar toda folha, congelando o modelo no `base_score`. Usa-se
a pseudo-hessiana de Gauss-Newton (`hess ≡ 1`), que é o que o próprio XGBoost faz
em objetivos robustos.
"""
from __future__ import annotations

import numpy as np

from src.feature_study.config import LOSS_MODELS

EXPECTILE_TAUS: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)

# 1 IQR do alvo no cluster 3 (4,40 m/s) — o delta efetivo da LSTM, que usa
# delta=1.0 sobre alvo escalonado por RobustScaler. Em m/s porque o estudo NÃO
# escalona o alvo (`preprocess_df` só transforma as features).
HUBER_DELTA: float = 4.4


def _expectile_key(tau: float) -> str:
    return f"exp{int(round(tau * 100))}"


def _huber_key(delta: float) -> str:
    return f"huber{int(round(delta * 10))}"


def loss_keys() -> tuple[str, ...]:
    """Tokens de perda, na ordem dos braços. O token vira SUFIXO do nome do arm."""
    return tuple(_expectile_key(t) for t in EXPECTILE_TAUS) + (_huber_key(HUBER_DELTA),)


def loss_spec(key: str) -> dict:
    """{'kind', 'param', 'label'} da perda. Levanta em chave desconhecida —
    um token errado no nome de um arm seria um experimento silenciosamente
    diferente do que o nome promete."""
    for tau in EXPECTILE_TAUS:
        if key == _expectile_key(tau):
            return {"kind": "expectile", "param": float(tau), "label": f"expectila τ={tau:g}"}
    if key == _huber_key(HUBER_DELTA):
        return {"kind": "huber", "param": float(HUBER_DELTA), "label": f"Huber δ={HUBER_DELTA:g}"}
    raise ValueError(f"perda desconhecida: {key!r} (válidas: {loss_keys()})")


def expectile_objective(tau: float):
    """Objetivo custom (LightGBM/XGBoost): perda quadrática assimétrica.

    `e = pred - y`, então `e < 0` é SUBESTIMAR. O peso é `tau` nesse lado, logo
    `tau > 0,5` encarece subestimar. Em `tau = 0,5` o passo de Newton
    (`grad/hess = e`) é idêntico ao do MSE.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau fora de (0, 1): {tau}")

    def objective(y_true, y_pred):
        e = np.asarray(y_pred, float) - np.asarray(y_true, float)
        w = np.where(e < 0, tau, 1.0 - tau)
        return 2.0 * w * e, 2.0 * w

    objective.__name__ = f"expectile_{_expectile_key(tau)}"
    return objective


def huber_objective(delta: float):
    """Objetivo custom (XGBoost): Huber com pseudo-hessiana de Gauss-Newton.

    `hess ≡ 1` em vez da hessiana exata (que é ~0 fora da região quadrática):
    com a exata, `min_child_weight=1` do XGBoost poda toda folha cujo somatório
    de hessianas não chega a 1, e o modelo nunca sai do `base_score`.
    """
    if delta <= 0:
        raise ValueError(f"delta deve ser positivo: {delta}")

    def objective(y_true, y_pred):
        e = np.asarray(y_pred, float) - np.asarray(y_true, float)
        return np.clip(e, -delta, delta), np.ones_like(e)

    objective.__name__ = f"huber_{_huber_key(delta)}"
    return objective


class _CenteredTarget:
    """Treina no alvo centrado e soma a média de volta na predição.

    Um objetivo CUSTOMIZADO desliga o `boost_from_average` do LightGBM e a
    estimativa de `base_score` do XGBoost: o boosting partiria de 0 e teria de
    escalar até ~9 m/s com os próprios passos, deixando um deslocamento
    constante para BAIXO (medido: −0,18 m/s com 40 árvores) — exatamente a
    direção que o estudo mede, o que transformaria um artefato de inicialização
    em "efeito da perda".

    Centralizar é seguro para qualquer perda porque o resíduo não muda:
    `pred' − y' = (pred − m) − (y − m) = pred − y`. Então o `delta` da Huber
    continua significando a mesma coisa, e as três bibliotecas passam a partir
    do mesmo ponto — que é o que a comparação pareada exige.
    """

    def fit(self, x, y, **kw):
        self.offset_ = float(np.mean(np.asarray(y, dtype=float)))
        return super().fit(x, np.asarray(y, dtype=float) - self.offset_, **kw)

    def predict(self, x, **kw):
        return np.asarray(super().predict(x, **kw), dtype=float) + self.offset_


def _subclass(base: type, name: str, fixed: dict) -> type:
    """Subclasse que fixa `fixed` no `__init__`.

    `fixed` tem a última palavra sobre `**kw`: um `clone()` do sklearn devolve
    `objective`/`loss_function` dentro de `kw` e a chamada teria argumento
    duplicado. Deixar `fixed` vencer também garante que nenhum caminho consiga
    trocar a perda por baixo dos panos.
    """

    class _Sub(_CenteredTarget, base):  # type: ignore[misc, valid-type]
        def __init__(self, random_state=None, **kw):
            super().__init__(**{**kw, **fixed, "random_state": random_state})

    _Sub.__name__ = _Sub.__qualname__ = name
    return _Sub


def _catboost_class(spec: dict) -> type:
    from catboost import CatBoostRegressor as _Base

    loss = (f"Expectile:alpha={spec['param']:g}" if spec["kind"] == "expectile"
            else f"Huber:delta={spec['param']:g}")
    return _subclass(_Base, "CatBoostRegressor", {"loss_function": loss, "verbose": 0})


def _lightgbm_class(spec: dict) -> type:
    from lightgbm import LGBMRegressor as _Base

    # Huber é nativa no LightGBM (`alpha` É o delta); expectila não existe.
    fixed = ({"objective": expectile_objective(spec["param"])} if spec["kind"] == "expectile"
             else {"objective": "huber", "alpha": spec["param"]})
    return _subclass(_Base, "LGBMRegressor", {**fixed, "verbose": -1})


def _xgboost_class(spec: dict) -> type:
    from xgboost import XGBRegressor as _Base

    objective = (expectile_objective(spec["param"]) if spec["kind"] == "expectile"
                 else huber_objective(spec["param"]))
    return _subclass(_Base, "XGBRegressor", {"objective": objective, "verbosity": 0})


_BUILDERS = {
    "CatBoostRegressor": _catboost_class,
    "LGBMRegressor": _lightgbm_class,
    "XGBRegressor": _xgboost_class,
}


def make_regressors(key: str) -> list[type]:
    """Subclasses pré-parametrizadas de LOSS_MODELS para a perda `key`.

    `key == ""` devolve `[]`: perda nativa, quem escolhe os regressores é o
    `select_regressors` do worker.
    """
    if not key:
        return []
    spec = loss_spec(key)
    return [_BUILDERS[name](spec) for name in LOSS_MODELS]
