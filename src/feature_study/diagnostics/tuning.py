"""Busca de hiperparâmetros (Optuna) do campeão de cada trimestre, mirando os EXTREMOS.

O campeão do estudo usa os hiperparâmetros padrão do LazyPredict e corta os picos (P99 ~3 m/s abaixo do INMET).
Aqui a busca otimiza, na VALIDAÇÃO, uma mistura do erro geral com o erro na cauda:

    objetivo = (1 − PESO_CAUDA) · RMSE  +  PESO_CAUDA · RMSE_P90       (P90 do observado na validação)

e o espaço de busca inclui o que ataca a cauda diretamente:
  - peso de amostra para os dias acima do P90 do TREINO (`peso_cauda`, 1 = sem peso), em todos os modelos;
  - perda assimétrica onde o modelo aceita: Expectile/Quantile no CatBoost, quantile no HistGradientBoosting.

O pipeline é o MESMO do estudo (o do LazyPredict, com o pré-processamento dele), só com outros parâmetros no passo
do modelo, e o teste só é tocado no fim, uma vez, para comparar padrão × otimizado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

PESO_CAUDA = 0.5
Q_CAUDA = 0.90


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def tail_metrics(y, pred, lim95: float, lim99: float) -> dict:
    """Erro geral e de cauda; POD/FAR nos limiares (do TREINO) de P95 e P99."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    cauda = y >= np.quantile(y, Q_CAUDA)
    m99 = y >= np.quantile(y, 0.99)
    out = {"RMSE": _rmse(y, pred), "RMSE_P90": _rmse(y[cauda], pred[cauda]),
           "vies_P99": float((pred[m99] - y[m99]).mean()), "P99_pred": float(np.quantile(pred, 0.99)),
           "P99_obs": float(np.quantile(y, 0.99))}
    for nome, lim in (("P95", lim95), ("P99", lim99)):
        ev, al = y >= lim, pred >= lim
        hit = int((ev & al).sum())
        out[f"POD_{nome}"] = hit / max(int(ev.sum()), 1)
        out[f"FAR_{nome}"] = int((~ev & al).sum()) / max(int(al.sum()), 1)
    return out


def objective_value(y, pred) -> float:
    y = np.asarray(y, float)
    cauda = y >= np.quantile(y, Q_CAUDA)
    return (1 - PESO_CAUDA) * _rmse(y, pred) + PESO_CAUDA * _rmse(y[cauda], np.asarray(pred)[cauda])


def suggest(trial, model: str, amplo: bool = False) -> dict:
    """Espaço de busca por modelo. Devolve parâmetros do modelo; `peso_cauda` é sugerido à parte.
    `amplo` usa o espaço estendido (mais parâmetros e faixas maiores)."""
    if amplo:
        return _suggest_amplo(trial, model)
    if model == "CatBoostRegressor":
        perda = trial.suggest_categorical("perda", ["RMSE", "Expectile", "Quantile"])
        p = {"depth": trial.suggest_int("depth", 4, 8),
             "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
             "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
             "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)}
        if perda != "RMSE":
            p["loss_function"] = f"{perda}:alpha={trial.suggest_float('alpha', 0.5, 0.9):.3f}"
        else:
            p["loss_function"] = "RMSE"
        return p
    if model == "HistGradientBoostingRegressor":
        perda = trial.suggest_categorical("perda", ["squared_error", "quantile"])
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
             "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
             "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
             "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
             "loss": perda}
        if perda == "quantile":
            p["quantile"] = trial.suggest_float("alpha", 0.5, 0.9)
        return p
    if model == "ExtraTreesRegressor":
        prof = trial.suggest_categorical("max_depth", [0, 12, 20, 30])
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_depth": None if prof == 0 else prof, "n_jobs": 2}
    if model == "MLPRegressor":
        camadas = trial.suggest_categorical("camadas", ["100", "128-64", "256-128", "64-64-32"])
        return {"hidden_layer_sizes": tuple(int(x) for x in camadas.split("-")),
                "alpha": trial.suggest_float("alpha_l2", 1e-6, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "max_iter": 500, "early_stopping": True}
    raise ValueError(f"sem espaço de busca para {model}")


def _suggest_amplo(trial, model: str) -> dict:
    if model == "CatBoostRegressor":
        perda = trial.suggest_categorical("perda", ["RMSE", "Expectile", "Quantile"])
        p = {"depth": trial.suggest_int("depth", 3, 10),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "iterations": trial.suggest_int("iterations", 200, 2000, step=100),
             "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
             "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
             "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50)}
        p["loss_function"] = ("RMSE" if perda == "RMSE"
                              else f"{perda}:alpha={trial.suggest_float('alpha', 0.5, 0.95):.3f}")
        return p
    if model == "HistGradientBoostingRegressor":
        perda = trial.suggest_categorical("perda", ["squared_error", "quantile", "absolute_error"])
        prof = trial.suggest_categorical("max_depth", [0, 4, 6, 8, 12])
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "max_iter": trial.suggest_int("max_iter", 100, 1500, step=50),
             "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 255),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200),
             "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 30.0, log=True),
             "max_features": trial.suggest_float("max_features", 0.3, 1.0),
             "max_depth": None if prof == 0 else prof, "loss": perda}
        if perda == "quantile":
            p["quantile"] = trial.suggest_float("alpha", 0.5, 0.95)
        return p
    if model == "ExtraTreesRegressor":
        prof = trial.suggest_categorical("max_depth", [0, 8, 12, 20, 30, 40])
        boot = trial.suggest_categorical("bootstrap", [False, True])
        p = {"n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
             "max_features": trial.suggest_float("max_features", 0.2, 1.0),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
             "max_depth": None if prof == 0 else prof, "bootstrap": boot, "n_jobs": 2}
        if boot:
            p["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
        return p
    if model == "MLPRegressor":
        camadas = trial.suggest_categorical("camadas", ["64", "128", "256", "128-64", "256-128", "512-256",
                                                        "64-64-32", "256-128-64"])
        return {"hidden_layer_sizes": tuple(int(x) for x in camadas.split("-")),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha_l2", 1e-6, 1.0, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "max_iter": 800, "early_stopping": True,
                "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 40)}
    raise ValueError(f"sem espaço de busca para {model}")


def fit_variant(default_pipe, x_tr, y_tr, params: dict, peso_cauda: float, lim_cauda: float):
    """Clona o pipeline do estudo, troca os parâmetros do passo do modelo e ajusta com peso na cauda."""
    passo = default_pipe.steps[-1][0]
    pipe = clone(default_pipe).set_params(**{f"{passo}__{k}": v for k, v in params.items()})
    w = np.where(y_tr >= lim_cauda, peso_cauda, 1.0)
    pipe.fit(x_tr, y_tr, **{f"{passo}__sample_weight": w})
    return pipe


def tune(default_pipe, model: str, x_tr, y_tr, x_va, y_va, *, n_trials: int = 40, timeout: int | None = None,
         seed: int = 42, amplo: bool = False):
    """Estudo Optuna na validação. Devolve (estudo, parâmetros do modelo, peso de cauda, limiar de cauda).

    No modo `amplo` o peso vai até 10 e o limiar a partir do qual o dia recebe o peso (P80/P90/P95 do treino)
    também é buscado; fora dele, peso em [1, 5] e limiar fixo no P90."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        params = suggest(trial, model, amplo)
        peso = trial.suggest_float("peso_cauda", 1.0, 10.0 if amplo else 5.0)
        q = trial.suggest_categorical("q_cauda", [0.8, 0.9, 0.95]) if amplo else Q_CAUDA
        lim_cauda = float(np.quantile(y_tr, q))
        trial.set_user_attr("lim_cauda", lim_cauda)
        pipe = fit_variant(default_pipe, x_tr, y_tr, params, peso, lim_cauda)
        pred = np.clip(np.asarray(pipe.predict(x_va), float).ravel(), 0, 80)
        trial.set_user_attr("params", {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()})
        return objective_value(y_va, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objetivo, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = study.best_trial
    params = {k: (tuple(v) if k == "hidden_layer_sizes" else v) for k, v in best.user_attrs["params"].items()}
    return study, params, float(best.params["peso_cauda"]), float(best.user_attrs["lim_cauda"])


def compare(default_pipe, tuned_pipe, frames: dict, lim95: float, lim99: float) -> pd.DataFrame:
    """Métricas de padrão × otimizado em cada split de `frames` ({split: (x, y)})."""
    linhas = []
    for split, (x, y) in frames.items():
        for nome, pipe in (("padrão", default_pipe), ("otimizado", tuned_pipe)):
            pred = np.clip(np.asarray(pipe.predict(x), float).ravel(), 0, 80)
            linhas.append({"split": split, "versao": nome, "objetivo": objective_value(y, pred),
                           **tail_metrics(y, pred, lim95, lim99)})
    return pd.DataFrame(linhas)
