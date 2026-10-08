"""O melhor modelo de ML e a LSTM, cada um com todas as features e com as selecionadas, nas MESMAS linhas.

Serve à pergunta prática "quais variáveis devemos de fato deixar no dataset?": cada modelo escolhe as suas
variáveis, na VALIDAÇÃO (nunca no teste), pela mesma regra — uma variável fica se, ao ser embaralhada, piora o RMSE
acima do piso de uma coluna de ruído e em pelo menos 3 dos 4 trimestres. O que os DOIS escolhem é o que se mantém
com confiança; o que só um escolhe é candidato.

  - `ml_predict`: o modelo de ML eleito pelo estudo (o `rank 1` do `top_models.json`) num conjunto de colunas;
  - `lstm_season_fits`: a LSTM em vários conjuntos de colunas, com as janelas completas para a UNIÃO deles (todos
    os conjuntos avaliam as mesmas linhas), e a importância por variável medida na validação;
  - `select_by_rule`: a regra de seleção, igual para os dois.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.core.config import CLIP_RANGE
from src.feature_study.core.worker import select_regressors
from src.feature_study.core.worker_lstm import (
    DEFAULTS, fit_arm_lstm, permutation_importance, split_features, stack_inputs, valid_rows,
)
from src.feature_study.data import groups_source as gs
from src.feature_study.data import sequences as sq
from src.pipelines.common import TARGET_VAR, preprocess_df

NOISE = "ctrl_noise"


def _instancia(cls, seed: int):
    """Como o LazyPredict instancia: com `random_state` se o modelo aceita. CatBoost sem logs nem arquivos."""
    kw = {}
    try:
        if "random_state" in cls().get_params():
            kw["random_state"] = seed
    except Exception:
        pass
    if cls.__name__ == "CatBoostRegressor":
        kw.update(verbose=0, allow_writing_files=False)
    if cls.__name__ == "LGBMRegressor":
        kw["verbose"] = -1
    return cls(**kw)


def ml_predict(model_name: str, train, val, test, cols: list[str], seed: int = 42):
    """Ajusta o modelo `model_name` (um dos do LazyPredict) em `cols` e devolve `(pred_val, pred_test)`, já truncadas
    na faixa física. O pré-processamento é o do estudo: `RobustScaler` ajustado só no treino."""
    cls = select_regressors([model_name])[0]
    x_tr, x_va = preprocess_df(train[cols], val[cols])
    _, x_te = preprocess_df(train[cols], test[cols])
    m = _instancia(cls, seed).fit(x_tr, train[TARGET_VAR].to_numpy(float))
    clip = lambda p: np.clip(np.asarray(p, float).ravel(), *CLIP_RANGE)  # noqa: E731
    return clip(m.predict(x_va)), clip(m.predict(x_te))


def lstm_season_fits(raw_dir, peaks: pd.DataFrame, train, val, test, sets: dict[str, list[str]], *, window: int = 24,
                     seed: int = 42, importance_set: str | None = None,
                     importance_blocks: dict[str, list[str]] | None = None, n_repeats: int = 3, **hp) -> dict:
    """A LSTM em cada conjunto de colunas de `sets`, num trimestre.

    Devolve `{"pred_test": {nome: Series alinhada ao índice de `test`, NaN onde a janela é incompleta},
    "importance": DataFrame | None, "linhas_fora": {split: n}}`. As janelas são lidas uma vez, para a união das colunas
    de todos os conjuntos, e as linhas sem janela completa saem de TODOS (mesmas linhas para todos). A importância
    (por bloco de colunas, em `importance_blocks`) é medida na VALIDAÇÃO do conjunto `importance_set`.
    """
    p = {**DEFAULTS, "window_hours": window, **hp}
    seq_all = sorted({c for g in gs.HOURLY_GROUPS for c in gs.hourly_columns(raw_dir, g)})
    partes = {n: split_features(cols, seq_all) for n, cols in sets.items()}
    uniao = sorted({c for seq, _ in partes.values() for c in seq})
    estacoes = sorted(set(train["estacao"].astype(str)) | set(test["estacao"].astype(str)))
    w = {k: sq.load_windows(raw_dir, df, uniao, p["window_hours"], peaks, stations=estacoes)
         for k, df in (("train", train), ("val", val), ("test", test))}
    ok = {k: valid_rows(v) for k, v in w.items()}
    quadros = {k: df[ok[k]].reset_index(drop=True) for k, df in (("train", train), ("val", val), ("test", test))}
    w = {k: v[ok[k]] for k, v in w.items()}
    pos = {c: i for i, c in enumerate(uniao)}

    preds, imp = {}, None
    for nome, (seq, static) in partes.items():
        idx = [pos[c] for c in seq]

        def entrada(k):
            df = quadros[k]
            est = df[static].to_numpy("float32") if static else np.empty((len(df), 0), "float32")
            return stack_inputs(w[k][:, :, idx], est)

        keep: dict = {}
        fit_arm_lstm(entrada("train"), quadros["train"][TARGET_VAR].to_numpy(), entrada("val"),
                     quadros["val"][TARGET_VAR].to_numpy(), entrada("test"), quadros["test"][TARGET_VAR].to_numpy(),
                     np.arange(len(quadros["test"])), seed=seed, model_out=keep, **p)
        s = pd.Series(np.nan, index=test.index, dtype="float64")
        s.loc[test.index[ok["test"]]] = keep["pred_test"]
        preds[nome] = s
        if nome == importance_set and importance_blocks is not None:
            imp = permutation_importance(keep["predict"], keep["x_val_scaled"], keep["y_val"], seq + static, {},
                                         n_repeats=n_repeats, seed=seed, blocos=importance_blocks)
    return {"pred_test": preds, "importance": imp, "linhas_fora": {k: int((~m).sum()) for k, m in ok.items()}}


def select_by_rule(imp_por_trimestre: dict[str, pd.DataFrame], *, ruido: str = NOISE, min_trimestres: int = 3) -> pd.DataFrame:
    """A regra de seleção, igual para ML e LSTM. `imp_por_trimestre[season]` tem `nome` e `d_rmse` (uma linha por
    repetição, medidas na VALIDAÇÃO). Mantém a variável se a piora média passa do piso do ruído (o maior efeito do
    bloco `ruido` entre os trimestres) e é positiva em pelo menos `min_trimestres`."""
    por_s = pd.DataFrame({s: d.groupby("nome").d_rmse.mean() for s, d in imp_por_trimestre.items()})
    piso = float(por_s.loc[ruido].max())
    v = por_s.drop(index=ruido)
    out = pd.DataFrame({"d_rmse": v.mean(axis=1), "trimestres > 0": (v > 0).sum(axis=1)})
    out["decisão"] = np.where((out["d_rmse"] > piso) & (out["trimestres > 0"] >= min_trimestres), "manter", "descartar")
    res = out.join(v)
    res.attrs["piso"] = piso        # `join` não propaga `attrs`
    return res


def consenso(ml: list[str], lstm: list[str], todas: list[str]) -> pd.DataFrame:
    """O que ML e LSTM escolhem: `ambos` (manter), `só ML`, `só LSTM` (candidatos) e `nenhum` (descartar)."""
    a, b = set(ml), set(lstm)
    rot = {v: ("ambos" if v in a and v in b else "só ML" if v in a else "só LSTM" if v in b else "nenhum") for v in todas}
    return pd.DataFrame({"variável": list(rot), "veredito": list(rot.values())})
