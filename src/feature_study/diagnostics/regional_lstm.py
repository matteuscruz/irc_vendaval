"""A LSTM nos mesmos desenhos do ML regional (local, regional, global, LOCO), para comparar de igual para igual.

Mesmas linhas (estação, dia), mesmo split de calendário, mesmas 49 colunas sem `anor`, mesmo EQM ajustado na
validação do treino. O que muda é só o modelo: a LSTM do estudo (`worker_lstm.fit_arm_lstm`: LSTM(96) → Dropout
(0,3) → Dense(1), Huber, early stopping na validação) vê a janela das 24 horas até o pico, em vez de só a hora do
pico. As janelas vêm de `era5_features.peak_windows_year`, com as mesmas regras da linha no pico.

Uma linha cuja janela tem alguma hora faltando (ex.: os primeiros dias de 2000, sem o ano anterior) fica de fora
das duas famílias: a coluna `janela_ok` da base marca as linhas válidas, e o ML também é treinado e avaliado só
nelas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.core.worker_lstm import DEFAULTS, fit_arm_lstm, stack_inputs, valid_rows
from src.feature_study.diagnostics import quantile_map as qm
from src.pipelines.common import TARGET_VAR

SEASONS = ("DJF", "MAM", "JJA", "SON")


def _with_extra(x: np.ndarray, extra: np.ndarray | None) -> np.ndarray:
    """Acrescenta colunas constantes na linha (ex.: dia do ano) a cada passo da janela."""
    return x if extra is None else stack_inputs(x, extra)


def fit_season(X: np.ndarray, base: pd.DataFrame, train_clusters, season: str | None, seed: int = 42,
               extra: np.ndarray | None = None, **hp) -> dict:
    """Ajusta a LSTM de um trimestre (ou de todos, com `season=None`) com as linhas de treino dos `train_clusters`
    e o EQM na validação deles.

    `X` é `(len(base), T, F)` alinhado com `base` (pode ser um memmap); `extra` `(len(base), k)` são colunas
    constantes na linha, repetidas em cada passo (o dia do ano do modelo anual). Devolve o modelo, o scaler e o
    mapa."""
    m = base.cluster_id.isin(list(train_clusters)) & base.janela_ok
    if season is not None:
        m &= base.season == season
    tr = np.flatnonzero((m & (base._split == "train")).to_numpy())
    va = np.flatnonzero((m & (base._split == "val")).to_numpy())
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"{season} {train_clusters}: treino ({len(tr)}) ou validação ({len(va)}) vazios")
    x_tr = _with_extra(np.asarray(X[tr]), None if extra is None else extra[tr])
    x_va = _with_extra(np.asarray(X[va]), None if extra is None else extra[va])
    y_tr, y_va = base[TARGET_VAR].to_numpy(float)[tr], base[TARGET_VAR].to_numpy(float)[va]
    mo: dict = {}
    metr, _, seg = fit_arm_lstm(x_tr, y_tr, x_va, y_va, x_va, y_va, base.row_id.to_numpy()[va], seed=seed,
                                model_out=mo, **{**DEFAULTS, **hp})
    pred_va = np.clip(mo["predict"](mo["x_val_scaled"]), 0, 80)
    return {"model": mo["model"], "scaler": mo["scaler"], "predict": mo["predict"], "mapa": qm.fit(pred_va, mo["y_val"]),
            "y_mean": mo["y_mean"], "y_std": mo["y_std"], "metrics": metr, "seconds": seg, "n_train": len(tr),
            "n_val": len(va)}


def save_fit(fit: dict, path_base) -> None:
    """Grava o modelo (`<base>.keras`) e o resto (`<base>.joblib`: scaler, mapa EQM, média e desvio do alvo)."""
    import joblib
    from pathlib import Path

    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fit["model"].save(str(path_base) + ".keras")
    joblib.dump({k: fit[k] for k in ("scaler", "mapa", "y_mean", "y_std", "n_train", "n_val")},
                str(path_base) + ".joblib")


def load_fit(path_base) -> dict:
    """Recarrega o que `save_fit` gravou, com o mesmo `predict` (saída na escala do alvo)."""
    import joblib
    import tensorflow as tf

    meta = joblib.load(str(path_base) + ".joblib")
    model = tf.keras.models.load_model(str(path_base) + ".keras")
    mu, sd = meta["y_mean"], meta["y_std"]
    meta.update(model=model, predict=lambda x: np.asarray(model.predict(x, batch_size=512, verbose=0), float).ravel() * sd + mu)
    return meta


def predict_rows(fit: dict, X: np.ndarray, idx: np.ndarray, batch: int = 50_000,
                 extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(sem EQM, com EQM) para as linhas `idx` de `X`; NaN onde a janela tem lacuna."""
    bruto = np.full(len(idx), np.nan, dtype="float32")
    sc = fit["scaler"]
    for i in range(0, len(idx), batch):
        x = _with_extra(np.asarray(X[idx[i:i + batch]]), None if extra is None else extra[idx[i:i + batch]])
        ok = valid_rows(x)
        if ok.any():
            f = x.shape[2]
            xs = sc.transform(x[ok].reshape(-1, f)).reshape(x[ok].shape).astype("float32")
            bruto[i:i + batch][ok] = np.clip(fit["predict"](xs), 0, 80)
    mapeado = np.full(len(idx), np.nan, dtype="float32")
    ok = np.isfinite(bruto)
    mapeado[ok] = qm.apply(bruto[ok], fit["mapa"])
    return bruto, mapeado


def doy_features(base: pd.DataFrame) -> np.ndarray:
    """(len(base), 2) seno e cosseno do dia do ano, como `regional.with_doy`."""
    a = 2 * np.pi * pd.to_datetime(base["time"]).dt.dayofyear.to_numpy() / 365.2425
    return np.column_stack([np.sin(a), np.cos(a)]).astype("float32")
