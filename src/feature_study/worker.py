"""Estágio `fit`: ajusta o LazyPredict INTEIRO para um lote de arms.

Uma unidade = (amostra, trimestre, lote de arms). Cada arm treina só na amostra
do trimestre (um modelo por trimestre, como nas pipelines de produção) e é
avaliado na validação (todos os modelos) e no TESTE COMPLETO do trimestre.

Saídas, com nomes determinísticos (a unidade é idempotente — pula o que já existe):
  units/<tag>/metrics__<season>__<arm>.parquet   R2/RMSE/Bias/Bias_P90/RMSE_P90
                                                 de cada modelo, em val e teste
  units/<tag>/resid__<season>__<arm>.parquet     resíduo (pred − y) por linha do
                                                 teste, só dos modelos de referência

O teste NÃO participa de nenhuma seleção aqui; é só onde se mede. Predições são
truncadas à faixa física [0, 80] m/s, como na inferência — sem isso um único
modelo degenerado (visto: KNeighbors em ~1e9) domina qualquer média.
"""
from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.arms import Arm
from src.feature_study.config import (
    CLIP_RANGE, FAST_MODELS, MODEL_SEED, REFERENCE_MODELS,
)
from src.feature_study.losses import make_regressors
from src.pipelines.common import RANDOM_STATE, TARGET_VAR, compute_metrics, preprocess_df

METRIC_COLS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]


def select_regressors(models: str = "all", loss: str = "") -> list:
    """`all` = os 39 do LazyPredict (mesmo filtro de `cluster_lazy`), `reference`
    = os 7 fixados a priori, `fast3` = os 3 do piloto.

    Com `loss`, o conjunto passa a ser o de `LOSS_MODELS` (os únicos que aceitam
    trocar a perda), ainda filtrado por `models`. Um modelo pedido por `models`
    que não exista nesse conjunto NÃO é erro aqui: o eixo de perda é
    intrinsecamente definido sobre um subconjunto, e `compute_effects` já
    restringe a comparação aos modelos presentes dos dois lados.
    """
    from src.pipelines.cluster_lazy import _FAST_REGRESSORS

    pool = make_regressors(loss) if loss else list(_FAST_REGRESSORS)
    if models == "all":
        return pool
    names = {"reference": REFERENCE_MODELS, "fast3": FAST_MODELS}.get(models)
    if names is None:
        raise ValueError(f"models inválido: {models!r} (all | reference | fast3)")
    chosen = [c for c in pool if c.__name__ in names]
    missing = set(names) - {c.__name__ for c in chosen}
    if missing and not loss:
        raise ValueError(f"modelos ausentes do LazyPredict: {sorted(missing)}")
    return chosen


NAN_METRICS = {c: np.nan for c in METRIC_COLS}


def _safe_metrics(y: np.ndarray, pred) -> tuple[dict | None, int]:
    """(métricas, nº de predições não finitas). Predição com NaN/inf devolve
    `None`: o modelo é registrado como falho, em vez de derrubar a unidade.

    Um único modelo degenerado (entre 39) já interrompeu um lote inteiro de
    arms. Falhar alto é o certo para dado quebrado, mas não para um regressor
    que diverge — esse é um resultado do arm, e fica visível na coluna
    `n_nonfinite` do leaderboard. NaN nunca é preenchido nem truncado: `np.clip`
    deixaria NaN passar, e inf viraria 80 m/s como se fosse uma predição.
    """
    pred = np.asarray(pred, float)
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        return None, bad
    return compute_metrics(np.asarray(y, float), np.clip(pred, *CLIP_RANGE)), 0


def _is_current(path: Path, models_mode: str, seed: int = MODEL_SEED,
                loss: str = "mse") -> bool:
    """Arquivo existente só vale se foi gerado com o MESMO conjunto de modelos,
    a MESMA seed e a MESMA perda.

    Sem isso, o piloto (3 modelos) e o run completo (39) compartilhando a
    pasta `units/full/` fariam o `fit` idempotente pular os arms do piloto e
    entregar métricas incompletas sem nenhum aviso. Arquivo antigo, sem a
    coluna, ou ilegível ⇒ refaz.
    """
    if not path.exists():
        return False
    try:
        cols = pd.read_parquet(path, columns=["models_mode", "model_seed"])
    except Exception:
        return False
    if not ((cols["models_mode"] == models_mode).all() and (cols["model_seed"] == seed).all()):
        return False
    # A coluna `loss` é NOVA. Lida à parte de propósito: juntá-la à leitura
    # acima faria TODO parquet anterior ao eixo de perda — o estudo inteiro no
    # volume, 37 arms × 4 trimestres × 5 seeds — levantar, devolver False e ser
    # refeito em silêncio. Ausente significa o que esses arquivos são: MSE.
    try:
        gravado = pd.read_parquet(path, columns=["loss"])["loss"]
    except Exception:
        return loss == "mse"
    return bool((gravado == loss).all())


def load_unit_frames(data_dir: Path, sample_tag: str, season: str):
    sample = pd.read_parquet(Path(data_dir) / f"sample_{sample_tag}.parquet")
    sample = sample[sample["season"] == season]
    test = pd.read_parquet(Path(data_dir) / "test.parquet")
    test = test[test["season"] == season]
    return (
        sample[sample["_split"] == "train"].reset_index(drop=True),
        sample[sample["_split"] == "val"].reset_index(drop=True),
        test.reset_index(drop=True),
    )


def fit_arm(train, val, test, features, regressors, seed: int = MODEL_SEED):
    """Ajusta os regressores em UM arm. Retorna (métricas, resíduos, segundos)."""
    from lazypredict.Supervised import LazyRegressor

    feats = list(features)
    t0 = time.time()
    x_tr, x_va = preprocess_df(train[feats], val[feats])
    _, x_te = preprocess_df(train[feats], test[feats])
    y_tr, y_va, y_te = (d[TARGET_VAR].to_numpy(float) for d in (train, val, test))

    reg = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True,
                        random_state=seed, regressors=regressors)
    scores, val_preds = reg.fit(x_tr, x_va, y_tr, y_va)
    fitted = reg.provide_models(x_tr, x_va, y_tr, y_va)
    lazy_r2 = scores["R-Squared"].to_dict()

    rows, resid = [], {}

    def record(name, split, y, pred):
        m, bad = _safe_metrics(y, pred)
        if m is None:
            print(f"   [AVISO] {name} gerou {bad} predição(ões) não finita(s) em {split} "
                  "— registrado como falho, sem métricas", flush=True)
        rows.append({"model": name, "split": split, "n": len(y), **(m or NAN_METRICS),
                     "n_nonfinite": bad, "lazypredict_R2_val": lazy_r2.get(name, np.nan)})
        return m is not None

    for name in val_preds.columns:
        record(name, "val", y_va, val_preds[name].to_numpy())
    for name, pipe in fitted.items():
        try:
            pred = np.asarray(pipe.predict(x_te), float).ravel()
        except Exception as exc:  # um modelo quebrado não derruba o arm
            print(f"   [AVISO] {name} falhou no teste ({type(exc).__name__}): {exc}", flush=True)
            continue
        if record(name, "test", y_te, pred) and name in REFERENCE_MODELS:
            resid[name] = (np.clip(pred, *CLIP_RANGE) - y_te).astype("float32")

    resid_df = pd.DataFrame(resid)
    resid_df.insert(0, "row_id", test["row_id"].to_numpy())
    return pd.DataFrame(rows), resid_df, time.time() - t0


def run_unit(
    data_dir, out_dir, sample_tag: str, season: str, arms: list[Arm],
    models: str = "all", seed: int = MODEL_SEED, skip_existing: bool = True,
    regressors: list | None = None, out_tag: str | None = None,
) -> list[Path]:
    """Ajusta todos os `arms` de um trimestre. Idempotente por arm.

    `out_tag` grava em outra pasta a MESMA amostra — é como o piloto repete uma
    unidade para confirmar o determinismo.

    Os regressores são escolhidos POR ARM, não uma vez por unidade: um arm do
    eixo de perda carrega a própria perda (`arm.loss`) e precisa das subclasses
    correspondentes. `regressors` explícito continua vencendo — é o atalho dos
    testes."""
    out = Path(out_dir) / "units" / (out_tag or sample_tag)
    out.mkdir(parents=True, exist_ok=True)
    models_mode = models if regressors is None else "custom"
    train, val, test = load_unit_frames(Path(data_dir), sample_tag, season)
    if train.empty or val.empty or test.empty:
        raise ValueError(f"amostra {sample_tag}/{season} sem treino, validação ou teste")

    written = []
    for arm in arms:
        regs = regressors if regressors is not None else select_regressors(models, arm.loss)
        arm_loss = arm.loss or "mse"
        m_path = out / f"metrics__{season}__{arm.name}.parquet"
        r_path = out / f"resid__{season}__{arm.name}.parquet"
        if skip_existing and r_path.exists() and _is_current(m_path, models_mode, seed, arm_loss):
            print(f"[fit] {sample_tag}/{season}/{arm.name}: já existe "
                  f"({models_mode}, {arm_loss}) — pulando", flush=True)
            written += [m_path, r_path]
            continue
        print(f"[fit] {sample_tag}/{season}/{arm.name}: {len(arm.features)} features, "
              f"{len(train)} treino, {len(val)} val, {len(test)} teste, "
              f"{len(regs)} modelos, perda {arm_loss}", flush=True)
        metrics, resid, secs = fit_arm(train, val, test, arm.features, regs, seed)
        metrics.insert(0, "arm", arm.name)
        metrics.insert(1, "season", season)
        metrics.insert(2, "tag", sample_tag)
        metrics["fit_seconds"] = round(secs, 2)
        metrics["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        metrics["model_seed"] = seed
        metrics["models_mode"] = models_mode
        metrics["loss"] = arm_loss
        metrics.to_parquet(m_path, index=False)
        resid.to_parquet(r_path, index=False)
        written += [m_path, r_path]
        print(f"[fit]   pronto em {secs:.0f}s", flush=True)
    return written


assert MODEL_SEED == RANDOM_STATE  # seeds fixas: o modelo usa a mesma seed das pipelines
