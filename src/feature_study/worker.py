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

import hashlib
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.arms import Arm
from src.feature_study.config import (
    CLIP_RANGE, CLUSTER_ID, FAST_MODELS, MODEL_SEED, REFERENCE_MODELS,
)
from src.feature_study.groups_source import REFERENCE_COLUMN
from src.feature_study.losses import make_regressors
from src.pipelines.common import (
    ERA5_GUST_PROXY, RANDOM_STATE, TARGET_VAR, compute_metrics, preprocess_df,
)

# Arms cuja predição por estação é persistida. `base` e `full` são as âncoras
# de toda comparação e os dois únicos que interessa plotar; gravar todos
# encheria o volume com tabelas que ninguém abre.
PRED_DUMP_ARMS = ("base", "full")

METRIC_COLS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]


def select_regressors(models="all", loss: str = "") -> list:
    """`all` = os 39 do LazyPredict (mesmo filtro de `cluster_lazy`), `reference`
    = os 7 fixados a priori, `fast3` = os 3 do piloto. `models` também aceita
    uma LISTA de nomes — é como a triagem top-5 chega até aqui.

    Com `loss`, o conjunto passa a ser o de `LOSS_MODELS` (os únicos que aceitam
    trocar a perda), ainda filtrado por `models`. Um modelo pedido por `models`
    que não exista nesse conjunto NÃO é erro aqui: o eixo de perda é
    intrinsecamente definido sobre um subconjunto, e `compute_effects` já
    restringe a comparação aos modelos presentes dos dois lados.

    Um nome desconhecido LEVANTA, em vez de ser ignorado: um erro de digitação
    devolveria lista vazia e o `LazyRegressor` treinaria zero modelos sem
    reclamar, produzindo um parquet de métricas vazio que só apareceria como
    resultado faltando na análise, horas depois.
    """
    from src.pipelines.cluster_lazy import _FAST_REGRESSORS

    pool = make_regressors(loss) if loss else list(_FAST_REGRESSORS)
    if isinstance(models, str):
        if models == "all":
            return pool
        names = {"reference": REFERENCE_MODELS, "fast3": FAST_MODELS}.get(models)
        if names is None:
            raise ValueError(f"models inválido: {models!r} (all | reference | fast3 | lista de nomes)")
    else:
        names = tuple(models)
        if not names:
            raise ValueError("lista de modelos vazia")
    chosen = [c for c in pool if c.__name__ in set(names)]
    missing = set(names) - {c.__name__ for c in chosen}
    if missing and not loss:
        raise ValueError(f"modelos ausentes do LazyPredict: {sorted(missing)}")
    return chosen


def _models_fingerprint(regressors) -> str:
    """Impressão digital de um CONJUNTO de modelos, para o `models_mode`.

    Sem isso, dois conjuntos custom diferentes — o top-5 do DJF e o do JJA, ou
    um top-5 recalculado depois de mudar `k` — carimbam ambos a string literal
    `"custom"`, e `_is_current` os considera equivalentes: o `skip_existing`
    devolve o ajuste ANTIGO como se fosse o novo, em silêncio.
    """
    nomes = sorted(c if isinstance(c, str) else c.__name__ for c in regressors)
    return hashlib.md5(",".join(nomes).encode()).hexdigest()[:12]


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


def _features_fingerprint(features) -> str:
    """Hash estável (entre processos/containers) da lista de features de um
    arm. O NOME do arm (`full`, `drop__nf_x`...) não muda quando uma nova
    variável entra no pool de `new_features` — mas a COMPOSIÇÃO de `full` e de
    tudo que referencia `full` (todo `drop__*`, `add_grp__dynamic`,
    `drop_grp__*`) muda. Sem este fingerprint, `_is_current` reaproveitaria em
    silêncio um ajuste feito com o conjunto de features ANTIGO."""
    chave = ",".join(sorted(features))
    return hashlib.md5(chave.encode()).hexdigest()[:12]


def _is_current(path: Path, models_mode: str, seed: int = MODEL_SEED,
                loss: str = "mse", features_fp: str | None = None) -> bool:
    """Arquivo existente só vale se foi gerado com o MESMO conjunto de
    modelos, a MESMA seed, a MESMA perda e a MESMA lista de features do arm.

    Sem o primeiro trio, o piloto (3 modelos) e o run completo (39)
    compartilhando a pasta `units/full/` fariam o `fit` idempotente pular os
    arms do piloto e entregar métricas incompletas sem nenhum aviso. Arquivo
    antigo, sem a coluna, ou ilegível ⇒ refaz.
    """
    if not path.exists():
        return False
    try:
        cols = pd.read_parquet(path, columns=["models_mode", "model_seed"])
    except Exception:
        return False
    if not ((cols["models_mode"] == models_mode).all() and (cols["model_seed"] == seed).all()):
        return False
    # `loss` e `features_fp` são colunas mais NOVAS que `models_mode`/`seed`.
    # Cada uma é lida à parte, de propósito: juntá-las à leitura acima faria
    # todo parquet gravado antes delas existir levantar, devolver False e ser
    # refeito em silêncio — o oposto do que a idempotência promete.
    try:
        gravado = pd.read_parquet(path, columns=["loss"])["loss"]
    except Exception:
        # Ausência aqui TEM significado: todo arquivo anterior ao eixo de
        # perda é, por construção, MSE.
        if loss != "mse":
            return False
    else:
        if not (gravado == loss).all():
            return False
    if features_fp is None:
        return True
    try:
        gravado_fp = pd.read_parquet(path, columns=["features_fp"])["features_fp"]
    except Exception:
        # Ausência aqui NÃO tem significado seguro — ao contrário de `loss`,
        # não há um valor "óbvio" para arquivos gravados antes desta coluna
        # existir (foi exatamente uma mudança de composição de features, sem
        # aviso, que motivou criá-la). Refaz por segurança.
        return False
    return bool((gravado_fp == features_fp).all())


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


def fit_arm(train, val, test, features, regressors, seed: int = MODEL_SEED,
            resid_models: frozenset[str] | None = None, fitted_out=None, preds_out=None):
    """Ajusta os regressores em UM arm. Retorna (métricas, resíduos, segundos).

    `resid_models` são os modelos que ganham resíduo POR LINHA no teste — o
    insumo de `compute_effects` e de tudo que se plota depois. O default são os
    7 `REFERENCE_MODELS`; no modo top-5 o chamador passa os 5 escolhidos, senão
    um campeão fora daquela tupla (GradientBoosting, Bagging, AdaBoost são
    candidatos reais) ficaria sem resíduo e o bootstrap cego.

    `fitted_out` recebe `{nome: pipeline ajustado}` — é como o chamador
    persiste o campeão em joblib sem reajustar nada. `preds_out` recebe a
    predição de teste já truncada em [0, 80], para a tabela por estação.
    Ambos são parâmetros de SAÍDA em vez de valores de retorno para não mudar
    a aridade de `fit_arm`, de que os testes e o `run_unit` já dependem.
    """
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

    guardar = frozenset(REFERENCE_MODELS) if resid_models is None else frozenset(resid_models)
    rows, resid, preds = [], {}, {}

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
        if record(name, "test", y_te, pred) and name in guardar:
            clipped = np.clip(pred, *CLIP_RANGE)
            resid[name] = (clipped - y_te).astype("float32")
            preds[name] = clipped.astype("float32")
            if fitted_out is not None:
                fitted_out[name] = pipe
    if preds_out is not None:
        preds_out.update(preds)

    resid_df = pd.DataFrame(resid)
    resid_df.insert(0, "row_id", test["row_id"].to_numpy())
    return pd.DataFrame(rows), resid_df, time.time() - t0


def _station_predictions(test, preds: dict, arm: str, season: str, tag: str, seed: int):
    """Predição de teste por (estação, dia, modelo), com lat/lon e o proxy
    ERA5 ao lado — é o que fecha o "MODELO vs ERA5" do mapa espacial e a série
    temporal de melhor/médio/pior caso, sem precisar reconstruir predição a
    partir do resíduo.

    Reusa `build_station_predictions_frame` (`src/pipelines/metrics_schema.py`)
    para sair no MESMO schema que lazy/mlp/lstm já gravam, e assim o
    `best_model_selector` e o dashboard leem esta tabela sem caso especial.
    """
    from src.pipelines.metrics_schema import build_station_predictions_frame

    def coluna(nome):
        # A presença de lat/lon é garantida uma vez, no `prepare` (que levanta
        # se faltarem). Aqui só degradamos para NaN, porque um erro dentro do
        # fan-out de centenas de unidades seria ruído, não sinal.
        return test[nome].to_numpy(float) if nome in test.columns else np.full(len(test), np.nan)

    # a rajada ERA5 da estrutura nova, se houver; senão o `wind_mag_max` antigo
    # (que é o VENTO médio, não uma rajada — ver `prepare.REFERENCE_CANDIDATES`)
    ref = next((c for c in (REFERENCE_COLUMN, ERA5_GUST_PROXY) if c in test.columns), ERA5_GUST_PROXY)
    proxy = coluna(ref)
    partes = []
    for nome, pred in preds.items():
        frame = build_station_predictions_frame(
            estacao=test["estacao"].to_numpy(),
            latitude=coluna("latitude"), longitude=coluna("longitude"),
            cluster_id=CLUSTER_ID, split="test",
            y_true=test[TARGET_VAR].to_numpy(float), y_pred=pred,
            time=test["time"].to_numpy(),
            extra_cols={"era5_proxy": proxy, "model": nome, "arm": arm,
                        "season": season, "tag": tag, "model_seed": seed},
        )
        partes.append(frame)
    return pd.concat(partes, ignore_index=True)


def run_unit(
    data_dir, out_dir, sample_tag: str, season: str, arms: list[Arm],
    models="all", seed: int = MODEL_SEED, skip_existing: bool = True,
    regressors: list | None = None, out_tag: str | None = None,
    pred_arms=PRED_DUMP_ARMS, champion: str = "", champion_arm: str = "base",
) -> list[Path]:
    """Ajusta todos os `arms` de um trimestre. Idempotente por arm.

    `out_tag` grava em outra pasta a MESMA amostra — é como o piloto repete uma
    unidade para confirmar o determinismo.

    Os regressores são escolhidos POR ARM, não uma vez por unidade: um arm do
    eixo de perda carrega a própria perda (`arm.loss`) e precisa das subclasses
    correspondentes. `regressors` explícito continua vencendo — é o atalho dos
    testes.

    Com um conjunto CUSTOM de modelos (a triagem top-5, ou `regressors`
    explícito), o `models_mode` leva a impressão digital do conjunto. Sem ela,
    dois top-5 diferentes carimbariam ambos `"custom"` e o `skip_existing`
    devolveria o ajuste errado sem avisar.

    `pred_arms` são os arms cuja predição de teste por estação é persistida —
    é o insumo da série temporal e do mapa espacial. Só esses, para não encher
    o volume com uma tabela por arm.
    """
    out = Path(out_dir) / "units" / (out_tag or sample_tag)
    out.mkdir(parents=True, exist_ok=True)
    custom = regressors is not None or not isinstance(models, str)
    models_mode = models if not custom else f"custom:{_models_fingerprint(regressors or models)}"
    train, val, test = load_unit_frames(Path(data_dir), sample_tag, season)
    if train.empty or val.empty or test.empty:
        raise ValueError(f"amostra {sample_tag}/{season} sem treino, validação ou teste")

    written = []
    for arm in arms:
        regs = regressors if regressors is not None else select_regressors(models, arm.loss)
        arm_loss = arm.loss or "mse"
        feat_fp = _features_fingerprint(arm.features)
        m_path = out / f"metrics__{season}__{arm.name}.parquet"
        r_path = out / f"resid__{season}__{arm.name}.parquet"
        if skip_existing and r_path.exists() and _is_current(m_path, models_mode, seed, arm_loss, feat_fp):
            print(f"[fit] {sample_tag}/{season}/{arm.name}: já existe "
                  f"({models_mode}, {arm_loss}) — pulando", flush=True)
            written += [m_path, r_path]
            continue
        print(f"[fit] {sample_tag}/{season}/{arm.name}: {len(arm.features)} features, "
              f"{len(train)} treino, {len(val)} val, {len(test)} teste, "
              f"{len(regs)} modelos, perda {arm_loss}", flush=True)
        # Os modelos que ganham resíduo são os EFETIVAMENTE ajustados neste
        # arm, não a tupla fixa de referência: no modo top-5 o campeão pode não
        # estar entre os 7, e sem resíduo `compute_effects` fica cego.
        guardar = frozenset(c.__name__ for c in regs) if custom else None
        preds_out: dict = {}
        # O campeão só é persistido no arm âncora e na seed base: são 4
        # arquivos (um por trimestre), e com ~680 colunas um ExtraTrees passa
        # de 100 MB. Guardar todos encheria o volume sem serventia.
        quer_modelo = bool(champion) and arm.name == champion_arm and seed == MODEL_SEED
        fitted_out: dict = {} if quer_modelo else None
        metrics, resid, secs = fit_arm(
            train, val, test, arm.features, regs, seed,
            resid_models=guardar, preds_out=preds_out, fitted_out=fitted_out,
        )
        if quer_modelo and champion in (fitted_out or {}):
            from src.feature_study.artifacts import champion_artifact, dump_champion_pipeline

            r2 = metrics.loc[(metrics["model"] == champion) & (metrics["split"] == "val"), "R2"]
            dump_champion_pipeline(champion_artifact(
                model=fitted_out[champion], model_name=champion, features=arm.features,
                x_train_raw=train[list(arm.features)], cluster_id=CLUSTER_ID, season=season,
                r2=float(r2.iloc[0]) if len(r2) else None, study_tag=out_tag or sample_tag, seed=seed,
            ), out_dir)
        metrics.insert(0, "arm", arm.name)
        metrics.insert(1, "season", season)
        metrics.insert(2, "tag", sample_tag)
        metrics["fit_seconds"] = round(secs, 2)
        metrics["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        metrics["model_seed"] = seed
        metrics["models_mode"] = models_mode
        metrics["loss"] = arm_loss
        metrics["features_fp"] = feat_fp
        # Consultável na análise, mas deliberadamente FORA de `_is_current`:
        # `models_mode` já carrega a mesma informação e acrescentar outra
        # coluna à leitura em etapas é mais uma chance de quebrar o volume.
        metrics["models_fp"] = _models_fingerprint(regs)
        metrics.to_parquet(m_path, index=False)
        resid.to_parquet(r_path, index=False)
        # `written[0]` é sempre o parquet de MÉTRICAS: os chamadores (e os
        # testes) indexam essa lista, então a tabela de predições entra depois.
        written += [m_path, r_path]
        if preds_out and arm.name in set(pred_arms):
            p_path = out / f"preds__{season}__{arm.name}.parquet"
            _station_predictions(
                test, preds_out, arm.name, season, sample_tag, seed,
            ).to_parquet(p_path, index=False)
            written.append(p_path)
        print(f"[fit]   pronto em {secs:.0f}s", flush=True)
    return written


assert MODEL_SEED == RANDOM_STATE  # seeds fixas: o modelo usa a mesma seed das pipelines
