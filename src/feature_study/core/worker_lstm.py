"""Estágio `fit` com a LSTM: os MESMOS arms, linhas e saídas do worker tabular.

O LazyPredict responde "quais variáveis acrescentam informação a um modelo de
árvore/linear que vê UM instante". Esta etapa repete a pergunta com um modelo
sequencial, que vê as últimas `T` horas até a hora do pico, para checar se as
conclusões (quais grupos importam, quais não) valem fora das árvores.

Contrato igual ao de `worker.run_unit`, para o `aggregate` (efeitos pareados,
bootstrap em blocos, controles negativos) rodar sem nenhuma mudança:

  units/<tag>/metrics__<season>__<arm>.parquet   R2/RMSE/Bias/Bias_P90/RMSE_P90, val e teste
  units/<tag>/resid__<season>__<arm>.parquet     resíduo (pred − y) por linha do teste

Diferenças deliberadas:
  - um único modelo, `LSTM` (LSTM → Dropout → Dense(1), Huber, o mesmo da pipeline
    `cluster_lstm`), no lugar dos 39/5 regressores;
  - a saída vai para `units/<tag>_lstm/`, nunca para a pasta dos modelos tabulares;
  - uma linha cuja janela tem hora faltando NÃO entra (nem no treino, nem na
    avaliação): nada é imputado. O critério é a janela completa para a UNIÃO das
    colunas de TODOS os arms do estudo, não para as colunas de cada arm: assim todos
    os arms treinam e são avaliados nas MESMAS linhas, e o `compute_effects` (que
    descarta por segurança qualquer par de arms com linhas diferentes) mede todas as
    comparações. Com o critério por arm, os arms com o grupo 2 ficavam com outro
    conjunto de linhas que o `base`, e `full_vs_base` e os efeitos do grupo 2 sumiam.

Entrada de cada arm: as colunas dos grupos 1–3 viram sequência (uma leitura por
hora); as demais (relevo, latitude/longitude, ruído e estáticas permutadas dos
controles) são constantes na linha e se repetem a cada passo.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.core.arms import Arm
from src.feature_study.core.config import CLIP_RANGE, MODEL_SEED
from src.feature_study.core.worker import (
    NAN_METRICS, PRED_DUMP_ARMS, _features_fingerprint, _is_current, _safe_metrics,
    _station_predictions, load_unit_frames,
)
from src.feature_study.data import groups_source as gs
from src.feature_study.data import sequences as sq
from src.pipelines.common import TARGET_VAR

LSTM_NAME = "LSTM"
DEFAULTS = {
    "window_hours": 24, "units": 96, "dropout": 0.3, "huber_delta": 1.0,
    "learning_rate": 1e-3, "epochs": 50, "batch_size": 64, "patience": 5,
}


def models_mode(**hp) -> str:
    """Carimbo do ajuste: muda se qualquer hiperparâmetro mudar, para o `skip_existing`
    nunca devolver um ajuste feito com outra configuração."""
    p = {**DEFAULTS, **hp}
    return (f"lstm:T{p['window_hours']}:u{p['units']}:d{p['dropout']}:lr{p['learning_rate']}"
            f":e{p['epochs']}:b{p['batch_size']}:p{p['patience']}:linhas-comuns")


def split_features(features, seq_columns) -> tuple[list[str], list[str]]:
    """(sequência, constantes): as colunas horárias dos grupos 1–3 e o resto, na ordem do arm."""
    seq_set = set(seq_columns)
    feats = list(features)
    return [c for c in feats if c in seq_set], [c for c in feats if c not in seq_set]


def stack_inputs(seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    """(n, T, Fseq) + (n, Fstatic) → (n, T, Fseq+Fstatic), repetindo a constante em cada passo."""
    if static.shape[1] == 0:
        return seq
    rep = np.repeat(static[:, None, :].astype("float32"), seq.shape[1], axis=1)
    return np.concatenate([seq, rep], axis=2)


def valid_rows(x: np.ndarray) -> np.ndarray:
    """Janelas sem nenhuma lacuna. Nada é preenchido: linha com NaN fica de fora."""
    return ~np.isnan(x).reshape(len(x), -1).any(axis=1)


def _scale(x_tr, *others):
    """RobustScaler por feature, ajustado só no treino, sobre todos os passos juntos."""
    from sklearn.preprocessing import RobustScaler

    f = x_tr.shape[2]
    sc = RobustScaler().fit(x_tr.reshape(-1, f))
    out = [sc.transform(a.reshape(-1, f)).reshape(a.shape).astype("float32") for a in (x_tr, *others)]
    return sc, out


def fit_arm_lstm(x_tr, y_tr, x_va, y_va, x_te, y_te, test_row_ids, *, seed: int = MODEL_SEED,
                 model_out: dict | None = None, **hp):
    """Ajusta a LSTM em UM arm. Retorna `(métricas, resíduos, segundos)`, no schema do worker
    tabular. `x_*` são `(n, T, F)` já empilhados; linhas com lacuna são descartadas aqui.

    `model_out` recebe `{"model", "scaler", "y_mean", "y_std", "test_valid"}` — é como o chamador
    reaproveita o modelo ajustado (importância por permutação) sem reajustar.
    """
    import tensorflow as tf

    from src.models.cluster_lstm_builder import ClusterLSTMRegressorBuilder

    p = {**DEFAULTS, **hp}
    t0 = time.time()
    m_tr, m_va, m_te = valid_rows(x_tr), valid_rows(x_va), valid_rows(x_te)
    if m_tr.sum() == 0 or m_va.sum() == 0 or m_te.sum() == 0:
        raise ValueError("sem linhas utilizáveis (janelas completas) em treino, validação ou teste")
    x_tr, y_tr = x_tr[m_tr], np.asarray(y_tr, float)[m_tr]
    x_va, y_va = x_va[m_va], np.asarray(y_va, float)[m_va]
    x_te, y_te = x_te[m_te], np.asarray(y_te, float)[m_te]
    ids = np.asarray(test_row_ids)[m_te]

    sc, (x_tr_s, x_va_s, x_te_s) = _scale(x_tr, x_va, x_te)
    mu, sd = float(y_tr.mean()), float(y_tr.std()) or 1.0

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    model = ClusterLSTMRegressorBuilder(
        units=p["units"], dropout=p["dropout"], huber_delta=p["huber_delta"],
        learning_rate=p["learning_rate"],
    ).build(x_tr_s.shape[2], x_tr_s.shape[1])
    model.fit(
        x_tr_s, ((y_tr - mu) / sd).astype("float32"),
        validation_data=(x_va_s, ((y_va - mu) / sd).astype("float32")),
        epochs=p["epochs"], batch_size=p["batch_size"], shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=p["patience"], restore_best_weights=True)],
    )

    def predict(x):
        return np.asarray(model.predict(x, batch_size=512, verbose=0), float).ravel() * sd + mu

    rows, resid = [], {}
    for split, x, y in (("val", x_va_s, y_va), ("test", x_te_s, y_te)):
        pred = predict(x)
        m, bad = _safe_metrics(y, pred)
        if m is None:
            print(f"   [AVISO] LSTM gerou {bad} predição(ões) não finita(s) em {split}", flush=True)
        rows.append({"model": LSTM_NAME, "split": split, "n": len(y), **(m or NAN_METRICS),
                     "n_nonfinite": bad, "lazypredict_R2_val": np.nan})
        if split == "test" and m is not None:
            resid[LSTM_NAME] = (np.clip(pred, *CLIP_RANGE) - y).astype("float32")
            if model_out is not None:
                model_out["pred_test"] = np.clip(pred, *CLIP_RANGE).astype("float32")
    if model_out is not None:
        model_out.update(model=model, scaler=sc, y_mean=mu, y_std=sd, test_valid=m_te, predict=predict,
                         x_test_scaled=x_te_s, y_test=y_te)

    resid_df = pd.DataFrame(resid)
    resid_df.insert(0, "row_id", ids)
    return pd.DataFrame(rows), resid_df, time.time() - t0


def permutation_importance(predict, x, y, names, group_of, *, n_repeats: int = 3, seed: int = MODEL_SEED) -> pd.DataFrame:
    """Quanto o erro PIORA ao embaralhar cada coluna e cada grupo, na LSTM já ajustada.

    `x` é `(n, T, F)` já escalado, `names` os nomes das F colunas na ordem da entrada. Embaralha-se
    a coluna ENTRE LINHAS, com a mesma permutação em todos os passos da janela: embaralhar passo a
    passo destruiria a sequência e mediria outra coisa. Uma linha por (bloco, repetição), com o ΔRMSE
    global e o dos dias acima do P90 do observado.

    É uso do modelo, não ganho fora da amostra (a mesma ressalva da importância por permutação das
    árvores): serve para comparar COMO a LSTM e as árvores distribuem o peso entre os grupos.
    """
    from src.feature_study.diagnostics.shap_groups import group_of_column

    x = np.array(x, copy=True)
    y = np.asarray(y, float)
    rng = np.random.default_rng(int(seed))
    alto = y >= np.quantile(y, 0.90)

    def rmse(pred):
        e = pred - y
        return float(np.sqrt(np.mean(e ** 2))), float(np.sqrt(np.mean(e[alto] ** 2)))

    base, base90 = rmse(predict(x))
    por_grupo: dict[str, list[int]] = {}
    for i, c in enumerate(names):
        por_grupo.setdefault(group_of_column(c, group_of), []).append(i)
    blocos = [("coluna", c, group_of_column(c, group_of), [i]) for i, c in enumerate(names)]
    blocos += [("grupo", g, g, cols) for g, cols in por_grupo.items()]

    linhas = []
    for nivel, nome, grupo, cols in blocos:
        guardado = x[:, :, cols].copy()
        for r in range(n_repeats):
            x[:, :, cols] = guardado[rng.permutation(len(x))]
            r_all, r_90 = rmse(predict(x))
            linhas.append({"nivel": nivel, "nome": nome, "grupo": grupo, "n_colunas": len(cols), "repeticao": r,
                           "d_rmse": r_all - base, "d_rmse_p90": r_90 - base90})
        x[:, :, cols] = guardado
    return pd.DataFrame(linhas)


def run_unit_lstm(
    data_dir, out_dir, raw_dir, sample_tag: str, season: str, arms: list[Arm], *,
    out_tag: str | None = None, seed: int = MODEL_SEED, skip_existing: bool = True,
    pred_arms=PRED_DUMP_ARMS, peak_hours: pd.DataFrame | None = None,
    importance_arms=("full",), importance_repeats: int = 3, **hp,
) -> list[Path]:
    """Ajusta a LSTM em todos os `arms` de um trimestre. Idempotente por arm.

    Lê as janelas UMA vez para a união das colunas de sequência de todos os arms e fatia por arm,
    em vez de reler os parquets a cada arm.
    """
    p = {**DEFAULTS, **hp}
    window = int(p["window_hours"])
    tag_out = out_tag or f"{sample_tag}_lstm"
    out = Path(out_dir) / "units" / tag_out
    out.mkdir(parents=True, exist_ok=True)
    mode = models_mode(**p)

    train, val, test = load_unit_frames(Path(data_dir), sample_tag, season)
    if train.empty or val.empty or test.empty:
        raise ValueError(f"amostra {sample_tag}/{season} sem treino, validação ou teste")

    pendentes = []
    for arm in arms:
        m_path = out / f"metrics__{season}__{arm.name}.parquet"
        r_path = out / f"resid__{season}__{arm.name}.parquet"
        i_path = out / f"importance__{season}__{arm.name}.parquet"
        quer_imp = arm.name in set(importance_arms)
        if (skip_existing and r_path.exists() and (i_path.exists() or not quer_imp)
                and _is_current(m_path, mode, seed, _features_fingerprint(arm.features))):
            print(f"[fit-lstm] {tag_out}/{season}/{arm.name}: já existe ({mode}) — pulando", flush=True)
            continue
        pendentes.append(arm)

    written = []
    for arm in arms:
        if arm not in pendentes:
            written += [out / f"metrics__{season}__{arm.name}.parquet", out / f"resid__{season}__{arm.name}.parquet"]
    meta_path = Path(data_dir) / "meta.json"
    group_of = json.loads(meta_path.read_text()).get("group_of", {}) if meta_path.exists() else {}
    if not pendentes:
        return written

    from src.feature_study.core.analysis import load_arms

    seq_all = sorted({c for g in gs.HOURLY_GROUPS for c in gs.hourly_columns(raw_dir, g)})
    split_cols = {a.name: split_features(a.features, seq_all) for a in pendentes}
    # união de TODOS os arms do estudo (não só dos pendentes): o conjunto de linhas tem de ser o
    # mesmo mesmo que a unidade seja repetida depois só com alguns arms
    todos = load_arms(data_dir)
    uniao = sorted({c for a in (*todos, *pendentes) for c in split_features(a.features, seq_all)[0]})
    cache_dir = Path(data_dir).parent / "_cache"
    peaks = peak_hours if peak_hours is not None else sq.load_peak_hours(cache_dir)
    estacoes = sorted(set(train["estacao"].astype(str)) | set(test["estacao"].astype(str)))
    print(f"[fit-lstm] {season}: janelas de {window} h × {len(uniao)} colunas de sequência "
          f"({len(train)}+{len(val)}+{len(test)} linhas)", flush=True)
    w = {k: sq.load_windows(raw_dir, df, uniao, window, peaks, stations=estacoes)
         for k, df in (("train", train), ("val", val), ("test", test))}
    ok = {k: valid_rows(v) for k, v in w.items()}
    n_fora = {k: int((~m).sum()) for k, m in ok.items()}
    print(f"[fit-lstm] {season}: linhas sem janela completa, fora de TODOS os arms: {n_fora}", flush=True)
    train, val, test = (df[ok[k]].reset_index(drop=True) for k, df in (("train", train), ("val", val), ("test", test)))
    w = {k: v[ok[k]] for k, v in w.items()}
    pos = {c: i for i, c in enumerate(uniao)}

    for arm in pendentes:
        seq, static = split_cols[arm.name]
        idx = [pos[c] for c in seq]

        def inputs(df, key):
            return stack_inputs(w[key][:, :, idx], df[static].to_numpy("float32") if static else np.empty((len(df), 0), "float32"))

        x_tr, x_va, x_te = inputs(train, "train"), inputs(val, "val"), inputs(test, "test")
        print(f"[fit-lstm] {tag_out}/{season}/{arm.name}: {len(seq)} sequência + {len(static)} constantes", flush=True)
        keep: dict = {}
        metrics, resid, secs = fit_arm_lstm(
            x_tr, train[TARGET_VAR].to_numpy(), x_va, val[TARGET_VAR].to_numpy(), x_te,
            test[TARGET_VAR].to_numpy(), test["row_id"].to_numpy(), seed=seed, model_out=keep, **p,
        )
        metrics.insert(0, "arm", arm.name)
        metrics.insert(1, "season", season)
        metrics.insert(2, "tag", sample_tag)
        metrics["fit_seconds"] = round(secs, 2)
        metrics["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        metrics["model_seed"] = seed
        metrics["models_mode"] = mode
        metrics["features_fp"] = _features_fingerprint(arm.features)
        metrics["models_fp"] = mode
        metrics["n_rows_dropped_test"] = n_fora["test"]
        m_path = out / f"metrics__{season}__{arm.name}.parquet"
        r_path = out / f"resid__{season}__{arm.name}.parquet"
        metrics.to_parquet(m_path, index=False)
        resid.to_parquet(r_path, index=False)
        written += [m_path, r_path]
        if arm.name in set(pred_arms) and "pred_test" in keep:
            p_path = out / f"preds__{season}__{arm.name}.parquet"
            _station_predictions(test, {LSTM_NAME: keep["pred_test"]}, arm.name, season, sample_tag, seed
                                 ).to_parquet(p_path, index=False)
            written.append(p_path)
        if arm.name in set(importance_arms):
            nomes = seq + static
            imp = permutation_importance(keep["predict"], keep["x_test_scaled"], keep["y_test"], nomes, group_of,
                                         n_repeats=importance_repeats, seed=seed)
            imp.insert(0, "arm", arm.name)
            imp.insert(1, "season", season)
            i_path = out / f"importance__{season}__{arm.name}.parquet"
            imp.to_parquet(i_path, index=False)
            written.append(i_path)
        print(f"[fit-lstm]   pronto em {secs:.0f}s", flush=True)
    return written
