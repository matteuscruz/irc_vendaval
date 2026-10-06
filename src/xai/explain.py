"""Explicabilidade (XAI) sobre modelos já treinados: SHAP e importância por
ablação.

Roda **fora da pipeline de treino**, sobre artefatos salvos. Não treina, não
altera dado e não reintroduz imputação no fluxo de produção: a única
"imputação" aqui é o instrumento de medida da ablação (§2), aplicada a uma
cópia da matriz de avaliação.

O que produz, por (cluster, trimestre):

1. **SHAP** — contribuição de cada feature para cada predição.
   `TreeExplainer` quando o estimador final é baseado em árvores (exato e
   rápido); caso contrário, `PermutationExplainer` sobre uma amostra.

2. **Importância por ablação ("feature imputation")** — substitui UMA feature
   de cada vez e mede quanto a métrica piora:
     - `mean`    : troca pela média do conjunto de referência. Responde
                   "quanto se perde se esta feature não estiver disponível?",
                   que é a pergunta operacional do projeto — na grade há
                   células fora da cobertura de algumas fontes.
     - `shuffle` : embaralha a coluna (importância por permutação). Quebra a
                   relação feature↔alvo preservando a distribuição marginal.
   Reporta ΔRMSE e ΔBias@P90 — esta última é a métrica de decisão do projeto,
   e uma feature pode ser irrelevante no erro global e decisiva na cauda.

A reconstrução da matriz de avaliação usa a partição e a climatologia gravadas
no PRÓPRIO artefato (`split`, `climatology`), não os padrões do código: um
modelo antigo precisa ser explicado com o dado que ele viu, não com o dado que
a pipeline montaria hoje.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_harmonic_climatology
from src.data.netcdf_loader import NetCDFLoader
from src.pipeline.data.splits import MonthBlockSplit
from src.pipelines.common import (
    ERA5_GUST_PROXY,
    TARGET_VAR,
    SPLIT_COL,
    assign_split_labels,
    build_flat_dataframe,
    compute_metrics,
    reject_inmet_derived_features,
    select_complete_rows,
)

ABLATION_STRATEGIES = ("mean", "shuffle")


@dataclass
class EvalMatrix:
    """Matriz de avaliação de um (cluster, trimestre), já alinhada ao artefato."""
    x: pd.DataFrame          # features CRUAS, na ordem de art["features"]
    y: np.ndarray            # rajada observada (m/s)
    era5: np.ndarray         # proxy ERA5 do dia (reconstrução de alvo em razão)
    meta: pd.DataFrame       # estacao, time


# ── Artefato ────────────────────────────────────────────────────────────────

def load_artifact(models_dir: str | Path, cluster_id: int, season: str | None = None) -> dict:
    """Artefato do cluster (por trimestre se existir, senão o agrupado)."""
    models_dir = Path(models_dir)
    names = ([f"best_model_c{cluster_id}_{season}.joblib"] if season else [])
    names.append(f"best_model_c{cluster_id}.joblib")
    for name in names:
        path = models_dir / name
        if path.exists():
            import joblib

            art = joblib.load(path)
            reject_inmet_derived_features(art["features"], source=name)
            return art
    raise FileNotFoundError(f"nenhum artefato para cluster {cluster_id}/{season} em {models_dir}")


def _estimator_and_transform(art: dict):
    """Separa o estimador final do pré-processamento do artefato.

    O campeão do LazyPredict é um `Pipeline` (ColumnTransformer + regressor);
    o do mlp é o estimador direto. Retorna `(estimator, to_model_space)`, onde
    `to_model_space` leva a matriz CRUA ao espaço que o estimador recebe ---
    incluindo o `RobustScaler` do artefato, que é como a inferência do projeto
    alimenta o modelo.
    """
    scaler = art["scaler"]
    model = art["model"]
    feats = list(art["features"])

    if hasattr(model, "steps"):
        pre, estimator = model[:-1], model[-1]

        def to_model_space(x_raw: pd.DataFrame) -> np.ndarray:
            scaled = pd.DataFrame(scaler.transform(x_raw[feats]), columns=feats)
            return np.asarray(pre.transform(scaled))
    else:
        estimator = model

        def to_model_space(x_raw: pd.DataFrame) -> np.ndarray:
            return np.asarray(scaler.transform(x_raw[feats]))

    return estimator, to_model_space


def predict(art: dict, x_raw: pd.DataFrame, era5: np.ndarray) -> np.ndarray:
    """Predição em m/s, com a mesma reconstrução de alvo da inferência."""
    estimator, to_model_space = _estimator_and_transform(art)
    raw = np.asarray(estimator.predict(to_model_space(x_raw)), dtype=float).ravel()
    if art.get("target_kind") == "ratio":
        raw = raw * np.clip(era5, 0.1, None)
    return np.clip(raw, 0.0, 80.0)


# ── Matriz de avaliação ─────────────────────────────────────────────────────

def build_eval_matrix(
    raw_dir: str, shp_dir: str, art: dict, split_label: str = "test",
) -> EvalMatrix:
    """Reconstrói as features do (cluster, trimestre) do artefato, no recorte
    pedido, usando a partição e a climatologia gravadas nele."""
    if not art.get("split"):
        raise ValueError(
            "artefato sem 'split' — não é possível reconstruir o dado que ele viu; "
            "retreine com a pipeline atual"
        )
    split = MonthBlockSplit.from_dict(art["split"])
    n_harm = int(art.get("climatology", {}).get("n_harmonics", 3))

    ds_inmet, ds_era5 = NetCDFLoader(raw_dir).load_extended()
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)
    times = pd.DatetimeIndex(ds_era5["time"].values)
    labels = split.label(times)
    clim = get_harmonic_climatology(
        ds_era5, ERA5_GUST_PROXY, times[labels == "train"], n_harm,
    ).reset_coords(drop=True)

    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, clim)
    df, _ = assign_split_labels(df, split)

    faltando = [f for f in art["features"] if f not in df.columns]
    if faltando:
        raise ValueError(
            f"{len(faltando)} feature(s) do artefato ausente(s) no dado reconstruído: "
            f"{faltando[:8]}{'...' if len(faltando) > 8 else ''}. "
            "As `nf_*` vêm de dataset/raw/new_features — confira se a região está "
            "disponível neste ambiente."
        )
    df = select_complete_rows(df, list(art["features"]), label="xai")

    df = df[df["cluster_id"].astype(str) == str(art["cluster_id"])]
    df = df[df[SPLIT_COL] == split_label]
    season = art.get("season")
    if season:
        from src.pipelines.common import month_to_season

        df = df[month_to_season(df["time"].dt.month) == season]
    if df.empty:
        raise ValueError(
            f"nenhuma linha para cluster {art['cluster_id']}/{season} no split "
            f"{split_label!r} — confira o artefato e o período disponível"
        )

    df = df.reset_index(drop=True)
    return EvalMatrix(
        x=df[list(art["features"])].reset_index(drop=True),
        y=df[TARGET_VAR].to_numpy(float),
        era5=df[ERA5_GUST_PROXY].to_numpy(float),
        meta=df[["estacao", "time"]].reset_index(drop=True),
    )


# ── SHAP ────────────────────────────────────────────────────────────────────

def shap_values(
    art: dict, data: EvalMatrix, max_samples: int = 2000, seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Valores SHAP por linha × feature. Retorna `(df, explainer_usado)`.

    Amostra `max_samples` linhas quando o conjunto é maior: o SHAP exato de
    árvore é linear no nº de linhas, e 2000 já estabiliza a média |SHAP|.
    """
    import shap

    estimator, to_model_space = _estimator_and_transform(art)
    feats = list(art["features"])

    x = data.x
    if len(x) > max_samples:
        x = x.sample(max_samples, random_state=seed).sort_index()
    x_model = to_model_space(x)

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(x_model, check_additivity=False)
        # `expected_value` do CatBoost só fica definido DEPOIS desta chamada
        # (vem da última coluna do ShapValues nativo); lido antes, vale ~0 e a
        # verificação de aditividade falharia por um viés inteiro.
        _assert_additive(explainer, estimator, x_model, values)
        kind = f"TreeExplainer({type(estimator).__name__})"
    except Exception as exc:  # modelo não-árvore: cai no agnóstico
        print(f"[xai] TreeExplainer indisponível ({type(exc).__name__}) — usando PermutationExplainer.")
        background = shap.sample(x_model, min(100, len(x_model)), random_state=seed)
        explainer = shap.PermutationExplainer(estimator.predict, background, seed=seed)
        values = explainer(x_model[: min(300, len(x_model))]).values
        x = x.iloc[: values.shape[0]]
        kind = f"PermutationExplainer({type(estimator).__name__})"

    values = np.asarray(values)
    if values.ndim == 3:  # (n, f, outputs)
        values = values[..., 0]
    return pd.DataFrame(values, columns=feats, index=x.index), kind


def _assert_additive(explainer, estimator, x_model, values, tol: float = 1e-3) -> None:
    """SHAP só é interpretável se `base + soma(shap) == predição`.

    Violação silenciosa distribuiria a contribuição errada entre as features e
    a leitura do gráfico ficaria inválida sem nada acusar, então é erro.
    """
    base = np.ravel(np.asarray(explainer.expected_value, dtype=float))[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., 0]
    erro = float(np.abs(base + arr.sum(axis=1) - estimator.predict(x_model)).max())
    if not np.isfinite(erro) or erro > tol:
        raise ValueError(
            f"SHAP não aditivo (erro máx {erro:.3g} > {tol}): base + soma(shap) não "
            "reconstrói a predição — os valores não são interpretáveis"
        )


def shap_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
    """|SHAP| médio por feature, com o sinal médio (direção do efeito)."""
    out = pd.DataFrame({
        "shap_abs_mean": shap_df.abs().mean(),
        "shap_mean": shap_df.mean(),
    })
    out["share_pct"] = 100 * out["shap_abs_mean"] / out["shap_abs_mean"].sum()
    return out.sort_values("shap_abs_mean", ascending=False)


# ── Importância por ablação ("feature imputation") ──────────────────────────

def ablation_importance(
    art: dict,
    data: EvalMatrix,
    strategy: str = "mean",
    n_repeats: int = 5,
    seed: int = 42,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Piora da métrica ao neutralizar cada feature, uma de cada vez.

    `reference` é a matriz de onde sai o valor de substituição (idealmente o
    conjunto de TREINO; na ausência, a própria matriz de avaliação). Usar a
    média do treino reproduz o que um imputer faria em produção.

    `mean` é determinístico (`n_repeats` ignorado); `shuffle` é repetido e
    reportado com média e desvio.
    """
    if strategy not in ABLATION_STRATEGIES:
        raise ValueError(f"strategy inválida: {strategy!r} ({' | '.join(ABLATION_STRATEGIES)})")

    ref = data.x if reference is None else reference
    base_pred = predict(art, data.x, data.era5)
    base = compute_metrics(data.y, base_pred)
    rng = np.random.default_rng(seed)

    rows = []
    for feat in art["features"]:
        deltas = []
        reps = 1 if strategy == "mean" else n_repeats
        for _ in range(reps):
            x_mod = data.x.copy()
            if strategy == "mean":
                x_mod[feat] = float(ref[feat].mean())
            else:
                x_mod[feat] = rng.permutation(x_mod[feat].to_numpy())
            m = compute_metrics(data.y, predict(art, x_mod, data.era5))
            deltas.append((m["RMSE"] - base["RMSE"], m["Bias_P90"] - base["Bias_P90"], m["R2"] - base["R2"]))
        arr = np.asarray(deltas, dtype=float)
        rows.append({
            "feature": feat,
            "d_RMSE": arr[:, 0].mean(),
            "d_RMSE_std": arr[:, 0].std(ddof=0) if reps > 1 else 0.0,
            "d_Bias_P90": arr[:, 1].mean(),
            "d_R2": arr[:, 2].mean(),
        })

    out = pd.DataFrame(rows).set_index("feature")
    out.attrs["baseline"] = base
    out.attrs["strategy"] = strategy
    return out.sort_values("d_RMSE", ascending=False)
