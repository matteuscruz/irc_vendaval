"""
Constantes e utilitários compartilhados pelos pipelines de cluster
(cluster_mlp, cluster_gan, gan_augment, cluster_lazy, mlp_pytorch).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import RobustScaler

from src.data.new_features import (  # noqa: F401 (reexportado)
    KNOWN_STATIC_FEATURES, NEW_FEATURE_PREFIX,
)

# ── Constantes ────────────────────────────────────────────────────────────────

TARGET_VAR = "daily_wind_gust_max"
ERA5_GUST_PROXY = "wind_mag_max"
RANDOM_STATE = 42

# NOTA: fonte das features ERA5 abaixo é o ERA5-Basin (src/data/
# original_features_basin.py::rebuild_original_from_basin, cobre 236/243
# estações), não mais o ERA5_Stratified.nc (só 30 estações) — nomes antigos
# preservados, só a proveniência mudou. "wind_mag_min" foi removida: sem
# equivalente no ERA5-Basin (só tem mean/max/std/p90). As 10 últimas linhas
# (sufixo `_basin`) são features do ERA5-Basin SEM equivalente no
# ERA5_Stratified.nc antigo (percentis, extremos, lags/rolling de
# precipitação) — não tinham nome antigo pra reaproveitar, por isso entram
# direto com o nome novo. Todas as pipelines usam ORIGINAL_FEATURES por
# padrão, então isso garante que TODAS as 22 features do ERA5-Basin (não só
# as 12 com equivalente antigo) alimentam o treino em todas elas.
ORIGINAL_FEATURES = [
    # ERA5 wind (U/V + magnitude diária)
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "wind_mag",
    "wind_mag_max",
    "wind_mag_std",
    # Direção do vento (codificação cíclica sin/cos)
    "wind_dir_sin",
    "wind_dir_cos",
    # Convecção e persistência
    "gust_factor",
    "lag1_wind_mag_max",
    "lag2_wind_mag_max",
    "lag3_wind_mag_max",
    "lag7_wind_mag_max",
    "rolling7d_wind_mag_max",
    # Termodinâmica
    "2m_temperature",
    "2m_dewpoint_temperature",
    "relative_humidity",
    "t2m_range",
    # Dinâmica sinótica
    "surface_pressure",
    "pressure_tendency",
    "total_precipitation",
    # Sazonalidade e climatologia ERA5
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "era5_clim_wind",
    # Localização da estação
    "latitude",
    "longitude",
    # Anomalia sinótica ERA5
    "wind_mag_max_anom",
    "rolling3d_wind_mag_max",
    "lag1_pressure_tendency",
    # ERA5-Basin sem equivalente antigo — percentis e extremos (relevantes
    # pro alvo ser rajada EXTREMA, não média)
    "ws_p90_basin",
    "msl_min_basin",
    "msl_range_basin",
    "t2m_max_basin",
    "t2m_min_basin",
    "rh_p10_basin",
    "tp_max_basin",
    "tp_lag1_basin",
    "tp_lag2_basin",
    "tp_roll3_basin",
]

# Nenhuma feature pode depender da observação INMET (lags do alvo, mediana
# por estação, climatologia observada): o modelo corrige o ERA5 na grade
# inteira, onde não existe estação medindo. INMET entra só como ALVO.
# Nomes usados por versões anteriores — a inferência recusa artefatos que os
# tenham como entrada (ver `reject_inmet_derived_features`).
INMET_DERIVED_FEATURES = (
    "lag1_gust_obs", "lag2_gust_obs", "lag3_gust_obs", "rolling7d_gust_obs",
    "gust_P50",
)


def reject_inmet_derived_features(features, source: str = "") -> None:
    found = [f for f in features if f in INMET_DERIVED_FEATURES]
    if found:
        where = f" ({source})" if source else ""
        raise ValueError(
            f"Modelo{where} usa features derivadas do INMET {found}, que não "
            "existem na grade ERA5 — retreine com o conjunto de features atual."
        )

# ── ERA5 Basin — grade regional pré-agregada (Sul/Sudeste, NaN fora da bacia) ──
# As 12 features abaixo JÁ estão em ORIGINAL_FEATURES (sob os nomes antigos
# reconstruídos — ver original_features_basin.py; as outras 10 sem
# equivalente antigo entraram direto em ORIGINAL_FEATURES também). Esse grupo
# fica então redundante por padrão (mesmo dado, nome diferente) — existe só
# pra manter os braços "basin"/"all_basin" do ablation study funcionando sem
# quebrar, não adiciona sinal novo além do que "original" já tem.
ERA5_BASIN_FEATURES = [
    "ws_mean_basin",        # Velocidade média do vento (== wind_mag)
    "ws_max_basin",         # Velocidade máxima do vento (== wind_mag_max)
    "ws_std_basin",         # Desvio-padrão da velocidade do vento (== wind_mag_std)
    "sin_dir_mean_basin",   # Direção média do vento, seno (== wind_dir_sin)
    "cos_dir_mean_basin",   # Direção média do vento, cosseno (== wind_dir_cos)
    "msl_mean_basin",       # Pressão média ao nível do mar (== surface_pressure)
    "d_msl_basin",          # Tendência de pressão (== pressure_tendency)
    "t2m_mean_basin",       # Temperatura média a 2m (== 2m_temperature)
    "t2m_range_basin",      # Amplitude térmica a 2m (== t2m_range)
    "rh_mean_basin",        # Umidade relativa média (== relative_humidity)
    "td_dep_mean_basin",    # Depressão do ponto de orvalho (usada em 2m_dewpoint_temperature)
    "tp_sum_basin",         # Precipitação total diária (== total_precipitation)
]

BASE_FEATURES = ORIGINAL_FEATURES + ERA5_BASIN_FEATURES

# ── Features ERA5 novas (dataset/raw/new_features) ─────────────────────────
# Descobertas em runtime por src/data/new_features.py — os nomes dependem dos
# arquivos presentes, então o grupo é resolvido a partir das colunas do
# DataFrame (prefixo `nf_`), não de uma lista fixa. `new_features_static` e
# `new_features_dynamic` recortam esse mesmo conjunto por KNOWN_STATIC_FEATURES
# (relevo/superfície vs. variáveis horárias agregadas por dia) — úteis para
# testar um subconjunto sem misturar sinal de fontes muito diferentes.
NEW_FEATURES_GROUP = "new_features"
NEW_FEATURES_STATIC_GROUP = "new_features_static"
NEW_FEATURES_DYNAMIC_GROUP = "new_features_dynamic"
NEW_FEATURES_TOKENS = (NEW_FEATURES_GROUP, NEW_FEATURES_STATIC_GROUP, NEW_FEATURES_DYNAMIC_GROUP)

# ── Grupos de features (ablation) ───────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "original": ORIGINAL_FEATURES,
    "era5_basin": ERA5_BASIN_FEATURES,
}
VALID_GROUPS = sorted([*FEATURE_GROUPS, *NEW_FEATURES_TOKENS])
DEFAULT_FEATURE_GROUPS = "original"


def _group_tokens(spec: str | None) -> list[str]:
    if not spec:
        return [DEFAULT_FEATURE_GROUPS]
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if tokens == ["all"]:
        return VALID_GROUPS
    invalid = [t for t in tokens if t not in VALID_GROUPS]
    if invalid:
        raise ValueError(
            f"Grupo(s) de features inválido(s): {invalid}. "
            f"Válidos: {VALID_GROUPS} (ou 'all')."
        )
    return tokens


def uses_new_features(spec: str | None) -> bool:
    return bool(set(NEW_FEATURES_TOKENS) & set(_group_tokens(spec)))


def resolve_feature_groups(spec: str | None, columns=None) -> list[str]:
    """Resolve um spec textual ("original,new_features") em lista de features,
    na ordem canônica (BASE_FEATURES, depois as `nf_*` em ordem alfabética).
    spec=None → "original"; "all" → todos os grupos.

    Os grupos `new_features`/`new_features_static`/`new_features_dynamic`
    precisam de `columns` (as colunas do DataFrame já montado): sem elas, ou
    sem nenhuma coluna `nf_*` correspondente, é erro — nunca vira um conjunto
    vazio em silêncio.
    """
    tokens = _group_tokens(spec)
    selected = {f for t in tokens if t in FEATURE_GROUPS for f in FEATURE_GROUPS[t]}
    features = [f for f in BASE_FEATURES if f in selected]
    nf_tokens = set(NEW_FEATURES_TOKENS) & set(tokens)
    if nf_tokens:
        if columns is None:
            raise ValueError(f"o(s) grupo(s) {sorted(nf_tokens)} são resolvidos a partir das colunas do DataFrame")
        all_nf = {c for c in columns if str(c).startswith(NEW_FEATURE_PREFIX)}
        chosen = set()
        if NEW_FEATURES_GROUP in nf_tokens:
            chosen |= all_nf
        if NEW_FEATURES_STATIC_GROUP in nf_tokens:
            chosen |= all_nf & KNOWN_STATIC_FEATURES
        if NEW_FEATURES_DYNAMIC_GROUP in nf_tokens:
            chosen |= all_nf - KNOWN_STATIC_FEATURES
        if not chosen:
            raise ValueError(
                f"grupo(s) {sorted(nf_tokens)} pedido(s), mas nenhuma feature correspondente foi "
                "carregada (confira dataset/raw/new_features/<região>/{sl,pl,static})"
            )
        features += sorted(chosen)
    return features


def select_complete_rows(df: pd.DataFrame, features: list[str], label: str = "") -> pd.DataFrame:
    """Regra única de dados das pipelines: só ERA5, nada imputado.

    1. Toda feature pedida precisa existir como coluna (erro se faltar).
    2. Com features novas (`nf_*`) ativas, só entram os clusters com TODAS as
       estações dentro da cobertura espacial delas — cluster parcialmente
       coberto sai inteiro.
    3. Linhas com NaN em qualquer feature ativa são descartadas (bordas de
       lag/rolling, estação fora da máscara da bacia, dia fora do período da
       feature nova).
    """
    tag = f"[{label}] " if label else ""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{tag}features ausentes no DataFrame: {missing}")

    nf = [f for f in features if f.startswith(NEW_FEATURE_PREFIX)]
    if nf:
        covered = df.groupby("estacao")[nf].apply(lambda g: bool(g.notna().any().all()))
        by_cluster = df[["estacao", "cluster_id"]].drop_duplicates().assign(
            covered=lambda d: d["estacao"].map(covered)
        ).groupby("cluster_id")["covered"].all()
        keep = by_cluster[by_cluster].index
        dropped = sorted(map(str, by_cluster[~by_cluster].index))
        print(
            f"{tag}Cobertura das features novas: clusters mantidos "
            f"{sorted(map(str, keep))}; descartados (não 100% cobertos): {dropped}"
        )
        df = df[df["cluster_id"].isin(keep)]
        if df.empty:
            raise ValueError(f"{tag}nenhum cluster 100% coberto pelas features novas")

    complete = df[features].notna().all(axis=1)
    n_drop = int((~complete).sum())
    if n_drop:
        print(f"{tag}{n_drop}/{len(df)} linhas com feature ausente descartadas (sem imputação)")
    return df[complete]


# A partir de 2025-08, INMET_Stratified.nc passou a vir de
# Training_Dataset_INMET_ERA5_Paired.csv (271 estações na bacia, real desde
# 2000) em vez do antigo arquivo que só cobria 2020-2024 — ver
# scripts/build_inmet_from_paired_csv.py.
#
# ATUALIZADO 2026-09-01: TRAIN_SLICE estendido pra incluir 2000-2007
# deliberadamente, apesar de um comentário anterior aqui (removido nesta
# mudança, ver git blame) dizer que isso já tinha sido testado e viesava o
# modelo — o PR referenciado por esse comentário não pôde ser recuperado
# (histórico git deste repo foi reconstruído após perda local, ver commit
# "recover project state after local .git loss"), então essa alegação não
# pôde ser reverificada. Decisão explícita do usuário de re-treinar mesmo
# assim e comparar métricas antes/depois pra confirmar ou refutar o viés.
# Cobertura de estações é esparsa em 2000-2007 (2 em 2000, 16 em 2002, 170+
# só a partir de 2008) — se as métricas de validação piorarem depois desta
# mudança, esse é o motivo mais provável.
TRAIN_SLICE = ("2000-01-01", "2018-12-31")
VAL_SLICE   = ("2019-01-01", "2019-12-31")
TEST_SLICE  = ("2020-01-01", "2025-12-31")

# Hemisfério sul: DJF=Verão, MAM=Outono, JJA=Inverno, SON=Primavera
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
_MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}


def month_to_season(month: "pd.Series") -> "pd.Series":
    return month.map(_MONTH_TO_SEASON)


# ── DataFrame flat ────────────────────────────────────────────────────────────

def build_flat_dataframe(
    ds_inmet: Any,
    ds_era5: Any,
    station_clusters: pd.DataFrame,
    ds_clim: Any,
    require_target: bool = True,
) -> pd.DataFrame:
    """DataFrame tabular: features ERA5 + target INMET + cluster.

    `require_target`: True (default, usado pelo treino) descarta linhas sem
    `TARGET_VAR` — necessário pra treinar/avaliar contra um alvo real. Os
    correctors de inferência (SpatialCorrector/DLSpatialCorrector) passam
    `require_target=False`: eles só usam features derivadas do ERA5 pra
    prever (o alvo INMET nunca é insumo do modelo, só serviria de
    comparação) — descartar linhas sem alvo aqui jogava fora exatamente as
    estações/dias sem observação INMET real (ex.: quase toda a grade em
    2000-2006, quando pouquíssimas estações estavam ativas), impedindo a
    extrapolação para justamente os anos sem dado real — o objetivo central
    da correção."""
    # latitude/longitude são coordenada da ESTAÇÃO nos dois datasets; vindo
    # dos dois lados o merge as renomeava para latitude_x/_y e a feature
    # sumia. Fonte única: a tabela de estações do INMET.
    station_ll = ["latitude", "longitude"]
    df_feat = ds_era5.to_dataframe().reset_index().drop(columns=station_ll, errors="ignore")
    df_targ = ds_inmet[[TARGET_VAR]].reset_coords(drop=True).to_dataframe().reset_index()

    df = pd.merge(
        df_feat, df_targ, on=["time", "estacao"],
        how="inner", validate="one_to_one",
    )
    df = pd.merge(
        df, ds_inmet[station_ll].to_dataframe().reset_index(), on="estacao",
        how="left", validate="many_to_one",
    )
    if require_target:
        df = df.dropna(subset=[TARGET_VAR])

    doy = df["time"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dayofyear"] = doy

    df_clim_pd = ds_clim.reset_coords(drop=True).to_dataframe(name="era5_clim_wind").reset_index()
    df = pd.merge(
        df, df_clim_pd, on=["dayofyear", "estacao"],
        how="left", validate="many_to_one",
    )
    df = df.dropna(subset=["era5_clim_wind"])

    df = pd.merge(
        df, station_clusters, on="estacao",
        how="left", validate="many_to_one",
    )

    df = df.sort_values(["estacao", "time"]).reset_index(drop=True)

    # Sazonalidade mensal cíclica — complementa day_sin/cos para diferenças inter-mensais
    _month = df["time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * _month / 12)
    df["month_cos"] = np.cos(2 * np.pi * _month / 12)

    return df


# ── Splits temporais ──────────────────────────────────────────────────────────

def make_split(df: pd.DataFrame, time_range: tuple[str, str]) -> pd.DataFrame:
    """Recorte por intervalo de datas. Usado pelo `cluster_gan`, que mantém o
    split por blocos de ano; lazy/mlp usam blocos de mês (`SPLIT_COL`)."""
    mask = (df["time"] >= time_range[0]) & (df["time"] <= time_range[1])
    return df[mask].copy()


# ── Split por blocos de mês (lazy, mlp e LSTM) ────────────────────────────────
# Coluna com o rótulo de split de cada linha, escrita uma única vez pelo loader
# de cada pipeline tabular. O rótulo viaja com o dado em vez de ser recalculado
# por cluster: `resolve_month_block_split` sorteia os blocos de validação sobre
# os ANOS presentes, então recalcular sobre um subconjunto (um cluster, um fold)
# devolveria uma partição diferente da usada no ajuste da climatologia.
SPLIT_COL = "_split"


def default_month_block_split(times, seed: int = RANDOM_STATE):
    """Partição padrão por blocos de mês: teste = Jan/Abr/Jul/Out de todos os
    anos (um mês por trimestre climático), validação = blocos (ano, mês)
    sorteados dentro dos meses de treino, estratificados por mês."""
    from src.pipeline.data.splits import resolve_month_block_split

    return resolve_month_block_split(times, seed=seed)


def assign_split_labels(df: pd.DataFrame, split=None, seed: int = RANDOM_STATE):
    """Escreve `SPLIT_COL` em `df` a partir de um `MonthBlockSplit`.

    Retorna `(df, split)`. Sem `split`, resolve o padrão sobre os dias
    presentes em `df["time"]`.
    """
    times = pd.DatetimeIndex(df["time"])
    if split is None:
        split = default_month_block_split(np.unique(times.values), seed=seed)
    df = df.copy()
    df[SPLIT_COL] = split.label(times)
    return df, split


def split_part(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Linhas de um split ("train"/"val"/"test"). Exige `SPLIT_COL` — a
    ausência é erro, não recálculo silencioso com outra partição."""
    if SPLIT_COL not in df.columns:
        raise ValueError(
            f"coluna {SPLIT_COL!r} ausente: o DataFrame precisa vir de um loader "
            "que aplique assign_split_labels()"
        )
    return df[df[SPLIT_COL] == label].copy()


def split_spec_from_labels(
    df: pd.DataFrame, test_months=(1, 4, 7, 10), seed: int = RANDOM_STATE,
) -> dict:
    """Especificação da partição a partir de `SPLIT_COL`, para o metadado do
    experimento e para os artefatos.

    Emite o schema CANÔNICO (`MonthBlockSplit.to_dict`), não um formato
    próprio: a inferência e as ferramentas de XAI reconstroem a partição com
    `MonthBlockSplit.from_dict`, e um segundo formato para a mesma coisa
    tornaria o artefato ilegível justamente por quem precisa dele.
    """
    from src.pipeline.data.splits import MonthBlockSplit

    val = df.loc[df[SPLIT_COL] == "val", "time"]
    units = tuple(sorted({(int(t.year), int(t.month)) for t in pd.DatetimeIndex(val)}))
    return MonthBlockSplit(
        test_months=tuple(test_months), val_units=units, seed=seed,
    ).to_dict()


# ── Diretório de experimento ──────────────────────────────────────────────────

# A lógica de diretórios foi movida para src/utils/artifact_manager.py


# ── Cluster merge ─────────────────────────────────────────────────────────────

def parse_cluster_merge(spec: str | None) -> list[list[int]]:
    """Converte spec textual ("1-2-3,5-6") em grupos [[1,2,3],[5,6]]."""
    if not spec:
        return []

    groups: list[list[int]] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        members = [int(x) for x in chunk.split("-") if x.strip()]
        for m in members:
            if m in seen:
                raise ValueError(f"cluster {m} aparece em mais de um grupo de merge")
            seen.add(m)
        groups.append(members)
    return groups


def apply_cluster_merge(
    df: pd.DataFrame, groups: list[list[int]]
) -> pd.DataFrame:
    """Remapeia cluster_id para rótulos agregados."""
    if not groups:
        return df

    label_map: dict[int, str] = {}
    for members in groups:
        label = "-".join(str(m) for m in sorted(members))
        for m in members:
            label_map[m] = label

    df = df.copy()
    df["cluster_id"] = df["cluster_id"].map(
        lambda cid: label_map.get(int(cid), str(int(cid)))
    )
    return df


# ── Pré-processamento (MLP/Lazy) ──────────────────────────────────────────────

def drop_incomplete_synthetic(s: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Linhas sintéticas (CSV do GAN) sem alguma feature ativa — ex.: CSV
    gerado antes das features novas existirem — são descartadas, não
    imputadas."""
    complete = s.reindex(columns=features).notna().all(axis=1)
    if not complete.all():
        print(f"   [AVISO] {int((~complete).sum())}/{len(s)} linhas sintéticas sem todas as "
              "features ativas — descartadas (sem imputação)")
    return s[complete]


def assert_no_missing(x, label: str = "features") -> None:
    """Sem imputação: NaN que chegue ao modelo é bug de montagem dos dados
    (`select_complete_rows` deveria ter descartado a linha)."""
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).any(axis=1).sum()) if arr.ndim > 1 else int((~np.isfinite(arr)).sum())
        raise ValueError(f"{bad} linha(s) com valor ausente/infinito em {label} — nada é imputado")


def preprocess_df(
    x_train: pd.DataFrame, x_other: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RobustScaler fitado apenas no treino; retorna DataFrames. Sem imputação:
    NaN em qualquer lado é erro."""
    assert_no_missing(x_train, "x_train")
    assert_no_missing(x_other, "x_eval")
    scaler = RobustScaler().fit(x_train)
    x_tr = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    x_ot = pd.DataFrame(scaler.transform(x_other), columns=x_other.columns)
    return x_tr, x_ot


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, p: int = 90
) -> dict[str, float]:
    mask = y_true >= np.percentile(y_true, p)
    rmse_all = root_mean_squared_error(y_true, y_pred)
    rmse_tail = (
        root_mean_squared_error(y_true[mask], y_pred[mask])
        if mask.sum() > 0 else float("nan")
    )
    bias_tail = (
        float((y_pred[mask] - y_true[mask]).mean())
        if mask.sum() > 0 else float("nan")
    )
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse_all,
        "Bias": float((y_pred - y_true).mean()),
        f"Bias_P{p}": bias_tail,
        f"RMSE_P{p}": rmse_tail,
    }
