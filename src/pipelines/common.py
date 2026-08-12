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
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import RobustScaler

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
    # Conectividade temporal — observação INMET autoregressiva
    "lag1_gust_obs",
    "lag2_gust_obs",
    "lag3_gust_obs",
    "rolling7d_gust_obs",
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

# ── ERA5 18UTC — snapshot pré-convectivo (Paraná; NaN fora da cobertura) ──
ERA5_18Z_FEATURES = [
    "cape_18z",                  # CAPE — energia convectiva disponível
    "msl_18z",                   # Pressão ao nível do mar 18UTC
    "wind10m_mag_18z",           # Magnitude vento 10m 18UTC
    "wind100m_mag_18z",          # Magnitude vento 100m 18UTC
    "wind_shear_sfc_18z",        # Cisalhamento 10m→100m
    "wind_shear_850_300_18z",    # Cisalhamento vertical 850→300 hPa
    "wind_shear_850_500_18z",    # Cisalhamento vertical 850→500 hPa
    "thickness_1000_500_18z",    # Espessura 1000-500 hPa (temp. média camada)
    "lapse_rate_850_500_18z",    # Lapse rate 850-500 hPa (instabilidade)
    "q850_18z",                  # Umidade específica 850 hPa
    "mslp_tendency_18z",         # Tendência pressão nível do mar 18UTC
]

# ── BT55 — temperatura de brilho ≤ 55°C (convecção profunda, Paraná) ──
BT55_FEATURES = [
    "bt55_flag",                 # Flag binário convecção profunda
    "bt55_rolling3d",            # Convecção recente (rolling 3 dias)
    "bt55_rolling7d",            # Tendência convectiva (rolling 7 dias)
    "bt55_frac_month",           # Fração mensal convecção (rolling 30 dias)
]

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

BASE_FEATURES = ORIGINAL_FEATURES + ERA5_18Z_FEATURES + BT55_FEATURES + ERA5_BASIN_FEATURES

# ── Grupos de features (ablation) ───────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "original": ORIGINAL_FEATURES,
    "era5_18z": ERA5_18Z_FEATURES,
    "bt55": BT55_FEATURES,
    "era5_basin": ERA5_BASIN_FEATURES,
}


def resolve_feature_groups(spec: str | None) -> list[str]:
    """Resolve um spec textual ("original,era5_18z,bt55") em uma lista de
    features, preservando a ordem canônica de BASE_FEATURES independente da
    ordem do spec. spec=None ou "all" retorna BASE_FEATURES completo.
    """
    if not spec or spec.strip().lower() == "all":
        return list(BASE_FEATURES)

    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    invalid = [t for t in tokens if t not in FEATURE_GROUPS]
    if invalid:
        raise ValueError(
            f"Grupo(s) de features inválido(s): {invalid}. "
            f"Válidos: {sorted(FEATURE_GROUPS)} (ou 'all')."
        )

    selected = {f for t in tokens for f in FEATURE_GROUPS[t]}
    return [f for f in BASE_FEATURES if f in selected]


def restrict_to_feature_coverage(df: pd.DataFrame, feature_groups: str | None) -> pd.DataFrame:
    """Restringe `df` às estações com cobertura REAL (não-NaN) das features
    era5_18z/bt55 pedidas em `feature_groups` — em vez de manter as 243
    estações com essas colunas imputadas/NaN pra ~197 delas.

    era5_18z é regional (só Paraná, ~46/243 estações reais); bt55 cobre
    ~57/243. Sem essa restrição, o braço de ablation "newfeatures" mede o
    efeito das features novas diluído por ~80% de estações onde elas nunca
    tiveram dado de verdade — pior teste do que testar só onde a feature
    existe. Não afeta 'original'/'era5_basin' (cobertura ampla, 236/243) nem
    arms que não pedem era5_18z/bt55 (retorna df sem alteração).
    """
    if not feature_groups:
        return df
    tokens = {t.strip().lower() for t in feature_groups.split(",") if t.strip()}
    cols = []
    if "era5_18z" in tokens:
        cols += ERA5_18Z_FEATURES
    if "bt55" in tokens:
        cols += BT55_FEATURES
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    covered = df.groupby("estacao")[cols].apply(lambda g: bool(g.notna().all().all()))
    covered_stations = covered[covered].index
    n_before = df["estacao"].nunique()
    out = df[df["estacao"].isin(covered_stations)]
    print(
        f"[restrict_to_feature_coverage] {len(covered_stations)}/{n_before} "
        f"estações com cobertura real de {tokens & {'era5_18z', 'bt55'}} — "
        "demais descartadas (evita diluir o teste com NaN/imputação)."
    )
    return out

# A partir de 2025-08, INMET_Stratified.nc passou a vir de
# Training_Dataset_INMET_ERA5_Paired.csv (271 estações na bacia, real desde
# 2000) em vez do antigo arquivo que só cobria 2020-2024 — ver
# scripts/build_inmet_from_paired_csv.py. A rede só fica robusta (170+
# estações simultâneas, ver validação no PR) a partir de 2008; antes disso
# a cobertura é esparsa demais (2 estações em 2000, 16 em 2002) pra treinar
# sem viesar. TEST começa em 2020 pra bater com o período que os modelos
# antigos (treinados só em 2020-2022) já foram avaliados, permitindo
# comparação direta de métricas entre a rede antiga e a nova.
TRAIN_SLICE = ("2008-01-01", "2018-12-31")
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
) -> pd.DataFrame:
    """DataFrame tabular: features ERA5 + target INMET + cluster."""
    df_feat = ds_era5.to_dataframe().reset_index()
    df_targ = ds_inmet[[TARGET_VAR]].to_dataframe().reset_index()

    df = pd.merge(
        df_feat, df_targ, on=["time", "estacao"],
        how="inner", validate="one_to_one",
    )
    df = df.dropna(subset=[TARGET_VAR])

    doy = df["time"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dayofyear"] = doy

    df_clim_pd = ds_clim.to_dataframe(name="era5_clim_wind").reset_index()
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

    # Lags autoregressivos da observação INMET — causal (shift dentro de cada estação)
    for _lag in [1, 2, 3]:
        df[f"lag{_lag}_gust_obs"] = df.groupby("estacao")[TARGET_VAR].shift(_lag)
    # Rolling 7d da observação: shift(1) antes do rolling evita incluir o dia atual
    df["rolling7d_gust_obs"] = (
        df.groupby("estacao")[TARGET_VAR]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    )

    # Sazonalidade mensal cíclica — complementa day_sin/cos para diferenças inter-mensais
    _month = df["time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * _month / 12)
    df["month_cos"] = np.cos(2 * np.pi * _month / 12)

    return df


# ── Splits temporais ──────────────────────────────────────────────────────────

def make_split(df: pd.DataFrame, time_range: tuple[str, str]) -> pd.DataFrame:
    mask = (df["time"] >= time_range[0]) & (df["time"] <= time_range[1])
    return df[mask].copy()


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

def preprocess_df(
    x_train: pd.DataFrame, x_other: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SimpleImputer + RobustScaler fitados apenas no treino; retorna DataFrames.

    `keep_empty_features=True`: por padrão o SimpleImputer DESCARTA colunas
    inteiramente NaN no treino (menos colunas na saída do que na entrada),
    o que quebra a atribuição `columns=x_train.columns` logo abaixo (shape
    mismatch) sempre que um subconjunto de linhas (ex.: um cluster pequeno,
    ou um fold de station-holdout) deixa alguma feature 100% NaN — ex.:
    features ERA5-18UTC/BT55 só cobrem a região do Paraná. Mantém a coluna
    (preenchida com 0.0) em vez de descartá-la.
    """
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True).fit(x_train)
    scaler = RobustScaler().fit(imputer.transform(x_train))

    x_tr = pd.DataFrame(
        scaler.transform(imputer.transform(x_train)),
        columns=x_train.columns,
    )
    x_ot = pd.DataFrame(
        scaler.transform(imputer.transform(x_other)),
        columns=x_other.columns,
    )
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
