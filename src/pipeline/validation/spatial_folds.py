"""Folds espaciais (station-holdout) para validação cruzada geograficamente
coesa, análogo a `interpolation comparisson/loocv/spatial_kfold.py`.

Por que não K-fold aleatório: numa correção espacial, os vizinhos mais
próximos de qualquer estação retida continuam quase certamente no conjunto de
treino de um fold aleatório -- a predição fica artificialmente fácil. Um fold
espacial (agrupando estações geograficamente próximas no MESMO fold, retida em
bloco) reproduz melhor a dificuldade real de extrapolar para uma estação nunca
vista.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FOLD_COL = "fold_id"


def assign_spatial_folds(
    stations_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Atribui um fold espacial (0..n_folds-1) a cada estação via KMeans em
    (lat, lon). `stations_df` deve ter uma linha por estação, com colunas
    `estacao`, `lat_col`, `lon_col`.

    Se `n_folds >= n_estações`, degenera para leave-one-station-out (cada
    estação isolada em seu próprio fold).
    """
    stations_df = stations_df.drop_duplicates(subset=["estacao"]).reset_index(drop=True)
    n_stations = len(stations_df)
    if n_stations == 0:
        raise ValueError("stations_df vazio — nenhuma estação para atribuir folds")

    if n_folds >= n_stations:
        return leave_one_station_out_folds(stations_df)

    coords = stations_df[[lat_col, lon_col]].to_numpy()
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    fold_ids = km.fit_predict(coords)

    return pd.DataFrame({
        "estacao": stations_df["estacao"].values,
        FOLD_COL: fold_ids,
    })


def leave_one_station_out_folds(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Cada estação é seu próprio fold (LOOCV) — `fold_id` é apenas um índice
    posicional 0..n-1, sem significado geográfico."""
    stations_df = stations_df.drop_duplicates(subset=["estacao"]).reset_index(drop=True)
    return pd.DataFrame({
        "estacao": stations_df["estacao"].values,
        FOLD_COL: np.arange(len(stations_df)),
    })


def build_station_folds(
    df: pd.DataFrame,
    mode: str,
    n_folds: int = 5,
    seed: int = 42,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Ponto de entrada único: `df` é o DataFrame flat de um cluster (uma
    linha por estação×dia). Deriva a tabela de estações e delega para
    `assign_spatial_folds`/`leave_one_station_out_folds` conforme `mode`
    ("spatial-kfold" ou "loocv"). Retorna DataFrame [estacao, fold_id]."""
    if mode not in ("spatial-kfold", "loocv"):
        raise ValueError(f"mode inválido: {mode!r} (use 'spatial-kfold' ou 'loocv')")

    stations_df = (
        df[["estacao", lat_col, lon_col]]
        .drop_duplicates(subset=["estacao"])
        .reset_index(drop=True)
    )

    if mode == "loocv":
        return leave_one_station_out_folds(stations_df)
    return assign_spatial_folds(
        stations_df, n_folds=n_folds, seed=seed, lat_col=lat_col, lon_col=lon_col
    )
