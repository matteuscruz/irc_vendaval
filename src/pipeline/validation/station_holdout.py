"""Orquestrador de station-holdout (spatial k-fold / LOOCV) reusável pelas
pipelines `cluster_mlp`/`cluster_lazy`.

Dado o DataFrame flat de um único cluster (já filtrado por `cluster_id`,
todas as estações e todo o período temporal disponível), itera folds
espaciais: em cada fold, um subconjunto de estações é totalmente excluído do
treino e usado só para avaliação, com climatologia populacional (ver
`climatology_population.py`) para evitar que a avaliação da estação held-out
se beneficie da própria série histórica dela.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src.pipelines.common import ERA5_GUST_PROXY, split_part
from src.pipeline.validation.spatial_folds import build_station_folds, FOLD_COL
from src.pipeline.validation.climatology_population import (
    compute_population_climatology,
    apply_population_climatology,
)


@dataclass
class HoldoutFold:
    fold_id: int
    held_out_stations: list
    df_train: pd.DataFrame
    df_eval: pd.DataFrame


def iter_holdout_folds(
    df_cluster: pd.DataFrame,
    mode: str,
    n_folds: int = 5,
    seed: int = 42,
    train_label: str = "train",
    eval_label: str = "test",
    clim_value_col: str = ERA5_GUST_PROXY,
) -> Iterator[HoldoutFold]:
    """Itera folds de station-holdout sobre `df_cluster`.

    Para cada fold: `df_train` contém as estações NÃO held-out, restritas às
    linhas de treino do split por blocos de mês, com `era5_clim_wind`
    recalculada como climatologia populacional (cluster-mean) sobre essas
    mesmas estações de treino. `df_eval` contém só as estações held-out,
    restritas às linhas de teste, com a mesma climatologia populacional
    aplicada (mapeada por dia-do-ano — a climatologia independe do ano).

    O recorte temporal vem da coluna de rótulos (`SPLIT_COL`) escrita pelo
    loader da pipeline, e não de um intervalo de datas: os blocos de validação
    são sorteados uma única vez sobre todos os anos, então recalcular a
    partição por fold devolveria um recorte diferente do usado no treino.

    `clim_value_col` deve ser a MESMA coluna usada na construção original de
    `era5_clim_wind` (ver `src/pipelines/common.py::build_flat_dataframe`) —
    `ERA5_GUST_PROXY` (`wind_mag_max`) em lazy e mlp. Um valor incorreto
    recalcularia a climatologia populacional a partir da coluna errada, sem
    levantar erro (silenciosamente incoerente).
    """
    folds = build_station_folds(df_cluster, mode=mode, n_folds=n_folds, seed=seed)
    fold_ids = sorted(folds[FOLD_COL].unique())

    for fid in fold_ids:
        held_out_stations = folds.loc[folds[FOLD_COL] == fid, "estacao"].tolist()
        train_stations = folds.loc[folds[FOLD_COL] != fid, "estacao"].tolist()
        if not train_stations or not held_out_stations:
            continue

        df_train_raw = split_part(
            df_cluster[df_cluster["estacao"].isin(train_stations)], train_label
        )
        df_eval_raw = split_part(
            df_cluster[df_cluster["estacao"].isin(held_out_stations)], eval_label
        )
        if df_train_raw.empty or df_eval_raw.empty:
            continue

        population_clim = compute_population_climatology(df_train_raw, value_col=clim_value_col)
        df_train = apply_population_climatology(df_train_raw, population_clim)
        df_eval = apply_population_climatology(df_eval_raw, population_clim)
        if df_train.empty or df_eval.empty:
            continue

        yield HoldoutFold(
            fold_id=int(fid),
            held_out_stations=held_out_stations,
            df_train=df_train,
            df_eval=df_eval,
        )
