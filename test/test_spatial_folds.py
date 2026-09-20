"""Testes para folds espaciais e climatologia populacional (station-holdout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.validation.spatial_folds import (
    assign_spatial_folds,
    leave_one_station_out_folds,
    build_station_folds,
    FOLD_COL,
)
from src.pipeline.validation.climatology_population import (
    compute_population_climatology,
    apply_population_climatology,
    CLIM_COL,
)
from src.pipeline.validation.station_holdout import iter_holdout_folds
from src.pipelines.common import SPLIT_COL


def _make_stations_df(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "estacao": [f"S{i}" for i in range(n)],
        "latitude": rng.uniform(-30, -25, n),
        "longitude": rng.uniform(-53, -48, n),
    })


def _make_flat_df(n_stations=6, n_days=20, seed=0):
    stations = _make_stations_df(n_stations, seed)
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    for _, st in stations.iterrows():
        for d in dates:
            rows.append({
                "estacao": st["estacao"],
                "latitude": st["latitude"],
                "longitude": st["longitude"],
                "time": d,
                "dayofyear": d.dayofyear,
                "wind_mag_max": rng.uniform(2, 20),
                "daily_wind_gust_max": rng.uniform(2, 20),
            })
    df = pd.DataFrame(rows)
    # `iter_holdout_folds` lê o rótulo de split da coluna escrita pelo loader
    # da pipeline; aqui metade dos dias vira treino e metade teste.
    cut = dates[len(dates) // 2]
    df[SPLIT_COL] = np.where(df["time"] < cut, "train", "test")
    return df


class TestSpatialFolds:
    def test_assign_spatial_folds_count(self):
        stations = _make_stations_df(20)
        folds = assign_spatial_folds(stations, n_folds=4, seed=1)
        assert set(folds[FOLD_COL].unique()) <= set(range(4))
        assert len(folds) == 20

    def test_assign_spatial_folds_no_station_in_two_folds(self):
        stations = _make_stations_df(20)
        folds = assign_spatial_folds(stations, n_folds=5, seed=1)
        assert folds["estacao"].is_unique

    def test_assign_spatial_folds_degenerates_to_loocv(self):
        stations = _make_stations_df(5)
        folds = assign_spatial_folds(stations, n_folds=10, seed=1)
        assert len(folds) == 5
        assert sorted(folds[FOLD_COL].tolist()) == list(range(5))

    def test_leave_one_station_out_folds(self):
        stations = _make_stations_df(8)
        folds = leave_one_station_out_folds(stations)
        assert len(folds) == 8
        assert sorted(folds[FOLD_COL].tolist()) == list(range(8))
        assert folds["estacao"].is_unique

    def test_build_station_folds_invalid_mode(self):
        df = _make_flat_df()
        with pytest.raises(ValueError):
            build_station_folds(df, mode="bogus")

    def test_build_station_folds_loocv_matches_station_count(self):
        df = _make_flat_df(n_stations=7)
        folds = build_station_folds(df, mode="loocv")
        assert len(folds) == 7


class TestPopulationClimatology:
    def test_excludes_held_out_station(self):
        df = _make_flat_df(n_stations=4, n_days=5)
        held_out = "S0"
        df_train_fold = df[df["estacao"] != held_out]

        pop_clim = compute_population_climatology(df_train_fold)

        # The held-out station's own values must not influence the result.
        with_held_out = compute_population_climatology(df)
        assert not with_held_out.equals(pop_clim)

    def test_defined_for_days_absent_from_training(self):
        """Sob o split por blocos de mês, os dias-do-ano avaliados não têm
        amostra de treino: a climatologia populacional precisa ficar definida
        mesmo assim, senão o fold inteiro é descartado."""
        df = _make_flat_df(n_stations=4, n_days=120)
        treino = df[df["time"].dt.month != 4]          # abril fora do treino
        abril = sorted(set(df.loc[df["time"].dt.month == 4, "dayofyear"]))

        pop_clim = compute_population_climatology(treino)

        assert pop_clim.notna().loc[abril].all()
        assert pop_clim.index.min() == 1 and pop_clim.index.max() == 366

    def test_apply_population_climatology_maps_by_dayofyear(self):
        df = _make_flat_df(n_stations=2, n_days=3)
        pop_clim = pd.Series({1: 10.0, 2: 11.0, 3: 12.0})
        out = apply_population_climatology(df, pop_clim)
        assert (out[CLIM_COL] == out["dayofyear"].map(pop_clim)).all()

    def test_apply_population_climatology_drops_uncovered_days(self):
        df = _make_flat_df(n_stations=2, n_days=3)
        pop_clim = pd.Series({1: 10.0})  # only dayofyear=1 covered
        out = apply_population_climatology(df, pop_clim)
        assert out["dayofyear"].isin([1]).all()


class TestIterHoldoutFolds:
    def test_no_station_overlap_between_train_and_eval(self):
        df = _make_flat_df(n_stations=6, n_days=10)
        for fold in iter_holdout_folds(
            df, mode="spatial-kfold", n_folds=3, seed=1,
        ):
            train_stations = set(fold.df_train["estacao"].unique())
            eval_stations = set(fold.df_eval["estacao"].unique())
            assert train_stations.isdisjoint(eval_stations)
            assert eval_stations == set(fold.held_out_stations)

    def test_loocv_mode_one_station_per_fold(self):
        df = _make_flat_df(n_stations=5, n_days=10)
        seen_held_out = []
        for fold in iter_holdout_folds(df, mode="loocv", seed=1):
            assert len(fold.held_out_stations) == 1
            seen_held_out.append(fold.held_out_stations[0])
        assert sorted(seen_held_out) == sorted(df["estacao"].unique())
