"""Split por blocos de mês nas pipelines tabulares (lazy/mlp).

O ponto central: a partição é resolvida UMA vez sobre todo o eixo de tempo e
viaja com o dado (`SPLIT_COL`). Recalcular por cluster ou por fold devolveria
outros blocos de validação — e a climatologia, ajustada só nos dias de treino,
ficaria incoerente com o treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines.common import (
    SPLIT_COL,
    assign_split_labels,
    default_month_block_split,
    split_part,
    split_spec_from_labels,
)


def _frame(start="2010-01-01", end="2019-12-31", stations=("A", "B")):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({
        "time": np.tile(dates, len(stations)),
        "estacao": np.repeat(list(stations), len(dates)),
    })


def test_test_split_is_one_month_per_climatic_season():
    df, split = assign_split_labels(_frame())
    meses_teste = sorted(set(pd.DatetimeIndex(split_part(df, "test")["time"]).month))

    assert meses_teste == [1, 4, 7, 10]
    assert split.test_months == (1, 4, 7, 10)


def test_train_and_val_never_touch_test_months():
    df, _ = assign_split_labels(_frame())
    for label in ("train", "val"):
        meses = set(pd.DatetimeIndex(split_part(df, label)["time"]).month)
        assert not meses & {1, 4, 7, 10}


def test_same_day_has_the_same_label_for_every_station():
    df, _ = assign_split_labels(_frame(stations=("A", "B", "C")))
    por_dia = df.groupby("time")[SPLIT_COL].nunique()

    assert (por_dia == 1).all()


def test_validation_blocks_are_whole_year_month_units():
    df, split = assign_split_labels(_frame())
    val = pd.DatetimeIndex(split_part(df, "val")["time"])
    unidades = {(t.year, t.month) for t in val}

    assert unidades == set(split.val_units)
    # Todo dia de um bloco sorteado é validação: o bloco não é partido.
    for ano, mes in unidades:
        no_bloco = df[(df["time"].dt.year == ano) & (df["time"].dt.month == mes)]
        assert (no_bloco[SPLIT_COL] == "val").all()


def test_every_training_month_keeps_both_train_and_val():
    _, split = assign_split_labels(_frame())
    meses_val = {m for _, m in split.val_units}

    assert meses_val == {2, 3, 5, 6, 8, 9, 11, 12}


def test_spec_recovered_from_labels_matches_the_split():
    df, split = assign_split_labels(_frame())
    spec = split_spec_from_labels(df)

    assert spec["scheme"] == "month_block"
    assert [tuple(u) for u in spec["val_units"]] == list(split.val_units)


def test_spec_is_the_canonical_schema_and_round_trips():
    """O artefato grava esta especificação e a inferência/XAI a relê com
    `MonthBlockSplit.from_dict`. Um formato próprio aqui tornaria o artefato
    ilegível justamente por quem precisa dele."""
    from src.pipeline.data.splits import MonthBlockSplit

    df, split = assign_split_labels(_frame())
    spec = split_spec_from_labels(df)

    assert set(spec) == set(split.to_dict())
    reconstruido = MonthBlockSplit.from_dict(spec)
    assert reconstruido.val_units == split.val_units
    assert reconstruido.test_months == split.test_months
    # Rotula igual — é disso que a climatologia depende.
    dias = pd.date_range("2012-01-01", "2012-12-31", freq="D")
    assert (reconstruido.label(dias) == split.label(dias)).all()


def test_spec_without_optional_fields_still_labels():
    """Artefatos antigos gravaram só test_months/val_units; com as unidades
    explícitas, val_fraction e seed não afetam a rotulagem."""
    from src.pipeline.data.splits import MonthBlockSplit

    minimo = MonthBlockSplit.from_dict({"test_months": [1, 4, 7, 10], "val_units": [[2015, 3]]})
    dias = pd.date_range("2015-01-01", "2015-12-31", freq="D")
    rotulos = pd.Series(minimo.label(dias), index=dias)

    assert set(rotulos[rotulos == "test"].index.month) == {1, 4, 7, 10}
    assert set(rotulos[rotulos == "val"].index.month) == {3}


def test_split_part_refuses_unlabeled_frame():
    with pytest.raises(ValueError, match=SPLIT_COL):
        split_part(_frame(), "train")


def test_split_is_reproducible_for_the_same_seed():
    times = pd.date_range("2010-01-01", "2019-12-31", freq="D").values
    a = default_month_block_split(times, seed=42)
    b = default_month_block_split(times, seed=42)
    c = default_month_block_split(times, seed=7)

    assert a.val_units == b.val_units
    assert a.val_units != c.val_units


def test_day_of_year_mean_climatology_would_lose_the_test_set():
    """Justifica a troca para a climatologia harmônica: sob blocos de mês, os
    dias-do-ano do teste não têm amostra de treino, então uma média por
    dia-do-ano ficaria indefinida e descartaria quase todo o teste."""
    df, _ = assign_split_labels(_frame())
    doy_treino = set(pd.DatetimeIndex(split_part(df, "train")["time"]).dayofyear)
    doy_teste = set(pd.DatetimeIndex(split_part(df, "test")["time"]).dayofyear)

    cobertos = doy_teste & doy_treino
    assert len(cobertos) / len(doy_teste) < 0.1
