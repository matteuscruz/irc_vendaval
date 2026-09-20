"""Split por blocos de mês e purga de janela (src/pipeline/data/splits.py)."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.data.splits import (
    MonthBlockSplit,
    purge_keep,
    resolve_month_block_split,
)

DAYS = pd.date_range("2016-01-01", "2019-12-31", freq="D")
TEST_MONTHS = [1, 4, 7, 10]


def test_test_months_labelled_in_every_year():
    labels = MonthBlockSplit().label(DAYS)
    is_test = np.isin(DAYS.month, TEST_MONTHS)
    assert set(labels[is_test]) == {"test"}
    assert set(labels[~is_test]) == {"train"}


def test_val_units_are_whole_train_months_and_deterministic():
    a = resolve_month_block_split(DAYS, val_fraction=0.25, seed=7)
    b = resolve_month_block_split(DAYS, val_fraction=0.25, seed=7)
    assert a.val_units and a.val_units == b.val_units
    assert all(m not in TEST_MONTHS for _, m in a.val_units)
    labels = a.label(DAYS)
    for y, m in a.val_units:
        in_unit = (DAYS.year == y) & (DAYS.month == m)
        assert set(labels[in_unit]) == {"val"}


def test_stratified_keeps_train_and_val_in_every_train_month():
    labels = resolve_month_block_split(DAYS, val_fraction=0.25, seed=0).label(DAYS)
    for month in set(range(1, 13)) - set(TEST_MONTHS):
        assert {"train", "val"} <= set(labels[DAYS.month == month])


def test_zero_val_fraction_has_no_validation():
    split = resolve_month_block_split(DAYS, val_fraction=0.0)
    assert split.val_units == ()
    assert "val" not in set(split.label(DAYS))


def test_round_trip_dict():
    split = resolve_month_block_split(
        DAYS, val_fraction=0.15, seed=3, date_range=("2016-03-01", "2019-11-30"),
    )
    assert MonthBlockSplit.from_dict(split.to_dict()) == split


def test_date_range_marks_out():
    labels = MonthBlockSplit(date_range=("2017-01-01", "2018-12-31")).label(DAYS)
    assert set(labels[DAYS < "2017-01-01"]) == {"out"}
    inside = (DAYS >= "2017-01-01") & (DAYS <= "2018-12-31")
    assert "out" not in set(labels[inside])


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        MonthBlockSplit(test_months=(13,))
    with pytest.raises(ValueError):
        MonthBlockSplit(val_fraction=1.0)
    with pytest.raises(ValueError):
        MonthBlockSplit(test_months=tuple(range(1, 13)))


def test_purge_keep():
    window_labels = np.array([["train", "train"], ["train", "test"], ["out", "out"]])
    target_labels = np.array(["train", "test", "out"])
    np.testing.assert_array_equal(
        purge_keep(window_labels, target_labels), [True, False, False],
    )
