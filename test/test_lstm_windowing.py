"""Convenção única de janelas da LSTM (src/pipeline/data/windowing.py)."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.data.windowing import (
    WindowSpec,
    build_daily_windows,
    build_hourly_day_windows,
    daily_window_positions,
)


def _days(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def test_daily_window_includes_target_row():
    mat = np.arange(10, dtype=float)[:, None]  # linha i vale i
    windows, target_pos, keep = build_daily_windows(mat, _days(10), 3)
    assert windows.shape == (8, 3, 1)
    np.testing.assert_array_equal(target_pos, np.arange(2, 10))
    # o último passo da janela é o próprio dia-alvo
    np.testing.assert_array_equal(windows[:, -1, 0], target_pos)
    np.testing.assert_array_equal(windows[0, :, 0], [0, 1, 2])
    assert keep.all()


def test_positions_match_windows():
    pos, target_pos = daily_window_positions(6, 4)
    np.testing.assert_array_equal(pos[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(pos[:, -1], target_pos)


def test_edge_padding_repeats_first_row():
    mat = np.arange(5, dtype=float)[:, None]
    windows, target_pos, _ = build_daily_windows(mat, _days(5), 3, pad="edge")
    assert windows.shape == (5, 3, 1)
    np.testing.assert_array_equal(target_pos, np.arange(5))
    np.testing.assert_array_equal(windows[0, :, 0], [0, 0, 0])
    np.testing.assert_array_equal(windows[1, :, 0], [0, 0, 1])
    np.testing.assert_array_equal(windows[4, :, 0], [2, 3, 4])


def test_gap_in_calendar_raises():
    with pytest.raises(ValueError, match="lacuna"):
        build_daily_windows(np.zeros((5, 1)), _days(6).delete(3), 2)


def test_short_series_yields_no_windows():
    windows, target_pos, keep = build_daily_windows(np.zeros((2, 3)), _days(2), 5)
    assert windows.shape == (0, 5, 3)
    assert len(target_pos) == 0 and len(keep) == 0


def test_purge_drops_windows_crossing_split():
    labels = np.array(["train"] * 5 + ["test"] * 5)
    _, target_pos, keep = build_daily_windows(np.zeros((10, 1)), _days(10), 3, labels=labels)
    # alvos 2..9; cruzam a fronteira os alvos 5 ([3..5]) e 6 ([4..6])
    assert set(target_pos[keep].tolist()) == {2, 3, 4, 7, 8, 9}


def test_hourly_window_is_the_target_day():
    hours = pd.date_range("2020-01-01", periods=72, freq="h")
    windows, days = build_hourly_day_windows(np.arange(72, dtype=float)[:, None], hours)
    assert windows.shape == (3, 24, 1)
    assert list(days) == list(pd.date_range("2020-01-01", periods=3, freq="D"))
    np.testing.assert_array_equal(windows[1, :, 0], np.arange(24, 48))


def test_missing_hour_drops_that_day():
    hours = pd.date_range("2020-01-01", periods=48, freq="h").delete(30)
    windows, days = build_hourly_day_windows(np.zeros((47, 2)), hours)
    assert windows.shape == (1, 24, 2)
    assert days[0] == pd.Timestamp("2020-01-01")


def test_day_offset_shifts_day_boundary():
    hours = pd.date_range("2020-01-01", periods=48, freq="h")
    windows, days = build_hourly_day_windows(
        np.arange(48, dtype=float)[:, None], hours, day_offset_hours=3,
    )
    # dia 2020-01-01 com offset 3h = 03:00 UTC do dia 1 até 02:00 UTC do dia 2
    assert list(days) == [pd.Timestamp("2020-01-01")]
    np.testing.assert_array_equal(windows[0, :, 0], np.arange(3, 27))


def test_window_spec_round_trip_and_validation():
    spec = WindowSpec("daily", 7)
    assert WindowSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        WindowSpec("hourly", 12)
    with pytest.raises(ValueError):
        WindowSpec("weekly", 7)
