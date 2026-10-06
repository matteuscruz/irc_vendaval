"""Janelas horárias da LSTM (`src/feature_study/data/sequences.py`).

Parquets sintéticos do teste dos grupos: `w10` vale `hora/24 + índice da estação`, então o
valor de cada passo identifica a hora e a estação de onde veio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.data import groups_source as gs
from src.feature_study.data import sequences as sq
from test.test_feature_study_groups_source import DIAS, ESTACOES, _escreve


@pytest.fixture
def raw(tmp_path):
    return _escreve(tmp_path / "raw", hora_pico=lambda i: 10 + (i % 5))


def _rows_and_peaks(raw, ndias=4, estacao="A509"):
    keys = gs.peak_keys(raw, ESTACOES)
    keys = keys[keys[gs.STATION_SRC] == estacao].sort_values("dia").iloc[10:10 + ndias]
    rows = pd.DataFrame({gs.STATION: keys[gs.STATION_SRC].to_numpy(), gs.TIME: keys["dia"].to_numpy()})
    peaks = pd.DataFrame({gs.STATION: keys[gs.STATION_SRC].astype(str).to_numpy(), gs.TIME: keys["dia"].to_numpy(),
                          gs.PEAK_HOUR_COLUMN: keys[gs.TIME_SRC].dt.hour.to_numpy()})
    return rows, peaks, keys


def test_the_last_step_of_each_window_is_the_peak_hour_row(raw):
    """O último passo tem de ser exatamente o que o estudo tabular usa como linha do dia."""
    rows, peaks, keys = _rows_and_peaks(raw)
    w = sq.load_windows(raw, rows, ["w10", "cape"], 6, peaks, stations=ESTACOES)
    daily, _ = gs.build_daily_table(raw, ESTACOES)
    esperado = daily.merge(rows, on=[gs.STATION, gs.TIME]).sort_values(gs.TIME)
    np.testing.assert_allclose(w[:, -1, 0], esperado["w10"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(w[:, -1, 1], esperado["cape"].to_numpy(), rtol=1e-5)


def test_earlier_steps_walk_back_one_hour_each(raw):
    rows, peaks, _ = _rows_and_peaks(raw)
    w = sq.load_windows(raw, rows, ["w10"], 5, peaks, stations=ESTACOES)
    hora = peaks[gs.PEAK_HOUR_COLUMN].to_numpy()
    # w10 = hora/24 + k (k = índice da estação A509 = 1): cada passo anterior perde 1/24
    for passo in range(5):
        np.testing.assert_allclose(w[:, passo, 0], (hora - (4 - passo)) / 24.0 + 1.0, atol=1e-5)


def test_a_window_crosses_midnight_into_the_previous_day(raw):
    """Pico às 02 UTC com janela de 6 h alcança 21–23 h do dia anterior."""
    raw2 = _escreve(raw.parent / "raw_madrugada", hora_pico=lambda i: 2)
    rows, peaks, _ = _rows_and_peaks(raw2)
    w = sq.load_windows(raw2, rows, ["w10"], 6, peaks, stations=ESTACOES)
    esperado = np.array([21, 22, 23, 0, 1, 2]) / 24.0 + 1.0
    np.testing.assert_allclose(w[0, :, 0], esperado, atol=1e-5)


def test_hours_that_do_not_exist_stay_nan_instead_of_being_filled(raw):
    rows = pd.DataFrame({gs.STATION: ["A509"], gs.TIME: [DIAS[0]]})
    peaks = pd.DataFrame({gs.STATION: ["A509"], gs.TIME: [DIAS[0]], gs.PEAK_HOUR_COLUMN: [1]})
    w = sq.load_windows(raw, rows, ["w10"], 6, peaks, stations=ESTACOES)
    assert np.isnan(w[0, :4, 0]).all() and np.isfinite(w[0, 4:, 0]).all()   # antes do 1º dia da série


def test_a_row_without_a_cached_peak_hour_raises(raw):
    rows = pd.DataFrame({gs.STATION: ["A509"], gs.TIME: [pd.Timestamp("1999-01-01")]})
    peaks = pd.DataFrame({gs.STATION: ["A509"], gs.TIME: [DIAS[0]], gs.PEAK_HOUR_COLUMN: [1]})
    with pytest.raises(ValueError, match="sem hora de pico"):
        sq.load_windows(raw, rows, ["w10"], 6, peaks, stations=ESTACOES)


def test_static_columns_are_not_sequences(raw):
    rows, peaks, _ = _rows_and_peaks(raw)
    with pytest.raises(ValueError, match="fora dos grupos horários"):
        sq.load_windows(raw, rows, ["sdor_ponto"], 6, peaks, stations=ESTACOES)


def test_windows_do_not_mix_stations(raw):
    rows1, peaks1, _ = _rows_and_peaks(raw, estacao="A504")
    rows2, peaks2, _ = _rows_and_peaks(raw, estacao="A515")
    rows = pd.concat([rows1, rows2], ignore_index=True)
    peaks = pd.concat([peaks1, peaks2], ignore_index=True)
    w = sq.load_windows(raw, rows, ["w10"], 3, peaks, stations=ESTACOES)
    k = {e: i for i, e in enumerate(sorted(ESTACOES))}
    for i, e in enumerate(rows[gs.STATION]):
        assert np.floor(w[i, -1, 0] + 1e-6) == k[e]
