"""Janelas horárias para a LSTM, sobre as MESMAS linhas do estudo tabular.

O estudo tem uma linha por (estação, dia), na hora do pico da rajada ERA5. Para
a LSTM, cada linha vira uma **janela das últimas `T` horas até essa hora de
pico**, com as mesmas colunas que o arm usa:

  - colunas dos grupos 1–3 (horárias nos parquets) entram como SEQUÊNCIA, uma
    leitura por hora;
  - o resto (relevo do grupo 4, latitude/longitude, ruído e estáticas
    permutadas dos controles) é constante na linha e fica de fora da sequência
    — o chamador o repete a cada passo.

Isso mantém tudo o que importa para comparar com o LazyPredict: mesmas linhas,
mesmo `row_id`, mesmos arms, mesma partição. O que muda é só o modelo e o fato
de ele ver a evolução das horas anteriores, em vez de um único instante.

Sem imputação: uma hora que falta na janela vira NaN, e a linha inteira é
tratada como não utilizável pela LSTM (ver `worker_lstm`). A hora do pico vem do
cache do `prepare` (`_cache/_daily_peak_cache.parquet`), nunca recalculada.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.data import groups_source as gs

PEAK_CACHE = gs.CACHE_NAME
_KEY_BASE = 10_000_000   # > horas desde 1970 (~4,8e5); separa as estações na chave inteira


def load_peak_hours(cache_dir) -> pd.DataFrame:
    """(estacao, time, hora_pico_utc) de cada estação-dia, do cache do `prepare`."""
    path = Path(cache_dir) / PEAK_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe: rode o `prepare` antes (ele grava a hora do pico de cada dia)"
        )
    df = pd.read_parquet(path, columns=[gs.STATION, gs.TIME, gs.PEAK_HOUR_COLUMN])
    df[gs.STATION] = df[gs.STATION].astype(str)
    return df


def with_peak_hour(rows: pd.DataFrame, peak_hours: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `hora_pico_utc` a `rows` (estacao, time). Levanta se alguma linha não casar:
    sem a hora não há janela, e escolher uma hora qualquer seria inventar o dado."""
    base = rows[[gs.STATION, gs.TIME]].copy()
    base[gs.STATION] = base[gs.STATION].astype(str)
    out = base.merge(peak_hours, on=[gs.STATION, gs.TIME], how="left", validate="many_to_one")
    if out[gs.PEAK_HOUR_COLUMN].isna().any():
        n = int(out[gs.PEAK_HOUR_COLUMN].isna().sum())
        raise ValueError(f"{n} linha(s) sem hora de pico no cache — o cache não é o do `prepare` desta população")
    return out


def window_hours(day, peak_hour, window: int) -> np.ndarray:
    """Matriz (n, T) de horas (inteiros, horas desde 1970) de cada janela, em ordem
    cronológica: a última coluna é a própria hora do pico."""
    end = pd.DatetimeIndex(day).values.astype("datetime64[h]").astype("int64") + np.asarray(peak_hour, "int64")
    return end[:, None] - np.arange(window - 1, -1, -1, dtype="int64")[None, :]


def load_windows(raw_dir, rows: pd.DataFrame, seq_cols: list[str], window: int,
                 peak_hours: pd.DataFrame, stations=None) -> np.ndarray:
    """Janelas `(n, T, F)` float32 de `seq_cols`, uma por linha de `rows`.

    `rows` precisa de `estacao` e `time` (o dia). NaN onde a hora não existe nos parquets.
    A leitura é em lotes e só guarda as horas que alguma janela usa, para caber na
    memória da máquina local (~4 GB) mesmo com ~170 colunas.
    """
    seq_cols = list(seq_cols)
    n = len(rows)
    out = np.full((n, window, len(seq_cols)), np.nan, dtype="float32")
    if not seq_cols or n == 0:
        return out

    estacoes = rows[gs.STATION].astype(str).to_numpy()
    todas = sorted(set(map(str, stations)) if stations is not None else set(estacoes))
    codigo = {e: i for i, e in enumerate(todas)}
    pick = with_peak_hour(rows, peak_hours)
    horas = window_hours(pick[gs.TIME], pick[gs.PEAK_HOUR_COLUMN].to_numpy(), window)
    chave = np.array([codigo[e] for e in estacoes], dtype="int64")[:, None] * _KEY_BASE + horas
    precisa = np.unique(chave)

    col_de = {}
    for g in gs.HOURLY_GROUPS:
        for c in gs.hourly_columns(raw_dir, g):
            col_de.setdefault(c, g)
    ausentes = [c for c in seq_cols if c not in col_de]
    if ausentes:
        raise ValueError(f"colunas fora dos grupos horários, não formam sequência: {ausentes[:5]}")

    for g in gs.HOURLY_GROUPS:
        cols = [c for c in seq_cols if col_de[c] == g]
        if not cols:
            continue
        ks, vs = [], []
        for d in gs._iter_row_groups(gs.group_path(raw_dir, g), [gs.STATION_SRC, gs.TIME_SRC, *cols], todas):
            h = d[gs.TIME_SRC].values.astype("datetime64[h]").astype("int64")
            k = d[gs.STATION_SRC].astype(str).map(codigo).to_numpy("int64") * _KEY_BASE + h
            usa = np.isin(k, precisa)
            if usa.any():
                ks.append(k[usa])
                vs.append(d.loc[usa, cols].to_numpy("float32"))
        if not ks:
            continue
        k_all, v_all = np.concatenate(ks), np.concatenate(vs)
        ordem = np.argsort(k_all, kind="stable")
        k_all, v_all = k_all[ordem], v_all[ordem]
        pos = np.clip(np.searchsorted(k_all, chave), 0, len(k_all) - 1)
        achou = k_all[pos] == chave
        destino = [seq_cols.index(c) for c in cols]
        for j, dst in enumerate(destino):
            col = np.where(achou, v_all[pos, j], np.nan)
            out[:, :, dst] = col
    return out
