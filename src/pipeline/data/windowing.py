"""Convenção ÚNICA de janelas da LSTM — diária e horária.

Antes, "janela = os `lookback` dias ANTERIORES ao dia-alvo" estava hardcoded em
3 lugares (_make_windows, spatial_correction_dl, grid_direct_predict), e o
dia-alvo ficava FORA da janela porque entrava pela âncora razão×ERA5. Com o
alvo em m/s não há mais âncora, então a janela precisa INCLUIR o dia-alvo —
senão o modelo nunca vê o tempo do próprio dia. Treino e inferência só chamam
as funções daqui, para essa convenção não divergir de novo.

- diário : alvo no dia i -> linhas [i-L+1 .. i]
- horário: alvo no dia D -> as 24 horas do dia D (00..23h), deslocáveis por
           `day_offset_hours` quando o "dia" do alvo não é o dia UTC
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.pipeline.data.splits import purge_keep

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class WindowSpec:
    resolution: str  # "daily" | "hourly"
    length: int
    day_offset_hours: int = 0

    def __post_init__(self) -> None:
        if self.resolution not in ("daily", "hourly"):
            raise ValueError(f"resolution inválida: {self.resolution!r} (daily | hourly)")
        if self.length < 1:
            raise ValueError(f"length precisa ser >= 1, recebido {self.length}")
        if self.resolution == "hourly" and self.length != HOURS_PER_DAY:
            raise ValueError(
                f"janela horária é sempre o dia inteiro ({HOURS_PER_DAY} passos), "
                f"recebido length={self.length}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WindowSpec":
        return cls(**d)


def daily_window_positions(
    n_rows: int, length: int, pad: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Posições das janelas diárias numa série de `n_rows` linhas.

    Retorna (pos[N, L], target_pos[N]), com pos[j] = [t-L+1 .. t] para
    t = target_pos[j]. `pad="none"` só gera alvos com janela completa
    (t >= L-1); `pad="edge"` gera um alvo por linha, repetindo a primeira linha
    no início da série (inferência: evita um buraco nos primeiros L-1 dias).
    """
    if length < 1:
        raise ValueError(f"length precisa ser >= 1, recebido {length}")
    if pad == "none":
        target_pos = np.arange(length - 1, n_rows)
    elif pad == "edge":
        target_pos = np.arange(n_rows)
    else:
        raise ValueError(f"pad inválido: {pad!r} (none | edge)")
    pos = target_pos[:, None] + np.arange(-(length - 1), 1)[None, :]
    return np.clip(pos, 0, None), target_pos


def assert_gap_free_daily(times) -> None:
    """Janela por posição só vale sobre calendário diário contínuo — antes,
    linhas sem alvo eram descartadas antes de janelar e uma janela de "7 dias"
    podia cobrir semanas."""
    t = pd.DatetimeIndex(times)
    if len(t) < 2:
        return
    bad = np.flatnonzero(np.asarray((t[1:] - t[:-1]) != pd.Timedelta(days=1)))
    if len(bad):
        i = bad[0]
        raise ValueError(
            "série diária com lacuna ou fora de ordem — janelas por posição "
            f"cruzariam dias inexistentes: {t[i].date()} -> {t[i + 1].date()}. "
            "Reindexe para um calendário diário contínuo antes de janelar."
        )


def build_daily_windows(
    mat,
    times,
    length: int,
    *,
    pad: str = "none",
    labels=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Janelas diárias de uma série (uma estação ou célula).

    mat: (n, F) já imputado/escalonado. times: (n,) diário contínuo.
    Retorna (windows[N, L, F], target_pos[N], keep[N]). `windows` é uma VIEW
    (sliding_window_view) — indexar com `keep` copia só o que fica. Com
    `labels`, `keep` aplica a purga de fronteira de split; sem, é tudo True.
    """
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"mat precisa ser 2-D (n, F), recebido shape {mat.shape}")
    if len(mat) != len(times):
        raise ValueError(f"mat ({len(mat)}) e times ({len(times)}) com tamanhos diferentes")
    assert_gap_free_daily(times)

    pos, target_pos = daily_window_positions(len(mat), length, pad)
    if len(target_pos) == 0:
        return (
            np.empty((0, length, mat.shape[1]), dtype=mat.dtype),
            target_pos,
            np.ones(0, dtype=bool),
        )
    src = mat if pad == "none" else np.concatenate(
        [np.repeat(mat[:1], length - 1, axis=0), mat]
    )
    windows = np.lib.stride_tricks.sliding_window_view(src, length, axis=0).transpose(0, 2, 1)

    keep = np.ones(len(target_pos), dtype=bool)
    if labels is not None:
        labels = np.asarray(labels)
        keep = purge_keep(labels[pos], labels[target_pos])
    return windows, target_pos, keep


def build_hourly_day_windows(
    mat_h,
    hour_times,
    *,
    day_offset_hours: int = 0,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Uma janela por dia: as 24 horas do dia D -> alvo = rajada máxima de D.

    `day_offset_hours` desloca a fronteira do dia (ex.: 3 = dia começando às
    03:00 UTC, i.e. dia local UTC-3). Dias sem as 24 horas completas são
    descartados. Retorna (windows[D, 24, F], dias[D]).
    """
    mat_h = np.asarray(mat_h)
    t = pd.DatetimeIndex(hour_times)
    if len(mat_h) != len(t):
        raise ValueError(f"mat_h ({len(mat_h)}) e hour_times ({len(t)}) com tamanhos diferentes")
    if t.has_duplicates:
        raise ValueError("timestamps horários duplicados")
    shifted = t - pd.Timedelta(hours=day_offset_hours)
    if (shifted != shifted.floor("h")).any():
        raise ValueError("timestamps horários fora da hora cheia")

    days = shifted.floor("D")
    uniq, day_idx = np.unique(days.values, return_inverse=True)
    grid = np.full((len(uniq), HOURS_PER_DAY), -1, dtype=np.int64)
    grid[day_idx, np.asarray(shifted.hour)] = np.arange(len(t))
    complete = (grid >= 0).all(axis=1)
    if not complete.any():
        return np.empty((0, HOURS_PER_DAY, mat_h.shape[1]), dtype=mat_h.dtype), pd.DatetimeIndex([])
    return mat_h[grid[complete]], pd.DatetimeIndex(uniq[complete])
