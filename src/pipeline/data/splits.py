"""Split temporal por BLOCOS DE MÊS para a LSTM, sem vazamento temporal.

Teste = meses fixos (default Jan/Abr/Jul/Out) de TODOS os anos, cobrindo as
quatro estações do ano. Treino = demais meses. Validação (EarlyStopping) =
blocos (ano, mês) sorteados DENTRO dos meses de treino, sempre para todas as
estações ao mesmo tempo — um mesmo dia nunca aparece em dois splits por
estações diferentes.

Meses de teste intercalados com os de treino criam fronteiras dentro de cada
ano. A purga de janela (`purge_keep`) impede que informação atravesse essas
fronteiras: uma amostra só entra no split do seu dia-alvo se TODOS os dias da
janela têm o mesmo rótulo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

LABEL_DTYPE = "<U5"  # "train" | "val" | "test" | "out"


@dataclass(frozen=True)
class MonthBlockSplit:
    test_months: tuple[int, ...] = (1, 4, 7, 10)
    val_fraction: float = 0.15
    seed: int = 42
    # Unidades (ano, mês) de validação já sorteadas — ver
    # resolve_month_block_split. Vazio = sem validação.
    val_units: tuple[tuple[int, int], ...] = ()
    # Datas fora deste intervalo recebem o rótulo "out" (descartadas).
    date_range: tuple[str, str] | None = None
    scheme: str = "month_block"

    def __post_init__(self) -> None:
        if self.scheme != "month_block":
            raise ValueError(f"scheme desconhecido: {self.scheme!r}")
        if not self.test_months or any(not 1 <= m <= 12 for m in self.test_months):
            raise ValueError(f"test_months inválidos: {self.test_months}")
        if len(set(self.test_months)) == 12:
            raise ValueError("todos os 12 meses no teste — não sobra treino")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction precisa estar em [0, 1), recebido {self.val_fraction}")

    def label(self, times) -> np.ndarray:
        """Rótulo de cada timestamp: train, val, test ou out."""
        t = pd.DatetimeIndex(times)
        months = np.asarray(t.month)
        labels = np.full(len(t), "train", dtype=LABEL_DTYPE)
        is_test = np.isin(months, self.test_months)
        labels[is_test] = "test"
        if self.val_units:
            codes = np.asarray(t.year) * 100 + months
            val_codes = [y * 100 + m for y, m in self.val_units]
            labels[np.isin(codes, val_codes) & ~is_test] = "val"
        if self.date_range is not None:
            lo, hi = (pd.Timestamp(d) for d in self.date_range)
            labels[np.asarray((t < lo) | (t > hi))] = "out"
        return labels

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "test_months": list(self.test_months),
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "val_units": [list(u) for u in self.val_units],
            "date_range": list(self.date_range) if self.date_range else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MonthBlockSplit":
        """`val_fraction` e `seed` descrevem COMO os blocos foram sorteados;
        com `val_units` explícito eles não afetam a rotulagem, então são
        opcionais — artefatos que só gravaram as unidades continuam legíveis."""
        return cls(
            test_months=tuple(int(m) for m in d["test_months"]),
            val_fraction=float(d.get("val_fraction", cls.val_fraction)),
            seed=int(d.get("seed", cls.seed)),
            val_units=tuple((int(y), int(m)) for y, m in d.get("val_units", [])),
            date_range=tuple(d["date_range"]) if d.get("date_range") else None,
            scheme=d.get("scheme", "month_block"),
        )


def resolve_month_block_split(
    times,
    *,
    test_months=(1, 4, 7, 10),
    val_fraction: float = 0.15,
    seed: int = 42,
    date_range: tuple[str, str] | None = None,
    stratify_by_month: bool = True,
) -> MonthBlockSplit:
    """Sorteia as unidades de validação a partir dos anos presentes em `times`.

    `stratify_by_month=True`: para cada mês de treino, sorteia
    `round(val_fraction × n_anos)` anos (mínimo 1), mas sempre deixa ao menos
    um ano daquele mês no treino — assim todo mês de treino tem amostras de
    treino e de validação. Com `False`, sorteia sobre o conjunto de todas as
    unidades (ano, mês) de treino.
    """
    base = MonthBlockSplit(
        test_months=tuple(sorted({int(m) for m in test_months})),
        val_fraction=float(val_fraction),
        seed=int(seed),
        date_range=tuple(date_range) if date_range else None,
    )
    t = pd.DatetimeIndex(times)
    train_t = t[base.label(t) == "train"]
    units = sorted(set(zip(train_t.year.tolist(), train_t.month.tolist())))
    if base.val_fraction == 0 or not units:
        return base

    rng = np.random.default_rng(base.seed)
    chosen: list[tuple[int, int]] = []
    if stratify_by_month:
        for month in sorted({m for _, m in units}):
            years = sorted(y for y, m in units if m == month)
            k = min(len(years) - 1, max(1, round(base.val_fraction * len(years))))
            if k > 0:
                chosen += [(int(y), month) for y in rng.choice(years, size=k, replace=False)]
    else:
        k = min(len(units) - 1, max(1, round(base.val_fraction * len(units))))
        if k > 0:
            chosen = [units[i] for i in rng.choice(len(units), size=k, replace=False)]
    return replace(base, val_units=tuple(sorted(chosen)))


def split_from_config(times, cfg: dict | None, seed: int = 42) -> MonthBlockSplit:
    """`data.split` do YAML → MonthBlockSplit com a validação já sorteada
    sobre os anos presentes em `times`."""
    cfg = dict(cfg or {})
    if cfg.get("scheme", "month_block") != "month_block":
        raise ValueError(f"split.scheme desconhecido: {cfg.get('scheme')!r}")
    if cfg.get("purge", "strict") != "strict":
        raise ValueError(f"split.purge desconhecido: {cfg.get('purge')!r} (só 'strict')")
    return resolve_month_block_split(
        times,
        test_months=cfg.get("test_months", (1, 4, 7, 10)),
        val_fraction=cfg.get("val_fraction", 0.15),
        seed=cfg.get("seed", seed),
        date_range=cfg.get("date_range"),
        stratify_by_month=cfg.get("val_stratify_by_month", True),
    )


def purge_keep(window_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
    """Máscara das amostras que ficam: dia-alvo dentro do intervalo de datas e
    todos os dias da janela com o mesmo rótulo do dia-alvo.

    window_labels: (N, L) — rótulo de cada passo da janela.
    target_labels: (N,)   — rótulo do dia-alvo.
    """
    wl = np.asarray(window_labels)
    tl = np.asarray(target_labels)
    return (tl != "out") & (wl == tl[:, None]).all(axis=1)
