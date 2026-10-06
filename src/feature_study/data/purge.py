"""Gap da purga temporal, derivado do alcance para trás das features.

Só as colunas do spec olham para fora do instante: o alcance está no próprio sufixo
(`gust10fg_lag1h`, `w10_max_prev3h`, `blh_delta3h`, `mslp_tend_3h`). O gap em dias é
`ceil(max alcance / 24)` sobre o conjunto de features efetivamente montado, nunca uma
constante solta: se uma feature com defasagem maior entrar, o gap acompanha sozinho.
Com as defasagens atuais (até 3 h) o gap é de 1 dia.
"""
from __future__ import annotations

import math
import re

# `gust10fg_lag1h`, `w10_max_prev3h`, `blh_delta3h`, `mslp_tend_3h`
_PADRAO_LOOKBACK = re.compile(r"(?:lag|prev|delta|tend_?)(\d+)h$")


def lookback_hours(column: str) -> int:
    """Quantas horas para TRÁS uma coluna enxerga. Zero para o que é do próprio instante."""
    m = _PADRAO_LOOKBACK.search(column)
    return int(m.group(1)) if m else 0


def purge_days(features) -> int:
    """Gap em DIAS que o split precisa nas fronteiras de mês."""
    horas = max((lookback_hours(f) for f in features), default=0)
    return math.ceil(horas / 24)
