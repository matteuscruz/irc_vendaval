"""Achatamento horário RAW: as 24 horas do dia viram 24 COLUNAS.

Terceira representação do dado horário no projeto, e a mais crua das três:

  agregada   `hourly_source.py` — 24h → mean/max/std/jump (18 colunas `nf_h_*`)
  sequência  `pipeline/data/windowing.py` — 24h → tensor (24, F), para a LSTM
  achatada   ESTE módulo — 24h → 24 colunas, uma por hora, sem redução

A motivação é a mesma do estudo de DJF (`investigacao_djf_rajadas_convectivas.ipynb`):
`mean`/`max` de 24 horas colapsam o perfil intradiário, que é justamente o que
separa uma rajada convectiva de fim de tarde de um evento sinótico. Aqui nada é
colapsado — o modelo recebe o perfil inteiro e decide sozinho o que olhar. O
preço é largura: ~679 colunas contra ~5.900 linhas de treino por trimestre, o
que torna os controles negativos (`ctrl__noise`, `ctrl__perm_static`) o portão
de validade do estudo, não um diagnóstico opcional.

Duas origens, mesmo achatamento:

  `hf_`   `dataset/raw/test_cluster_3_hourly.nc` — já vem por ESTAÇÃO
          (`time`, `estacao`), sem grade. É a base do estudo RAW.
  `hfn_`  `dataset/raw/new_features/cluster3/sl/*.nc` — vem em GRADE lat/lon e
          precisa ser extraída a pontos antes (ver `src/data/new_features.py`,
          modo horário). São as variáveis novas.

Nenhum dos dois usa o prefixo `nf_`: `arms.usable_new_features` varre por `nf_`
para descobrir features automaticamente, e o roteamento dos grupos temáticos
depende de distinguir base (`hf_`) de novas (`hfn_`).

Regras, iguais às do resto do projeto:
  - Nenhuma imputação. Um dia com qualquer uma das 24 horas faltando é
    descartado INTEIRO para aquela variável — as 24 colunas dela viram NaN.
  - Dia = UTC (`DAY_OFFSET_HOURS = 0`), como em `hourly_builder.py`.
  - `wd_h` fica de fora: direção em graus não interpola em 0°/360°;
    `sin_dir_h`/`cos_dir_h` cobrem a direção.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.hourly_source import (
    DEFAULT_HOURLY_FILE, _require_regular_hourly,
)

HOURLY_FLAT_PREFIX = "hf_"
HOURLY_FLAT_NEW_PREFIX = "hfn_"
DAY_OFFSET_HOURS = 0
HOURS = tuple(range(24))

# Variáveis da base horária por estação (`test_cluster_3_hourly.nc`).
BASE_HOURLY_VARS: tuple[str, ...] = (
    "ws_h", "sin_dir_h", "cos_dir_h",
    "msl_h", "t2m_h", "rh_h", "td_dep_h",
    "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h",
)

# Quantas horas para TRÁS cada variável enxerga. Tudo que não está aqui é do
# próprio instante (0). É daqui que sai o gap da purga temporal — ver
# `purge_days`: uma coluna que olha 72h para trás faz uma linha de treino em 1º
# de fevereiro conter informação de 30-31 de janeiro, que é teste.
LOOKBACK_HOURS: dict[str, int] = {
    "tp_roll24h": 24,
    "tp_roll48h": 48,
    "tp_roll72h": 72,
}


def flat_name(var: str, hour: int, prefix: str = HOURLY_FLAT_PREFIX) -> str:
    """`ws_h` + hora 7 → `hf_ws_h07`. O sufixo `_h` do nome da variável é
    removido para não virar `hf_ws_h_h07`."""
    return f"{prefix}{var.removesuffix('_h')}_h{hour:02d}"


def hourly_flat_columns(variables=BASE_HOURLY_VARS, prefix: str = HOURLY_FLAT_PREFIX) -> list[str]:
    """Nomes na ordem (variável, hora) — a ordem das colunas preserva a
    sequência intradiária, que é o ponto do achatamento."""
    return [flat_name(v, h, prefix) for v in variables for h in HOURS]


def base_variable_of(column: str) -> str:
    """`hf_ws_h07` → `ws_h`; `hfn_blh_h13` → `blh`. Inverso de `flat_name`,
    usado pelo roteamento dos grupos temáticos e pela purga."""
    for prefix in (HOURLY_FLAT_NEW_PREFIX, HOURLY_FLAT_PREFIX):
        if column.startswith(prefix):
            corpo = column[len(prefix):]
            raiz = corpo.rsplit("_h", 1)[0]
            return raiz if prefix == HOURLY_FLAT_NEW_PREFIX else _restore_suffix(raiz)
    return column


def _restore_suffix(raiz: str) -> str:
    """As variáveis da base têm sufixo `_h` no arquivo (`ws_h`), menos as
    `tp_roll*`. `flat_name` o removeu; aqui ele volta, para casar com
    `LOOKBACK_HOURS` e com os nomes do NetCDF."""
    return raiz if raiz.startswith("tp_roll") else f"{raiz}_h"


# Nomes da estrutura nova que carregam o alcance para trás no próprio sufixo:
# `gust10fg_lag1h`, `w10_max_prev3h`, `blh_delta3h`, `mslp_tend_3h`.
_PADRAO_LOOKBACK = re.compile(r"(?:lag|prev|delta|tend_?)(\d+)h$")


def lookback_hours(column: str) -> int:
    """Quantas horas para TRÁS uma coluna enxerga. Zero para o que é do próprio
    instante. Vale para as duas estruturas: o registro `LOOKBACK_HOURS` cobre
    as `tp_roll*` da base horária antiga e o padrão de nome cobre as
    defasagens do spec."""
    m = _PADRAO_LOOKBACK.search(column)
    if m:
        return int(m.group(1))
    return LOOKBACK_HOURS.get(base_variable_of(column), 0)


def purge_days(features) -> int:
    """Gap em DIAS que o split precisa nas fronteiras de mês, derivado do
    conjunto de features — nunca uma constante solta.

    Hoje dá 3, por causa de `tp_roll72h`. Cai para 2 se essa coluna sair, e
    para 0 num conjunto sem nenhuma `tp_roll*`. Se uma feature com lag for
    acrescentada depois, basta registrá-la em `LOOKBACK_HOURS` e o gap
    acompanha sozinho.
    """
    horas = max((lookback_hours(f) for f in features), default=0)
    return math.ceil(horas / 24)


def flatten_hourly(
    values: np.ndarray, variables, n_days: int, n_stations: int,
    prefix: str = HOURLY_FLAT_PREFIX,
) -> dict[str, np.ndarray]:
    """`values`: (n_vars, n_days*24, n_stations) → {coluna: (n_days*n_stations,)}.

    A saída varia ESTAÇÃO mais rápido que dia, para casar com
    `MultiIndex.from_product([dias, estacoes])`.

    A completude é avaliada por (dia, estação, variável) ANTES de separar as
    horas: se qualquer uma das 24 for NaN, as 24 colunas daquela variável
    naquele dia saem NaN. Zerar só a hora faltante seria imputação implícita —
    o modelo veria um perfil fisicamente impossível.
    """
    cols: dict[str, np.ndarray] = {}
    for i, var in enumerate(variables):
        por_dia = values[i].reshape(n_days, 24, n_stations)
        bad = np.isnan(por_dia).any(axis=1)  # (dias, estações)
        for h in HOURS:
            hora = por_dia[:, h, :].copy()
            hora[bad] = np.nan
            cols[flat_name(var, h, prefix)] = hora.reshape(-1)
    return cols


def load_hourly_flat(
    raw_dir: str | Path, filename: str = DEFAULT_HOURLY_FILE,
    variables=BASE_HOURLY_VARS,
) -> pd.DataFrame | None:
    """Base horária achatada por estação: `estacao`, `time` e 24 colunas
    `hf_<var>_h<HH>` por variável. `None` se o arquivo não existir — o estudo
    segue sem essa fonte, como já acontece com `new_features/` vazio."""
    path = Path(raw_dir) / filename
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        presentes = [v for v in variables if v in ds.data_vars]
        if not presentes:
            return None
        times = pd.DatetimeIndex(ds.time.values)
        grade = _require_regular_hourly(times)
        dias = pd.DatetimeIndex(grade[:, 0])
        estacoes = [str(e) for e in ds.estacao.values]
        valores = np.stack([np.asarray(ds[v].values, dtype="float64") for v in presentes])
        cols = flatten_hourly(valores, presentes, len(dias), len(estacoes))

    idx = pd.MultiIndex.from_product([dias, estacoes], names=["time", "estacao"])
    return pd.DataFrame(cols, index=idx).reset_index()


def merge_hourly_flat(
    df: pd.DataFrame, raw_dir: str | Path, filename: str = DEFAULT_HOURLY_FILE,
    variables=BASE_HOURLY_VARS,
) -> pd.DataFrame:
    """Junta as colunas `hf_*` a `df` por (estacao, time). Estação/dia sem
    correspondente fica NaN, e `select_complete_rows` descarta depois — sem
    imputar em nenhum ponto."""
    flat = load_hourly_flat(raw_dir, filename, variables)
    if flat is None:
        return df
    df = df.copy()
    df["estacao"] = df["estacao"].astype(str)
    flat["estacao"] = flat["estacao"].astype(str)
    return df.merge(flat, on=["estacao", "time"], how="left")
