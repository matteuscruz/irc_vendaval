"""BASE antiga (ERA5 horário por estação) na hora do pico, SEM nenhum grupo.

A base antiga é horária (`cluster_3_hourly.nc`, 12 variáveis); a estrutura por grupo
tem uma linha por estação-dia na hora do pico da rajada ERA5. Para treinar um modelo
só com a BASE sobre as MESMAS linhas dos arms do estudo, lê-se cada variável da base
na hora do pico (`hora_pico_utc`) do próprio dia. Nada é imputado: o que faltar fica NaN.

Fica de fora de propósito latitude/longitude (grupos 1/4 no spec) e qualquer coluna dos
quatro parquets: "sem nenhum grupo".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_FILE = "cluster_3_hourly.nc"
BASE_VARS = ("ws_h", "wd_h", "sin_dir_h", "cos_dir_h", "msl_h", "t2m_h", "rh_h", "td_dep_h",
             "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h")
PREFIX = "base_"


def base_columns() -> list[str]:
    return [PREFIX + v for v in BASE_VARS]


def base_at_peak(raw_dir, keys: pd.DataFrame) -> pd.DataFrame:
    """`keys`: colunas `estacao`, `time` (dia UTC) e `hora_pico_utc`. Devolve `keys`
    mais as 12 variáveis da base, lidas em `time + hora_pico_utc`."""
    with xr.open_dataset(Path(raw_dir) / BASE_FILE) as ds:
        horas = pd.DatetimeIndex(ds.time.values)
        est = [str(e) for e in ds.estacao.values]
        pos_t = horas.get_indexer(pd.DatetimeIndex(keys["time"]) + pd.to_timedelta(keys["hora_pico_utc"], unit="h"))
        pos_e = pd.Index(est).get_indexer(keys["estacao"].astype(str))
        out = keys.copy()
        ok = (pos_t >= 0) & (pos_e >= 0)
        for v in BASE_VARS:
            arr = np.asarray(ds[v].transpose("time", "estacao").values, dtype=float)
            col = np.full(len(keys), np.nan)
            col[ok] = arr[pos_t[ok], pos_e[ok]]
            out[PREFIX + v] = col
    return out
