"""Segunda fonte de features novas: `dataset/raw/test_cluster_3_hourly.nc`.

Diferente de `src/data/new_features.py` (que lê uma GRADE lat/lon e interpola
para as estações), este arquivo já vem por ESTAÇÃO — `(time, estacao)`, sem
dimensão espacial — então não há grade para validar nem célula da bacia para
mascarar. É uma fonte à parte, com seu próprio agregador horário→diário.

Motivação (achado da investigação do DJF, `investigacao_djf_rajadas_convectivas.ipynb`):
a agregação diária (mean/max/std) apaga rajadas convectivas de minutos. Este
módulo testa se estatísticas horárias — em especial as que capturam VARIAÇÃO
dentro do dia, não só nível — carregam o sinal que a agregação diária perde.

Regras (mesma disciplina do resto do projeto):
  - Nenhuma imputação: um dia com qualquer uma das 24 horas faltando (NaN)
    é DESCARTADO inteiro para aquela variável, nunca preenchido.
  - Toda feature sai com o prefixo `nf_h_` — `nf_` para ser descoberta
    automaticamente por `arms.usable_new_features` (mesma convenção do
    projeto), `h_` para não colidir por nome com as features da grade
    (ex.: `nf_t2m_max` do grupo 1 é uma variável DIFERENTE de `nf_h_t2m_max`
    aqui — fontes distintas, mesmo símbolo seria enganoso).

Vetorizado: a série horária do arquivo é regular (1h, sem buracos, múltiplo de
24 — verificado), então cada variável vira um array (n_dias, 24, n_estações)
por `reshape` puro, em vez de um `groupby` do pandas por dia.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NEW_FEATURE_PREFIX = "nf_h_"
DEFAULT_HOURLY_FILE = "test_cluster_3_hourly.nc"

# Estatística por variável. `range` = max − min do dia (amplitude diurna —
# quedas bruscas de pressão/temperatura são assinatura de frente/tempestade).
# `jump` = maior salto ABSOLUTO entre horas consecutivas do dia (uma rajada
# que dura 1-2h aparece aqui e se dilui no mean/max/std diário).
HOURLY_STATS: dict[str, tuple[str, ...]] = {
    "ws_h": ("mean", "max", "std", "jump"),
    "msl_h": ("mean", "min", "range"),
    "t2m_h": ("mean", "max", "range"),
    "rh_h": ("mean", "min"),
    "td_dep_h": ("mean", "max"),   # depressão do ponto de orvalho: max = ar mais seco (potencial convectivo)
    "tp_h": ("sum",),
    "tp_roll24h": ("max",),
    "sin_dir_h": ("mean",),
    "cos_dir_h": ("mean",),
}
# wd_h fica de fora: direção em graus não agrega por mean/max sem cruzar o
# zero (0°/360°) — sin_dir_h/cos_dir_h já cobrem a direção de forma agregável.


def _require_regular_hourly(times: pd.DatetimeIndex) -> np.ndarray:
    """(n_dias, 24) de timestamps, um dia por linha. Levanta se a série não for
    horária regular começando à meia-noite — a forma vetorizada exige isso."""
    if len(times) % 24 != 0:
        raise ValueError(f"{len(times)} timesteps não é múltiplo de 24h — série irregular")
    deltas = np.diff(times.values)
    if not np.all(deltas == deltas[0]) or deltas[0] != np.timedelta64(1, "h"):
        raise ValueError("série de tempo não é horária regular (1h) sem buracos")
    if times[0].hour != 0:
        raise ValueError(f"primeiro timestep não é meia-noite: {times[0]}")
    return times.values.reshape(-1, 24)


def _stat(arr: np.ndarray, stat: str) -> np.ndarray:
    """`arr`: (n_dias, 24). Um dia com QUALQUER NaN vira NaN inteiro (sem
    imputação) — verificado antes de cada estatística, não depois."""
    bad = np.isnan(arr).any(axis=1)
    with np.errstate(invalid="ignore"):
        if stat == "mean":
            out = arr.mean(axis=1)
        elif stat == "max":
            out = arr.max(axis=1)
        elif stat == "min":
            out = arr.min(axis=1)
        elif stat == "std":
            out = arr.std(axis=1, ddof=0)
        elif stat == "sum":
            out = arr.sum(axis=1)
        elif stat == "range":
            out = arr.max(axis=1) - arr.min(axis=1)
        elif stat == "jump":
            out = np.abs(np.diff(arr, axis=1)).max(axis=1)
        else:
            raise ValueError(f"estatística desconhecida: {stat!r}")
    out[bad] = np.nan
    return out


def load_hourly_daily(raw_dir: str | Path, filename: str = DEFAULT_HOURLY_FILE) -> pd.DataFrame | None:
    """Agregado diário por estação: colunas `nf_h_<var>_<stat>`, colunas
    `estacao`/`time`. `None` se o arquivo não existir — o estudo segue sem essa
    fonte, como já acontece quando `new_features/` está vazio."""
    path = Path(raw_dir) / filename
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        variaveis = [v for v in HOURLY_STATS if v in ds.data_vars]
        if not variaveis:
            return None
        times = pd.DatetimeIndex(ds.time.values)
        day_grid = _require_regular_hourly(times)
        dias = pd.DatetimeIndex(day_grid[:, 0])
        estacoes = [str(e) for e in ds.estacao.values]

        cols: dict[str, np.ndarray] = {}
        for var in variaveis:
            valores = ds[var].values  # (time, estacao)
            por_dia = valores.reshape(len(dias), 24, len(estacoes))  # (dias, 24, estações)
            for stat in HOURLY_STATS[var]:
                nome = f"{NEW_FEATURE_PREFIX}{var.removesuffix('_h')}_{stat}"
                # uma coluna por estação, empilhadas na ordem (dia, estação) — mesma ordem do índice abaixo
                por_estacao = np.stack([_stat(por_dia[:, :, i], stat) for i in range(len(estacoes))], axis=1)
                cols[nome] = por_estacao.reshape(-1)  # (dias*estações,), varia estação mais rápido

    idx = pd.MultiIndex.from_product([dias, estacoes], names=["time", "estacao"])
    out = pd.DataFrame(cols, index=idx).reset_index()
    return out


def merge_hourly_features(df: pd.DataFrame, raw_dir: str | Path, filename: str = DEFAULT_HOURLY_FILE) -> pd.DataFrame:
    """Junta as colunas `nf_h_*` a `df` por (estacao, time). Estação/dia sem
    arquivo horário correspondente fica com NaN nessas colunas — a mesma
    regra de completude do estudo (`select_complete_rows`) as descarta depois,
    sem imputar."""
    hourly = load_hourly_daily(raw_dir, filename)
    if hourly is None:
        return df
    df = df.copy()
    df["estacao"] = df["estacao"].astype(str)
    hourly["estacao"] = hourly["estacao"].astype(str)
    return df.merge(hourly, on=["estacao", "time"], how="left")
