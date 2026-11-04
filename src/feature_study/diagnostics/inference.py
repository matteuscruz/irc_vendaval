"""Inferência de sanidade: o melhor modelo de cada trimestre, só com as colunas selecionadas, aplicado à GRADE.

O estudo escolheu um modelo por trimestre (`summary/top_models.json`, rank 1) e um conjunto de colunas
(`sel__val12`), mas nunca gravou essa combinação ajustada. Aqui ela é reajustada com o MESMO treino, seed e
pré-processamento do estudo (`worker.fit_arm`), e aplicada a `test_inference_cluster_3.parquet`: ~74 células de
0,25° do cluster 3, horárias, 2000–2025 — pontos e um ano que o modelo nunca viu.

O desenho é o do treino: uma linha por (célula, dia UTC) na hora de maior `gust10fg` do dia; a previsão é a rajada
máxima diária. O arquivo horário passa de 8 GB em float64, então é lido grupo de linhas a grupo de linhas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from src.feature_study.core.config import CLIP_RANGE, MODEL_SEED
from src.feature_study.core.worker import fit_arm, load_unit_frames, select_regressors
from src.pipeline.data.splits import MonthBlockSplit
from src.pipelines.common import preprocess_df

LAT, LON, TIME = "lat", "lon", "time"
PEAK = "gust10fg"
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
STUDY_END_YEAR = 2024
TEST_MONTHS = MonthBlockSplit().test_months      # jan/abr/jul/out, o split do estudo
# códigos de `amostra` no NetCDF
FORA, TREINO, TESTE, NOVO = 0, 1, 2, 3


def _peak_of(df: pd.DataFrame) -> pd.DataFrame:
    """Linha de maior `gust10fg` por (célula, dia); descarta o dia com alguma hora de `gust10fg` faltando."""
    df = df.assign(dia=df[TIME].dt.floor("D"))
    chave = [LAT, LON, "dia"]
    falta = df[PEAK].isna().groupby([df[c] for c in chave]).transform("any")
    n_h = df.groupby(chave)[TIME].transform("size")
    df = df[~falta]
    idx = df.groupby(chave)[PEAK].idxmax()
    out = df.loc[idx].copy()
    out["_n_horas"] = n_h.loc[idx].to_numpy()
    return out


def daily_peak_rows(parquet, columns) -> pd.DataFrame:
    """Uma linha por (célula, dia UTC) na hora do pico de `gust10fg`, lendo um grupo de linhas por vez.

    Uma célula (ou um dia) pode cair na fronteira entre dois grupos: o pico de cada pedaço é guardado e o
    dia é re-reduzido no fim, somando as horas vistas, para só aceitar dia com as 24 horas."""
    cols = list(dict.fromkeys([LAT, LON, TIME, PEAK, *columns]))
    pf = pq.ParquetFile(parquet)
    partes = []
    for i in range(pf.num_row_groups):
        partes.append(_peak_of(pf.read_row_group(i, columns=cols).to_pandas()))
    d = pd.concat(partes, ignore_index=True)
    chave = [LAT, LON, "dia"]
    horas = d.groupby(chave)["_n_horas"].transform("sum")
    d = d.loc[d.groupby(chave)[PEAK].idxmax()]
    d = d[horas.loc[d.index] == 24]
    d = d.rename(columns={TIME: "hora_pico"}).rename(columns={"dia": TIME}).drop(columns="_n_horas")
    d["season"] = d[TIME].dt.month.map(SEASON_OF_MONTH)
    return d.sort_values([TIME, LAT, LON]).reset_index(drop=True)


def amostra(time: pd.Series, is_station_cell: pd.Series) -> np.ndarray:
    """0 = célula sem estação até 2024 (fora do treino), 1 = célula de estação em mês de treino até 2024,
    2 = célula de estação em mês de teste até 2024, 3 = 2025 (nunca visto)."""
    t = pd.to_datetime(time)
    novo = t.dt.year > STUDY_END_YEAR
    teste = t.dt.month.isin(TEST_MONTHS)
    est = is_station_cell.to_numpy(bool)
    out = np.where(est, np.where(teste, TESTE, TREINO), FORA)
    return np.where(novo, NOVO, out).astype("int8")


def fit_champions(data_dir, champion: dict[str, str], columns, *, sample_tag: str = "full",
                  seed: int = MODEL_SEED) -> dict:
    """Reajusta o campeão de cada trimestre só com `columns`. Devolve, por trimestre, o pipeline, o treino
    (para o pré-processamento ser ajustado nele) e as métricas de val/teste do reajuste."""
    out = {}
    for s, nome in champion.items():
        train, val, test = load_unit_frames(Path(data_dir), sample_tag, s)
        pipes = {}
        metr, _, seg = fit_arm(train, val, test, columns, select_regressors([nome]), seed=seed,
                               resid_models={nome}, fitted_out=pipes)
        if nome not in pipes:
            raise RuntimeError(f"{s}: o reajuste de {nome} não produziu modelo")
        out[s] = {"model": nome, "pipe": pipes[nome], "train": train[list(columns)], "metrics": metr, "seconds": seg}
    return out


def predict_grid(grid: pd.DataFrame, fitted: dict, columns) -> np.ndarray:
    """Previsão do modelo do trimestre de cada linha, truncada em [0, 80] m/s como no estudo."""
    pred = np.full(len(grid), np.nan, dtype="float32")
    for s, f in fitted.items():
        # linha com qualquer coluna faltando fica sem previsão: nada é imputado (regra do projeto)
        m = ((grid["season"] == s) & grid[list(columns)].notna().all(axis=1)
             & np.isfinite(grid[list(columns)].to_numpy(float)).all(axis=1)).to_numpy()
        if not m.any():
            continue
        _, x = preprocess_df(f["train"], grid.loc[m, list(columns)].reset_index(drop=True))
        pred[m] = np.clip(np.asarray(f["pipe"].predict(x), float).ravel(), *CLIP_RANGE)
    return pred


def to_dataset(grid: pd.DataFrame, attrs: dict | None = None) -> xr.Dataset:
    """(time, lat, lon) com ERA5 na hora do pico, a previsão, a diferença, a hora do pico e a amostra."""
    d = grid.assign(delta=grid["pred_gust_max"] - grid["era5_gust_max"],
                    hora_pico_utc=grid["hora_pico"].dt.hour.astype("int8"))
    ds = d.set_index([TIME, LAT, LON])[["era5_gust_max", "pred_gust_max", "delta", "hora_pico_utc",
                                        "amostra"]].to_xarray()
    ds["hora_pico_utc"] = ds["hora_pico_utc"].fillna(-1).astype("int8")
    ds["amostra"] = ds["amostra"].fillna(-1).astype("int8")
    for v, (un, desc) in {"era5_gust_max": ("m s-1", "gust10fg do ERA5 na hora do pico do dia"),
                          "pred_gust_max": ("m s-1", "rajada máxima diária corrigida (modelo do trimestre)"),
                          "delta": ("m s-1", "pred_gust_max - era5_gust_max"),
                          "hora_pico_utc": ("h", "hora UTC do pico de gust10fg (-1 = dia descartado)"),
                          "amostra": ("1", "0 fora do treino, 1 treino, 2 teste, 3 2025 (nunca visto), -1 sem dado")
                          }.items():
        ds[v].attrs.update(units=un, description=desc)
    ds.attrs.update(attrs or {})
    return ds


def write_netcdf(ds: xr.Dataset, path, complevel: int = 5) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": complevel} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    return path
