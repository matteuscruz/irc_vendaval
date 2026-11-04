"""Um NetCDF de predições por cluster: grade (mapas) + estações com o INMET (métricas), 2000–2025.

    uv run python scripts/exportar_predicoes_cluster.py

Entradas: `artifacts/feature_study/global/predicoes/predicoes_rajada_{rs_sc,c5,c12}.nc` (grade) e
`predicoes_estacoes.parquet` (estações), gerados por `exportar_predicoes_nc.py` e `exportar_predicoes_estacoes.py`.
Saída: `artifacts/feature_study/global/predicoes/por_cluster/predicoes_rajada_cluster<N>.nc`.
Descrição completa em `documentation/15_pipeline_do_modelo.md`, seção 9.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
P = RAIZ / "artifacts/feature_study/global/predicoes"
OUT = P / "por_cluster"
GRADE = {1: "rs_sc", 2: "rs_sc", 3: "rs_sc", 4: "rs_sc", 5: "c5", 12: "c12"}
MOD_GRADE = ["era5", "ml_local", "ml_global_trimestral", "ml_global_anual", "lstm_local", "lstm_global_anual"]
MOD_EST = ["era5", "ml_local", "ml_global_trimestral", "ml_global_anual", "lstm_local", "lstm_global_trimestral",
           "lstm_global_anual"]
CONJ = {"treino": 0, "validacao": 1, "teste": 2, "2025": 3}


def conjunto_por_dia(est: pd.DataFrame, tempo: pd.DatetimeIndex) -> np.ndarray:
    """O split é de calendário (igual para todas as estações): um código por dia; −1 = dia sem estação."""
    m = est.drop_duplicates("data").set_index("data").conjunto.map(CONJ)
    return m.reindex(tempo).fillna(-1).astype("int8").to_numpy()


def exportar(cl: int, est_all: pd.DataFrame) -> Path:
    g = xr.open_dataset(P / f"predicoes_rajada_{GRADE[cl]}.nc")
    msk = g.cluster_id == cl
    lat_ok = msk.any("lon").values
    lon_ok = msk.any("lat").values
    g = g.isel(lat=np.where(lat_ok)[0], lon=np.where(lon_ok)[0]).load()
    msk = g.cluster_id == cl
    ds = xr.Dataset(coords={"time": g.time, "lat": g.lat, "lon": g.lon})
    for v in MOD_GRADE:
        ds[v] = g[v].where(msk).astype("float32")
        ds[v].attrs = {**g[v].attrs, "units": "m s-1"}
    ds["celula_no_cluster"] = msk.astype("int8")
    ds["celula_no_cluster"].attrs = {"long_name": "1 = célula pertence ao cluster; 0 = fora (valores vazios)"}

    e = est_all[est_all.cluster_id == cl]
    tempo = pd.DatetimeIndex(g.time.values)
    meta = e.drop_duplicates("estacao").set_index("estacao")[["latitude", "longitude"]].sort_index()
    ds = ds.assign_coords(estacao=meta.index.to_numpy().astype(str),
                          estacao_lat=("estacao", meta.latitude.to_numpy()),
                          estacao_lon=("estacao", meta.longitude.to_numpy()))
    ds.estacao_lat.attrs = {"long_name": "latitude da estação INMET", "units": "degrees_north"}
    ds.estacao_lon.attrs = {"long_name": "longitude da estação INMET", "units": "degrees_east"}
    piv = e.set_index(["estacao", "data"])
    for v in ["inmet", *MOD_EST]:
        arr = (piv[v].unstack("data").reindex(index=meta.index, columns=tempo)).to_numpy("float32")
        ds[f"est_{v}"] = (("estacao", "time"), arr)
        txt = ("rajada máxima diária MEDIDA pelo INMET (referência)" if v == "inmet"
               else g[v].attrs["long_name"] if v in g else
               "LSTM global por trimestre: um modelo por trimestre, treinado com todos os clusters")
        ds[f"est_{v}"].attrs = {"units": "m s-1", "long_name": f"na estação: {txt}"}
    ds["conjunto"] = ("time", conjunto_por_dia(e, tempo))
    ds["conjunto"].attrs = {"long_name": "conjunto do dia nas estações",
                            "flag_values": "0 1 2 3 -1",
                            "flag_meanings": "treino validacao teste 2025 sem_dado",
                            "observacao": "métricas honestas só com conjunto 2 (teste) ou 3 (2025)"}
    ds.attrs = {"title": f"Rajada máxima diária prevista, cluster {cl}, 2000–2025",
                "cluster": cl,
                "conteudo": "variáveis (time, lat, lon): previsão na grade ERA5 0,25°; variáveis est_* (estacao, "
                            "time): previsões e rajada medida nas estações INMET do cluster",
                "documentacao": "15_pipeline_do_modelo.md, seção 9",
                "fonte": "irc_vendaval, scripts/exportar_predicoes_cluster.py"}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"predicoes_rajada_cluster{cl}.nc"
    enc = {v: {"zlib": True, "complevel": 5} for v in ds.data_vars}
    ds.to_netcdf(dest, encoding=enc)
    return dest


def main() -> None:
    est = pd.read_parquet(P / "predicoes_estacoes.parquet")
    est["data"] = pd.to_datetime(est.data)
    for cl in GRADE:
        d = exportar(cl, est)
        x = xr.open_dataset(d)
        print(f"{d.name}: {d.stat().st_size / 1e6:.0f} MB, grade {x.sizes['lat']}×{x.sizes['lon']} "
              f"({int(x.celula_no_cluster.sum())} células), {x.sizes['estacao']} estações")


if __name__ == "__main__":
    main()
