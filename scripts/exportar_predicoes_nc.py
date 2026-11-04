"""Exporta as previsões de rajada na grade, 2000–2025, em NetCDF: um arquivo por região, uma variável por modelo.

    uv run python scripts/exportar_predicoes_nc.py

Entradas (já gravadas):
- `artifacts/feature_study/global/mapa_ml_anual_<regiao>.nc`: ERA5 e os modelos de ML (`scripts/mapa_extrapolacao.py`);
- `artifacts/feature_study/global/lstm_grade_anos/<ano>.parquet`: as LSTMs (Modal, `global_lstm.py --stage grade-anos`).

Saída: `artifacts/feature_study/global/predicoes/predicoes_rajada_<regiao>.nc`, dimensões (time, lat, lon), em m/s:
`era5`, `ml_local`, `ml_global_trimestral`, `ml_global_anual`, `lstm_local`, `lstm_global_anual` e `cluster_id`.
Cada valor é a rajada máxima diária prevista na célula (na hora do pico do ERA5). Todos os modelos já incluem o
mapeamento de quantis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
G = RAIZ / "artifacts/feature_study/global"
OUT = G / "predicoes"

REGIOES = {"rs_sc": (1, 2, 3, 4), "c12": (12,), "c5": (5,)}
# variável no arquivo de ML → nome final
ML = {"era5_gust_max": "era5", "ml_local_eqm": "ml_local", "ml_global_trim_eqm": "ml_global_trimestral",
      "ml_anual_eqm": "ml_global_anual"}
LSTM = {"ls_local": "lstm_local", "ls_anual": "lstm_global_anual"}
DESCRICAO = {
    "era5": "rajada do ERA5 (gust10fg) na hora do pico diário, sem correção",
    "ml_local": "ML local: um modelo por cluster e trimestre, treinado só com as estações do cluster",
    "ml_global_trimestral": "ML global por trimestre: um modelo por trimestre, treinado com todos os clusters",
    "ml_global_anual": "ML global anual: um único modelo (CatBoost) para todos os clusters e o ano todo",
    "lstm_local": "LSTM local: um modelo por cluster e trimestre, treinado só com as estações do cluster",
    "lstm_global_anual": "LSTM global anual: um único modelo para todos os clusters e o ano todo",
}


def exportar(regiao: str, lstm: pd.DataFrame) -> Path:
    ml = xr.open_dataset(G / f"mapa_ml_anual_{regiao}.nc")[list(ML)].rename(ML).load()
    ml["time"] = pd.to_datetime(ml.time.values).normalize()
    lat, lon = ml.lat.values, ml.lon.values
    sel = lstm[lstm.lat.round(3).isin(np.round(lat, 3)) & lstm.lon.round(3).isin(np.round(lon, 3))
               & lstm.cluster_id.isin(REGIOES[regiao])].copy()
    sel["lat"], sel["lon"] = sel.lat.round(3), sel.lon.round(3)
    ml = ml.assign_coords(lat=np.round(lat, 3), lon=np.round(lon, 3))
    ls = sel.set_index(["time", "lat", "lon"])[list(LSTM)].rename(columns=LSTM).to_xarray()
    ds = xr.merge([ml, ls.reindex_like(ml)])
    cl = sel.drop_duplicates(["lat", "lon"]).set_index(["lat", "lon"]).cluster_id.to_xarray().reindex_like(ml)
    ds["cluster_id"] = cl.fillna(-1).astype("int16")
    for v, txt in DESCRICAO.items():
        ds[v] = ds[v].astype("float32")
        ds[v].attrs = {"units": "m s-1", "long_name": txt}
    ds["cluster_id"].attrs = {"long_name": "cluster de vento da célula (-1 = fora dos clusters)"}
    ds.attrs = {"title": f"Rajada máxima diária prevista na grade ERA5, região {regiao}, 2000–2025",
                "observacao": "dias de 2000–2024 nos meses de treino/validação foram vistos pelos modelos nas "
                              "células com estação; jan/abr/jul/out de 2000–2024 e 2025 não foram",
                "fonte": "irc_vendaval, scripts/exportar_predicoes_nc.py"}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"predicoes_rajada_{regiao}.nc"
    enc = {v: {"zlib": True, "complevel": 5} for v in ds.data_vars}
    ds.to_netcdf(dest, encoding=enc)
    return dest


def main() -> None:
    arqs = sorted((G / "lstm_grade_anos").glob("*.parquet"))
    if len(arqs) < 26:
        sys.exit(f"só {len(arqs)} anos da LSTM em {G / 'lstm_grade_anos'}: rode `global_lstm.py --stage grade-anos`")
    lstm = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
    lstm["time"] = pd.to_datetime(lstm.time).dt.normalize()
    for regiao in REGIOES:
        dest = exportar(regiao, lstm)
        print(f"gravado: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
