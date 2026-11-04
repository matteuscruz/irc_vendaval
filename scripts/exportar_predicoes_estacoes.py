"""Exporta as previsões de cada modelo nas estações INMET, uma linha por (estação, dia), para calcular métricas.

    uv run python scripts/exportar_predicoes_estacoes.py

Entradas (já gravadas em `artifacts/feature_study/global/`): `base_global.parquet`, `previsoes_ml.parquet`,
`previsoes_lstm.parquet` e `estacoes.csv`.

Saída: `artifacts/feature_study/global/predicoes/predicoes_estacoes.{parquet,csv}`. Colunas e uso em
`documentation/15_pipeline_do_modelo.md`. Para métricas honestas, use só `conjunto` em {"teste", "2025"}.
Também imprime a tabela de estações e frequência de vendaval por cluster usada no documento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
G = RAIZ / "artifacts/feature_study/global"
OUT = G / "predicoes"
TARGET = "daily_wind_gust_max"
VENDAVAL = 20.0  # m/s (72 km/h)
CONJUNTO = {"train": "treino", "val": "validacao", "test": "teste", "novo": "2025"}
# desenho → nome final (os mesmos dos .nc de grade); "B" = modelo local do próprio cluster da estação
MODELOS = [("ml", "B", "ml_local"), ("ml", "G", "ml_global_trimestral"), ("ml", "GA", "ml_global_anual"),
           ("lstm", "B", "lstm_local"), ("lstm", "G", "lstm_global_trimestral"), ("lstm", "GA", "lstm_global_anual")]


def montar() -> pd.DataFrame:
    base = pd.read_parquet(G / "base_global.parquet", columns=["row_id", "estacao", "cluster_id", "time", "season",
                                                              "_split", "janela_ok", TARGET, "era5_gust_max"])
    base = base[base.janela_ok].drop(columns="janela_ok").reset_index(drop=True)
    prev = {"ml": pd.read_parquet(G / "previsoes_ml.parquet"), "lstm": pd.read_parquet(G / "previsoes_lstm.parquet")}
    d = base.copy()
    for fam, ch, nome in MODELOS:
        p = prev[fam].set_index("row_id")
        if ch == "B":
            v = np.full(len(d), np.nan, dtype="float32")
            for cl in d.cluster_id.unique():
                k = (d.cluster_id == cl).to_numpy()
                v[k] = p[f"B{cl}_eqm"].reindex(d.row_id[k]).to_numpy()
        else:
            v = p[f"{ch}_eqm"].reindex(d.row_id).to_numpy()
        d[nome] = v.astype("float32")
    est = pd.read_csv(G / "estacoes.csv").drop_duplicates("estacao")[["estacao", "latitude", "longitude"]]
    d = d.merge(est, on="estacao", how="left")
    d = d.rename(columns={"time": "data", "season": "trimestre", TARGET: "inmet", "era5_gust_max": "era5"})
    d["conjunto"] = d["_split"].map(CONJUNTO)
    d["data"] = pd.to_datetime(d.data).dt.normalize()
    cols = ["estacao", "latitude", "longitude", "cluster_id", "data", "trimestre", "conjunto", "inmet", "era5",
            *[n for _, _, n in MODELOS]]
    d = d[cols].sort_values(["cluster_id", "estacao", "data"]).reset_index(drop=True)
    for c in ["inmet", "era5"]:
        d[c] = d[c].astype("float32")
    return d


def resumo_clusters(d: pd.DataFrame) -> pd.DataFrame:
    """Estações, estação-dias e frequência de vendaval (rajada INMET ≥ 20 m/s) por cluster."""
    return d.groupby("cluster_id").apply(lambda s: pd.Series({
        "estacoes": s.estacao.nunique(), "estacao_dias": len(s),
        "dias_vendaval": int((s.inmet >= VENDAVAL).sum()),
        "vendaval_por_1000_dias": 1000 * (s.inmet >= VENDAVAL).mean(),
        "P99_inmet": s.inmet.quantile(0.99), "media_inmet": s.inmet.mean(),
        "inicio": s.data.min().year, "fim": s.data.max().year}))


def main() -> None:
    d = montar()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "predicoes_estacoes.parquet", index=False)
    d.to_csv(OUT / "predicoes_estacoes.csv", index=False, float_format="%.3f", date_format="%Y-%m-%d")
    print(f"{len(d):,} linhas, {d.estacao.nunique()} estações → {OUT / 'predicoes_estacoes.parquet'} (+ .csv)")
    print("faltantes por coluna:", d.isna().sum()[lambda x: x > 0].to_dict())
    print(resumo_clusters(d).round(2).to_string())
    ok = d[d.conjunto == "teste"]
    rmse = lambda s, c: float(np.sqrt(np.mean((s[c] - s.inmet) ** 2)))  # noqa: E731
    print("RMSE no teste 2000–2024 por cluster:")
    print(ok.groupby("cluster_id").apply(lambda s: pd.Series({c: rmse(s, c) for c in
          ["era5", *[n for _, _, n in MODELOS]]})).round(2).to_string())


if __name__ == "__main__":
    main()
