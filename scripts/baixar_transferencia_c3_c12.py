"""Cluster 3 → Cluster 12: baixa do volume do Modal o resultado do modelo TREINADO no
cluster 3 e TESTADO no cluster 12 e grava em `resultados_preliminares/` no schema de
`build_station_predictions_frame` (`src/pipelines/metrics_schema.py`).

Precisa do Modal autenticado (`modal setup`) com o perfil que tem o volume:

    MODAL_PROFILE=<perfil> uv run python scripts/baixar_transferencia_c3_c12.py \
        --volume <nome-do-volume> --caminho <pasta/ou/arquivo no volume>

Ou, se o arquivo já foi baixado:

    uv run python scripts/baixar_transferencia_c3_c12.py --local <arquivo.parquet|.csv>

Aceita parquet ou CSV com, no mínimo, `estacao`, `time`, `y_true` e `y_pred` (nomes
alternativos comuns são reconhecidos: `y_test`/`obs`/`target` e `pred`/`prediction`).
Saídas:
  07_cluster3_para_cluster12_teste.parquet           predição por (estação, dia, modelo)
  07_cluster3_para_cluster12_metricas.parquet        métricas por modelo × trimestre × estação
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.pipelines.common import compute_metrics  # noqa: E402

SAIDA = RAIZ / "resultados_preliminares"
SINONIMOS = {"y_true": ("y_true", "y_test", "obs", "target", "daily_wind_gust_max"),
             "y_pred": ("y_pred", "pred", "prediction", "y_hat"),
             "estacao": ("estacao", "station", "station_id"),
             "time": ("time", "date", "data")}
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def ler(caminho: Path) -> pd.DataFrame:
    arquivos = sorted(caminho.rglob("*")) if caminho.is_dir() else [caminho]
    partes = []
    for f in arquivos:
        if f.suffix == ".parquet":
            partes.append(pd.read_parquet(f))
        elif f.suffix == ".csv":
            partes.append(pd.read_csv(f))
    if not partes:
        raise SystemExit(f"nenhum parquet/csv em {caminho}")
    return pd.concat(partes, ignore_index=True)


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    for alvo, nomes in SINONIMOS.items():
        achado = next((n for n in nomes if n in df.columns), None)
        if achado is None:
            raise SystemExit(f"coluna {alvo!r} não encontrada (aceitos: {nomes}); colunas: {list(df.columns)}")
        df = df.rename(columns={achado: alvo})
    df["time"] = pd.to_datetime(df["time"])
    if "season" not in df.columns:
        df["season"] = df["time"].dt.month.map(SEASON)
    if "model" not in df.columns:
        df["model"] = "desconhecido"
    df["cluster_treino"], df["cluster_teste"] = 3, 12
    df["cluster_id"] = "12"
    df["split"] = df.get("split", "test")
    return df


def metricas(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for nivel, chaves in (("total", ["model"]), ("trimestre", ["model", "season"]),
                          ("estacao", ["model", "estacao"])):
        for k, d in df.groupby(chaves):
            k = k if isinstance(k, tuple) else (k,)
            y, p = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p)
            m = compute_metrics(y[ok], p[ok]) if ok.sum() > 1 else {}
            linha = {"nivel": nivel, **dict(zip(chaves, k)), "n": int(ok.sum()), **m}
            for ref in ("era5_proxy", "era5_gust_max"):
                if ref in d.columns:
                    linha["RMSE_era5"] = float(np.sqrt(np.nanmean((d[ref] - d.y_true) ** 2)))
            linhas.append(linha)
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume")
    ap.add_argument("--caminho")
    ap.add_argument("--local", type=Path)
    a = ap.parse_args()
    if a.local:
        bruto = ler(a.local)
    else:
        if not (a.volume and a.caminho):
            ap.error("use --local ou --volume + --caminho")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["modal", "volume", "get", a.volume, a.caminho, tmp], check=True)
            bruto = ler(Path(tmp))
    df = normalizar(bruto)
    SAIDA.mkdir(exist_ok=True)
    df.to_parquet(SAIDA / "07_cluster3_para_cluster12_teste.parquet", index=False, compression="zstd")
    met = metricas(df)
    met.to_parquet(SAIDA / "07_cluster3_para_cluster12_metricas.parquet", index=False, compression="zstd")
    print(met[met.nivel == "total"].to_string(index=False))


if __name__ == "__main__":
    main()
