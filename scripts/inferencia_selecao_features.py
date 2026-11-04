"""Inferência de sanidade: melhor modelo por trimestre + colunas selecionadas, aplicados à grade do cluster 3.

    uv run python scripts/inferencia_selecao_features.py

Gera `artifacts/feature_study/inferencia/results_feature_selction.nc` e imprime:
  - o RMSE do reajuste contra o do Modal (tem de bater; senão para);
  - a distribuição de ERA5 e da previsão por trimestre e amostra;
  - ERA5 e previsão contra o INMET nas células das estações (meses de teste até 2024 e 2025);
  - quantas células têm `anor_r75_std` fora do intervalo visto no treino.
E salva o mapa da correção média por trimestre ao lado do .nc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402

LOCAL = RAIZ / "artifacts/feature_study/cluster3_groups"
MODAL = RAIZ / "artifacts/feature_study/cluster3_groups_modal"
GRADE = RAIZ / "dataset/raw/test_inference_cluster_3.parquet"
GRUPO4 = RAIZ / "dataset/raw/feature_study_cluster_3/features_grupo4_cluster3.parquet"
INMET = RAIZ / "dataset/raw/INMET_Stratified.nc"
SAIDA = RAIZ / "artifacts/feature_study/inferencia/results_feature_selction.nc"
ARM = "sel__val12"
TOLERANCIA = 0.05


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def colunas_selecionadas() -> list[str]:
    """As colunas do arm, na ordem do arm; confere com `config/features_maping.json`."""
    arm = next(a for a in json.loads((LOCAL / "data/arms.json").read_text()) if a["name"] == ARM)
    mapa = json.loads((RAIZ / "config/features_maping.json").read_text())
    do_mapa = {c for g in mapa["colunas_por_familia"].values() for f in g.values() for c in f}
    if set(arm["features"]) != do_mapa:
        raise SystemExit(f"features_maping.json difere do arm {ARM}: {sorted(set(arm['features']) ^ do_mapa)}")
    return list(arm["features"])


def rmse_modal(season: str, model: str) -> float:
    m = pd.read_parquet(MODAL / f"units/full/metrics__{season}__{ARM}.parquet")
    return float(m[(m.model == model) & (m.split == "test")].RMSE.iloc[0])


def main(tolerancia: float) -> None:
    cols = colunas_selecionadas()
    campeao = champions(MODAL / "summary")
    print(f"{len(cols)} colunas | campeões: {campeao}")

    fitted = inf.fit_champions(LOCAL / "data", campeao, cols)
    for s, f in fitted.items():
        meu = float(f["metrics"].query("split == 'test'").RMSE.iloc[0])
        ref = rmse_modal(s, f["model"])
        print(f"  {s} {f['model']:<30} RMSE teste reajuste {meu:.3f} | Modal {ref:.3f} | {f['seconds']:.0f}s")
        if abs(meu - ref) > tolerancia:
            raise SystemExit(f"{s}: reajuste não reproduz o Modal (|{meu:.3f} - {ref:.3f}| > {tolerancia})")

    print("lendo a grade (por grupo de linhas)...", flush=True)
    grade = inf.daily_peak_rows(GRADE, cols)
    print(f"  {len(grade)} linhas (célula × dia), {grade.groupby(['lat', 'lon']).ngroups} células")

    g4 = pd.read_parquet(GRUPO4, columns=["codigo_estacao", "grid_lat", "grid_lon"])
    celulas_est = set(zip(g4.grid_lat.round(2), g4.grid_lon.round(2)))
    est_cell = pd.Series([(round(a, 2), round(b, 2)) in celulas_est for a, b in zip(grade.lat, grade.lon)])
    grade["amostra"] = inf.amostra(grade["time"], est_cell)
    grade["era5_gust_max"] = grade[inf.PEAK].astype("float32")
    grade["pred_gust_max"] = inf.predict_grid(grade, fitted, cols)
    falt = grade[cols].isna()
    print(f"  {int(falt.any(axis=1).sum())} linhas sem previsão por coluna faltando (nada é imputado); "
          f"colunas: {falt.sum()[falt.sum() > 0].to_dict()}")

    # anor fora do intervalo do treino
    tr = pd.concat([f["train"] for f in fitted.values()])
    lo, hi = tr["anor_r75_std"].min(), tr["anor_r75_std"].max()
    cel = grade.groupby(["lat", "lon"])["anor_r75_std"].first()
    print(f"\nanor_r75_std no treino: [{lo:.3g}, {hi:.3g}] | células fora: {int(((cel < lo) | (cel > hi)).sum())} de {len(cel)}")

    attrs = {"titulo": "Inferência de sanidade: melhor modelo por trimestre + colunas selecionadas (sel__val12)",
             "modelos": json.dumps({s: f["model"] for s, f in fitted.items()}),
             "rmse_teste_reajuste": json.dumps({s: round(float(f["metrics"].query("split == 'test'").RMSE.iloc[0]), 3)
                                                for s, f in fitted.items()}),
             "colunas": ",".join(cols), "seed": 42, "regra_modelo": "rank 1 de summary/top_models.json",
             "desenho": "uma linha por célula e dia UTC, na hora de maior gust10fg; previsão = rajada máxima diária",
             "fonte": GRADE.name}
    out = inf.write_netcdf(inf.to_dataset(grade, attrs), SAIDA)
    print(f"\ngravado: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    nomes = {inf.FORA: "fora (sem estação)", inf.TREINO: "treino", inf.TESTE: "teste", inf.NOVO: "2025"}
    r = (grade.assign(amostra=grade.amostra.map(nomes))
         .groupby(["season", "amostra"])
         .agg(n=("pred_gust_max", "size"), era5_media=("era5_gust_max", "mean"), pred_media=("pred_gust_max", "mean"),
              era5_p99=("era5_gust_max", lambda x: x.quantile(.99)), pred_p99=("pred_gust_max", lambda x: x.quantile(.99))))
    r["vies_pred_menos_era5"] = r.pred_media - r.era5_media
    print("\nDistribuição por trimestre e amostra (m/s):\n" + r.round(2).to_string())

    # contra o INMET nas células das estações
    obs = xr.open_dataset(INMET)["daily_wind_gust_max"].to_dataframe().reset_index()
    obs = obs[obs.estacao.isin(g4.codigo_estacao)].dropna()
    obs = obs.merge(g4.rename(columns={"codigo_estacao": "estacao", "grid_lat": "lat", "grid_lon": "lon"}), on="estacao")
    obs["lat"], obs["lon"] = obs.lat.round(2), obs.lon.round(2)
    gm = grade.assign(lat=grade.lat.round(2), lon=grade.lon.round(2))
    j = obs.merge(gm[["lat", "lon", "time", "season", "amostra", "era5_gust_max", "pred_gust_max"]], on=["lat", "lon", "time"])
    linhas = []
    for rot, m in [("teste 2000–2024", j.amostra == inf.TESTE), ("2025 (nunca visto)", j.amostra == inf.NOVO),
                   ("treino (dentro da amostra)", j.amostra == inf.TREINO)]:
        for s in ("Todos", "DJF", "MAM", "JJA", "SON"):
            d = j[m] if s == "Todos" else j[m & (j.season == s)]
            if d.empty:
                continue
            y = d.daily_wind_gust_max
            linhas.append({"recorte": rot, "trimestre": s, "n": len(d), "estacoes": d.estacao.nunique(),
                           "RMSE_ERA5": _rmse(d.era5_gust_max, y), "RMSE_pred": _rmse(d.pred_gust_max, y),
                           "vies_ERA5": float((d.era5_gust_max - y).mean()), "vies_pred": float((d.pred_gust_max - y).mean())})
    t = pd.DataFrame(linhas)
    t["ganho_RMSE"] = t.RMSE_ERA5 - t.RMSE_pred
    print("\nContra o INMET nas células das estações (ganho > 0 = previsão melhor que o ERA5):\n" + t.round(3).to_string(index=False))
    t.to_csv(SAIDA.with_name("sanidade_vs_inmet.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = xr.open_dataset(out)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    lim = float(np.nanpercentile(np.abs(ds.delta.groupby("time.season").mean().values), 99))
    for ax, s in zip(axs, ["DJF", "MAM", "JJA", "SON"]):
        m = ds.delta.where(ds.time.dt.season == s).mean("time")
        im = ax.pcolormesh(ds.lon, ds.lat, m, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        ax.scatter(g4.grid_lon, g4.grid_lat, s=14, c="k", marker="^")
        ax.set_title(f"{s}: correção média (pred − ERA5)")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axs, label="m/s")
    fig.savefig(SAIDA.with_name("mapa_correcao_media.png"), dpi=120)
    print(f"mapa: {SAIDA.with_name('mapa_correcao_media.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerancia", type=float, default=TOLERANCIA)
    main(ap.parse_args().tolerancia)
