"""RS/SC: o modelo do cluster 3 em regiões novas e modelos novos (por cluster, regional e regional-LOCO).

    uv run python scripts/treinar_rs_sc.py --stage all          # grade → base → treino → mapas
    uv run python scripts/treinar_rs_sc.py --stage treino       # só o treino (grade e base já prontas)

Etapas:
  grade   colunas selecionadas (sem `anor`) do ERA5 bruto `dataset/raw/era5_rs_sc`, uma linha por (célula, dia) no
          pico, para as células dos clusters 1–4 e das estações INMET → `rs_sc/_grade_diaria_pico.parquet`.
  base    linhas (estação, dia) com alvo INMET e o split do estudo → `rs_sc/base_estacoes.parquet`.
  treino  desenhos, todos avaliados nos meses de teste de 2000–2024 e em 2025 de cada cluster:
            A  o modelo do cluster 3 do estudo (Corrigido ML), sem treino novo;
            B  um modelo por cluster (treino só com as estações do cluster);
            C  um modelo regional (estações dos clusters 1–4 juntas);
            L  regional deixando um cluster inteiro de fora (LOCO) e avaliado nele.
          → `rs_sc/metricas.csv`, `rs_sc/params.json`, `rs_sc/previsoes_estacoes.parquet` (A, B do próprio
          cluster, C e L por estação-dia) e os modelos A, C e L em `rs_sc/modelos/`.
  mapas   aplica A, C e L à grade, um cluster por vez → `rs_sc/results_rs_sc_c<N>.nc`.
Modelos: os campeões do cluster 3 por trimestre, 49 colunas sem `anor`, + EQM ajustado na validação do treino.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.inferencia_selecao_features import LOCAL, MODAL, colunas_selecionadas  # noqa: E402
from src.feature_study.core.worker import load_unit_frames  # noqa: E402
from src.feature_study.data import era5_features as ef  # noqa: E402
from src.feature_study.data import station_dataset as sdb  # noqa: E402
from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics import quantile_map as qm  # noqa: E402
from src.feature_study.diagnostics import regional as rg  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402

RAW = RAIZ / "dataset/raw/era5_rs_sc"
INMET = RAIZ / "dataset/raw/INMET_Stratified.nc"
SHP = RAIZ / "dataset/shp/shp_vento.shp"
OUT = RAIZ / "artifacts/feature_study/rs_sc"
CLUSTERS = (1, 2, 3, 4)
GRADE = OUT / "_grade_diaria_pico.parquet"
BASE = OUT / "base_estacoes.parquet"


def colunas() -> list[str]:
    return [c for c in colunas_selecionadas() if not c.startswith("anor")]


def celulas_e_estacoes():
    lat, lon = ef.grid_of(RAW)
    poli = {c: ef.cells_in_cluster(SHP, c, lat, lon) for c in CLUSTERS}
    st = sdb.stations_of(INMET, SHP.parent, CLUSTERS, lat, lon)
    return poli, st


def etapa_grade(anos) -> None:
    poli, st = celulas_e_estacoes()
    cel = sorted(set().union(*poli.values()) | set(zip(st.grid_lat, st.grid_lon)))
    print(f"grade: {len(cel)} células ({ {c: len(v) for c, v in poli.items()} } no polígono), "
          f"{len(st)} estações", flush=True)
    partes = []
    for y in anos:
        t0 = time.time()
        partes.append(ef.build_daily_peak(RAW, cel, [y], columns=colunas(), static=()))
        print(f"  {y}: {len(partes[-1])} linhas em {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(partes, ignore_index=True).to_parquet(GRADE, index=False)


def etapa_base() -> None:
    _, st = celulas_e_estacoes()
    st.to_csv(OUT / "estacoes.csv", index=False)
    base = sdb.build(pd.read_parquet(GRADE), st, INMET, colunas())
    base.to_parquet(BASE, index=False)
    print(base.groupby(["cluster_id", "_split"]).size().unstack().to_string())


def modelo_do_estudo(cols) -> dict:
    """Desenho A: o Corrigido ML do cluster 3 (treino e EQM do estudo, sem nada de RS/SC além do cluster 3)."""
    fit = inf.fit_champions(LOCAL / "data", champions(MODAL / "summary"), cols)
    for s, f in fit.items():
        _, val, _ = load_unit_frames(LOCAL / "data", "full", s)
        f["mapa"] = qm.fit(inf.predict_grid(val, {s: f}, cols), val[TARGET_VAR])
    return fit


def etapa_treino() -> None:
    """Métricas de todos os desenhos nas linhas (estação, dia) e os modelos salvos para a etapa `mapas`.

    A grade NÃO é carregada aqui: com ~5 milhões de linhas, junto com os modelos, ela estourava a memória."""
    import joblib

    cols = colunas()
    champ = champions(MODAL / "summary")
    base = pd.read_parquet(BASE)
    metricas, params = [], {}
    pe = {k: np.full(len(base), np.nan, dtype="float32") for k in ("A", "B", "C", "L")}
    (OUT / "modelos").mkdir(parents=True, exist_ok=True)

    def registrar(desenho, treino, fit, alvo_cluster=None, salvar=None):
        b, m = rg.predict(base, fit, cols)
        mt = rg.evaluate(base, {"modelo": b, "Corrigido ML": m}, rotulo={"desenho": desenho, "treino": treino})
        if alvo_cluster is not None:
            mt = mt[mt.cluster == alvo_cluster]
        metricas.append(mt)
        pd.concat(metricas, ignore_index=True).to_csv(OUT / "metricas.csv", index=False)   # grava a cada desenho
        params[f"{desenho} | {treino}"] = {s: {"model": f["model"], "n_train": int(f.get("n_train", len(f["train"]))),
                                               "n_val": int(f.get("n_val", 0))} for s, f in fit.items()}
        if salvar:
            joblib.dump({s: {k: f[k] for k in ("model", "pipe", "train", "mapa")} for s, f in fit.items()},
                        OUT / "modelos" / f"{salvar}.joblib", compress=3)
        ganho = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")].pivot_table(
            index="cluster", columns="versao", values="RMSE")
        print(f"  {desenho} [{treino}] RMSE teste por cluster:\n{ganho.round(3).to_string()}", flush=True)
        return m

    t0 = time.time()
    pe["A"] = registrar("A: modelo do cluster 3 (estudo)", "3 (estudo)", modelo_do_estudo(cols), salvar="A")
    for c in CLUSTERS:
        mb = registrar("B: por cluster", str(c), rg.fit_set(base[base.cluster_id == c], cols, champ))
        pe["B"][(base.cluster_id == c).to_numpy()] = mb[(base.cluster_id == c).to_numpy()]
    pe["C"] = registrar("C: regional", "1+2+3+4", rg.fit_set(base, cols, champ), salvar="C")
    for c in CLUSTERS:
        outros = "+".join(str(x) for x in CLUSTERS if x != c)
        ml = registrar("L: regional sem o cluster avaliado", outros,
                       rg.fit_set(base[base.cluster_id != c], cols, champ), alvo_cluster=c, salvar=f"L{c}")
        pe["L"][(base.cluster_id == c).to_numpy()] = ml[(base.cluster_id == c).to_numpy()]
    print(f"treino concluído em {time.time() - t0:.0f}s", flush=True)

    (base[["estacao", "cluster_id", "time", "season", "_split", TARGET_VAR, "era5_gust_max"]]
     .assign(**{f"pred_{k}": v for k, v in pe.items()}).to_parquet(OUT / "previsoes_estacoes.parquet", index=False))
    (OUT / "params.json").write_text(json.dumps({"campeoes": champ, "colunas": cols, "modelos": params}, indent=1))


def etapa_mapas() -> None:
    """Aplica os modelos salvos (A, C e L do próprio cluster) à grade, um cluster por vez."""
    import gc

    import joblib
    import pyarrow.dataset as pads

    cols = colunas()
    champ = champions(MODAL / "summary")
    poli, _ = celulas_e_estacoes()
    fonte = pads.dataset(GRADE)
    for c in CLUSTERS:
        alvo = {(round(a, 3), round(b, 3)) for a, b in poli[c]}
        lats = sorted({float(a) for a, _ in poli[c]})
        g = fonte.to_table(filter=pads.field("lat").isin(lats),
                           columns=["lat", "lon", "time", "season", *cols]).to_pandas()
        g = g[[(round(a, 3), round(b, 3)) in alvo for a, b in zip(g.lat, g.lon)]].reset_index(drop=True)
        g[cols] = g[cols].astype("float32")
        out = {"era5_gust_max": g[inf.PEAK].astype("float32").to_numpy()}
        for k, nome in (("pred_A", "A"), ("pred_C", "C"), ("pred_L", f"L{c}")):
            fit = joblib.load(OUT / "modelos" / f"{nome}.joblib")
            out[k] = rg.predict(g, fit, cols)[1]
            del fit
            gc.collect()
        ds = g[["time", "lat", "lon"]].assign(**out).set_index(["time", "lat", "lon"]).to_xarray()
        for v, desc in {"era5_gust_max": "gust10fg do ERA5 na hora do pico do dia",
                        "pred_A": "Corrigido ML do cluster 3 (estudo), sem treino em RS/SC",
                        "pred_C": "Corrigido ML regional (clusters 1–4)",
                        "pred_L": f"Corrigido ML regional sem o cluster {c} (LOCO)"}.items():
            ds[v].attrs["description"] = desc
        ds.attrs.update({"cluster": c, "colunas": ",".join(cols), "campeoes": json.dumps(champ)})
        p = inf.write_netcdf(ds, OUT / f"results_rs_sc_c{c}.nc")
        print(f"gravado: {p} ({p.stat().st_size / 1e6:.1f} MB, {len(g)} linhas)", flush=True)
        del g, ds, out
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grade", "base", "treino", "mapas", "all"], default="all")
    ap.add_argument("--anos", default="2000-2025")
    a = ap.parse_args()
    i, f = (int(x) for x in a.anos.split("-"))
    if a.stage in ("grade", "all"):
        etapa_grade(range(i, f + 1))
    if a.stage in ("base", "all"):
        etapa_base()
    if a.stage in ("treino", "all"):
        etapa_treino()
    if a.stage in ("mapas", "all"):
        etapa_mapas()
