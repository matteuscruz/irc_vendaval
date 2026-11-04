"""Mapa de extrapolação: o ML global anual (CatBoost, 1 modelo para os clusters 1–4 e 12 e o ano todo) + EQM
aplicado a TODAS as células da grade, inclusive onde não há estação.

    uv run python scripts/mapa_extrapolacao.py

Reajusta o modelo exatamente como em `treinar_global.py --stage ml-anual` (mesma base, seed 42) e prevê as grades
diárias no pico já calculadas (RS/SC: `rs_sc/_grade_diaria_pico.parquet`; cluster 12:
`inferencia_c12/_grade_diaria_pico.parquet`). Salva os modelos em `artifacts/feature_study/global/modelos_ml/` (joblib: pipeline, treino para o escalonador e mapa
EQM, por trimestre ou `ALL`). Grava `artifacts/feature_study/global/mapa_ml_anual_<regiao>.nc`
(time, lat, lon) com `era5_gust_max`, `ml_anual` e `ml_anual_eqm`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.treinar_global import BASE, OUT, REGIOES, colunas  # noqa: E402
from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics import regional as rg  # noqa: E402


def _grade(reg, cols):
    g = pd.read_parquet(REGIOES[reg][1], columns=["lat", "lon", "time", "season", *cols])
    g[cols] = g[cols].astype("float32")
    return g.dropna(subset=cols).reset_index(drop=True)


def _ajustar_e_salvar(cols, cols_a, dst) -> None:
    import gc

    import joblib

    from scripts.inferencia_selecao_features import MODAL
    from src.feature_study.diagnostics.predictor_curves import champions

    champ = champions(MODAL / "summary")
    base = pd.read_parquet(BASE)
    base = base[base.janela_ok].reset_index(drop=True)
    so_pipe = lambda f: {k: {kk: v[kk] for kk in ("model", "pipe", "train", "mapa")} for k, v in f.items()}  # noqa: E731
    dst.mkdir(parents=True, exist_ok=True)
    tarefas = [("ml_global_anual", lambda: rg.fit_single(rg.with_doy(base), cols_a)),
               ("ml_global_trimestre", lambda: rg.fit_set(base, cols, champ)),
               *[(f"ml_local_c{c}", (lambda c=c: rg.fit_set(base[base.cluster_id == c], cols, champ))) for c in rg.GLOBAL]]
    for nome, f in tarefas:
        if (dst / f"{nome}.joblib").exists():
            continue
        joblib.dump(so_pipe(f()), dst / f"{nome}.joblib", compress=3)
        gc.collect()
        print(f"salvo: {nome}", flush=True)


def main() -> None:
    import gc
    import json

    import joblib

    from src.feature_study.data import era5_features as ef

    cols = colunas()
    cols_a = [*cols, *rg.DOY_COLS]
    dst = OUT / "modelos_ml"
    _ajustar_e_salvar(cols, cols_a, dst)
    SHP = Path(__file__).resolve().parents[1] / "dataset/shp/shp_vento.shp"
    for reg, (raw, grade_p, clus) in REGIOES.items():
        lat, lon = ef.grid_of(raw)
        cel_de = {c: {(round(a, 3), round(b, 3)) for a, b in (ef.points_of_cluster(raw, c) if Path(raw).is_file()
                                                              else ef.cells_in_cluster(SHP, c, lat, lon))} for c in clus}
        anos = sorted(pd.read_parquet(grade_p, columns=["time"]).time.dt.year.unique())
        partes = []
        for y in anos:                                   # um ano por vez: a grade inteira não cabe com os modelos
            g = pd.read_parquet(grade_p, columns=["lat", "lon", "time", "season", *cols],
                                filters=[("time", ">=", pd.Timestamp(f"{y}-01-01")), ("time", "<", pd.Timestamp(f"{y + 1}-01-01"))])
            g[cols] = g[cols].astype("float32")
            g = g.dropna(subset=cols).reset_index(drop=True)
            g = g[[x in set().union(*cel_de.values()) for x in zip(g.lat.round(3), g.lon.round(3))]].reset_index(drop=True)
            out = {"era5_gust_max": g["gust10fg"].to_numpy("float32")}
            f = joblib.load(dst / "ml_global_anual.joblib")
            out["ml_anual"], out["ml_anual_eqm"] = rg.predict(rg.with_doy(g).assign(season=rg.ANUAL), f, cols_a)
            f = joblib.load(dst / "ml_global_trimestre.joblib")
            out["ml_global_trim"], out["ml_global_trim_eqm"] = rg.predict(g, f, cols)
            lb, le = np.full(len(g), np.nan, "float32"), np.full(len(g), np.nan, "float32")
            gl = list(zip(g.lat.round(3), g.lon.round(3)))
            for c in clus:
                m = np.array([x in cel_de[c] for x in gl])
                if m.any():
                    f = joblib.load(dst / f"ml_local_c{c}.joblib")
                    lb[m], le[m] = rg.predict(g[m].reset_index(drop=True), f, cols)
            out["ml_local"], out["ml_local_eqm"] = lb, le
            del f
            partes.append(g[["time", "lat", "lon"]].assign(**out))
            del g
            gc.collect()
            print(f"  {reg} {y}", flush=True)
        ds = pd.concat(partes, ignore_index=True).set_index(["time", "lat", "lon"]).to_xarray()
        ds.attrs.update({"regiao": reg, "modelos": json.dumps({"ml_anual": "CatBoost global anual",
                         "ml_global_trim": "campeões por trimestre, todos os clusters",
                         "ml_local": "campeões por trimestre, treino só no cluster"})})
        p = inf.write_netcdf(ds, OUT / f"mapa_ml_anual_{reg}.nc")
        print(f"gravado: {p} ({p.stat().st_size / 1e6:.0f} MB)", flush=True)
        del ds, partes
        gc.collect()


if __name__ == "__main__":
    main()
