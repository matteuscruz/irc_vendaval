"""Generalização: o modelo do cluster 3 (melhor por trimestre + colunas selecionadas) aplicado a OUTRO cluster.

    uv run python scripts/inferencia_cluster.py --cluster 12

1. Calcula as colunas selecionadas a partir do ERA5 bruto de `dataset/raw/new_features/cluster<N>/`
   (`src/feature_study/data/era5_features.py`, regras conferidas contra o cluster 3), para as células com centro
   dentro do polígono e para as células das estações INMET do cluster.
2. Reajusta o modelo padrão do cluster 3 (igual ao estudo) e o mapa de quantis (Empirical Quantile Mapping, EQM,
   ajustado na validação do cluster 3; ver `documentation/11_mapeamento_de_quantis.md`). O "Corrigido ML" dos
   notebooks é `pred_mapeado_sem_anor`: modelo sem as colunas `anor_*` + EQM.
3. Grava `artifacts/feature_study/inferencia_c<N>/results_cluster<N>.nc` com ERA5, padrão e mapeado, e
   `estacoes.csv` com as estações INMET do cluster e a célula de cada uma.
Nada do cluster N entra em treino ou calibração: é dado completamente novo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.inferencia_selecao_features import LOCAL, MODAL, colunas_selecionadas  # noqa: E402
from src.data.cluster_assigner import assign_station_clusters  # noqa: E402
from src.feature_study.core.worker import load_unit_frames  # noqa: E402
from src.feature_study.data import era5_features as ef  # noqa: E402
from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics import quantile_map as qm  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402

INMET = RAIZ / "dataset/raw/INMET_Stratified.nc"
SHP = RAIZ / "dataset/shp/shp_vento.shp"


def main(cluster: int, anos: list[int], grade_pronta: Path | None = None, raw: Path | None = None) -> None:
    raw = Path(raw) if raw else RAIZ / f"dataset/raw/new_features/cluster{cluster}"
    out_dir = RAIZ / f"artifacts/feature_study/inferencia_c{cluster}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = colunas_selecionadas()
    cols_sa = [c for c in cols if not c.startswith("anor")]
    com_anor = ef.has_static(raw, "anor")          # downloads sem `static/` (ex.: RS/SC) só têm a versão sem anor

    lat, lon = ef.grid_of(raw)
    poligono = ef.cells_in_cluster(SHP, cluster, lat, lon)

    ds_in = xr.open_dataset(INMET)
    cl = assign_station_clusters(ds_in, str(SHP.parent))
    est = cl[cl.cluster_id == cluster].estacao.tolist()
    st = pd.DataFrame({"estacao": est, "latitude": ds_in.latitude.sel(estacao=est).values,
                       "longitude": ds_in.longitude.sel(estacao=est).values})
    st["grid_lat"] = [float(lat[np.abs(lat - a).argmin()]) for a in st.latitude]
    st["grid_lon"] = [float(lon[np.abs(lon - b).argmin()]) for b in st.longitude]
    st["dist_celula_km"] = ef.haversine(st.latitude, st.longitude, st.grid_lat, st.grid_lon)
    st["dias_inmet"] = ds_in.daily_wind_gust_max.sel(estacao=est).notnull().sum("time").values
    st.to_csv(out_dir / "estacoes.csv", index=False)
    celulas = sorted(set(poligono) | set(zip(st.grid_lat, st.grid_lon)))
    print(f"cluster {cluster}: {len(poligono)} células no polígono, {len(st)} estações INMET, "
          f"{len(celulas)} células calculadas", flush=True)

    cache = out_dir / "_grade_diaria_pico.parquet"
    if grade_pronta is not None:
        # grade já calculada por outro caminho (cluster 3: test_inference_cluster_3.parquet; RS/SC: a grade
        # compartilhada dos clusters 1–4). Fica só com as células deste cluster.
        grade = pd.read_parquet(grade_pronta)
        alvo = {(round(a, 3), round(b, 3)) for a, b in celulas}
        grade = grade[[(round(a, 3), round(b, 3)) in alvo for a, b in zip(grade.lat, grade.lon)]].reset_index(drop=True)
    elif cache.exists():
        grade = pd.read_parquet(cache)
    else:
        partes = []
        for y in anos:
            t0 = time.time()
            partes.append(ef.build_daily_peak(raw, celulas, [y], columns=cols if com_anor else cols_sa,
                                              static=("anor",) if com_anor else ()))
            print(f"  {y}: {len(partes[-1])} linhas em {time.time() - t0:.0f}s", flush=True)
        grade = pd.concat(partes, ignore_index=True)
        grade.to_parquet(cache, index=False)
    com_anor = com_anor or all(c in grade.columns for c in cols if c.startswith("anor"))
    faltam = [c for c in (cols if com_anor else cols_sa) if c not in grade.columns]
    if faltam:
        raise SystemExit(f"colunas não calculadas: {faltam}")

    def ajustar(colunas):
        """Modelo padrão do cluster 3 com `colunas` + mapa de quantis na validação do cluster 3."""
        fit_ = inf.fit_champions(LOCAL / "data", champions(MODAL / "summary"), colunas)
        mapas = {}
        for s, f in fit_.items():
            _, val, _ = load_unit_frames(LOCAL / "data", "full", s)
            mapas[s] = qm.fit(inf.predict_grid(val, {s: f}, colunas), val[TARGET_VAR])
        bruto_ = inf.predict_grid(grade, fit_, colunas)
        mapeado_ = np.full(len(grade), np.nan, dtype="float32")
        for s, mapa in mapas.items():
            m = (grade["season"] == s).to_numpy()
            mapeado_[m] = qm.apply(bruto_[m], mapa)
        return fit_, bruto_, mapeado_

    # sem `anor`: no cluster 3 o anor identifica a estação, e em região nova leva o MLP do DJF a +12 m/s.
    # As mesmas colunas sem ele são o arm `sel__val12_sem_anor` do estudo (o "Corrigido ML").
    fit_sa, bruto_sa, mapeado_sa = ajustar(cols_sa)
    if com_anor:
        fit, bruto, mapeado = ajustar(cols)
    else:
        fit, bruto, mapeado = fit_sa, None, None
        print("  sem static/anor: só a versão sem anor (Corrigido ML)", flush=True)
    for s, f in fit_sa.items():
        m_ = f["metrics"]
        rmse = float(m_[m_["split"] == "test"].RMSE.iloc[0])
        print(f"  sem anor {s} {f['model']}: RMSE teste do reajuste no cluster 3 {rmse:.3f}", flush=True)

    # fora da faixa do treino (cluster 3): fração de linhas com alguma coluna fora de [mín, máx]
    cf = cols if com_anor else cols_sa
    tr = pd.concat([f["train"] for f in fit.values()])
    lo_, hi_ = tr[cf].min(), tr[cf].max()
    fora = ((grade[cf] < lo_) | (grade[cf] > hi_))
    fora.mean().rename("frac_fora_do_treino").sort_values(ascending=False).to_csv(out_dir / "fora_da_faixa.csv")

    pc = set(poligono)
    preds = {"pred_padrao_sem_anor": bruto_sa, "pred_mapeado_sem_anor": mapeado_sa}
    if com_anor:
        preds = {"pred_padrao": bruto, "pred_mapeado": mapeado, **preds}
    g = grade.assign(era5_gust_max=grade[inf.PEAK].astype("float32"), **preds,
                     hora_pico_utc=grade.hora_pico.dt.hour.astype("int8"))
    ds = g.set_index(["time", "lat", "lon"])[["era5_gust_max", *preds, "hora_pico_utc"]].to_xarray()
    mask = xr.DataArray(np.array([[(a, b) in pc for b in ds.lon.values] for a in ds.lat.values]),
                        coords={"lat": ds.lat, "lon": ds.lon}, dims=("lat", "lon"))
    ds["no_poligono"] = mask.astype("int8")
    ds["hora_pico_utc"] = ds.hora_pico_utc.fillna(-1).astype("int8")
    for v, desc in {"era5_gust_max": "gust10fg do ERA5 na hora do pico do dia",
                    "pred_padrao": "modelo do cluster 3 (padrão), colunas selecionadas",
                    "pred_mapeado": "padrão + mapa de quantis (validação do cluster 3)",
                    "pred_padrao_sem_anor": "padrão sem as colunas anor_* (arm sel__val12_sem_anor)",
                    "pred_mapeado_sem_anor": "padrão sem anor + mapa de quantis (validação do cluster 3)",
                    "no_poligono": "1 = centro da célula no polígono do cluster; 0 = só célula de estação"}.items():
        if v in ds:
            ds[v].attrs["description"] = desc
    ds.attrs.update({"cluster": cluster, "modelos": json.dumps({s: f["model"] for s, f in fit.items()}),
                     "treino": "cluster 3, 2000–2024, meses de treino; nada do cluster alvo",
                     "colunas": ",".join(cf)})
    path = inf.write_netcdf(ds, out_dir / f"results_cluster{cluster}.nc")
    print(f"gravado: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("linhas com alguma coluna fora da faixa do treino:", f"{float(fora.any(axis=1).mean()):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--anos", default="2000-2025")
    ap.add_argument("--grade", type=Path, default=None,
                    help="parquet de linhas (célula, dia) no pico já calculado; pula o cálculo a partir do bruto")
    ap.add_argument("--raw", type=Path, default=None,
                    help="pasta do ERA5 bruto (padrão: dataset/raw/new_features/cluster<N>)")
    a = ap.parse_args()
    i, f = (int(x) for x in a.anos.split("-"))
    main(a.cluster, list(range(i, f + 1)), a.grade, a.raw)
