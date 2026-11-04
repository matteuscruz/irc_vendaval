"""ML e LSTM nos mesmos desenhos (local, regional RS/SC, global e LOCO global), clusters 1–4 e 12.

    uv run python scripts/treinar_global.py --stage base      # base comum (RS/SC + cluster 12), split de calendário
    uv run python scripts/treinar_global.py --stage janelas   # janelas de 24 h para a LSTM (~1 h, em disco)
    uv run python scripts/treinar_global.py --stage ml        # todos os desenhos do ML (~30 min)
    uv run python scripts/treinar_global.py --stage ml-anual  # só o GA (modelo único do ano todo), sem refazer o resto
    # LSTM no Modal (GPU, pago): ver src/modal/global_lstm.py (upload → fit → download)
    uv run python scripts/treinar_global.py --stage avaliar   # métricas das duas famílias, lado a lado

Desenhos (`regional.designs`): B local (cada cluster), C regional RS/SC (1–4), G global (1–4 + 12), L global sem o
cluster avaliado, GA global anual (UM modelo para todos os trimestres, com o dia do ano como coluna, e um EQM só;
no ML, um CatBoost). O ML tem também o A (o modelo do cluster 3 do estudo, sem treino novo). Mesmas linhas para as
duas famílias: só (estação, dia) com janela de 24 h completa (`janela_ok`). Avaliação nos meses de teste de
2000–2024 e em 2025. Saídas em `artifacts/feature_study/global/`.
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

from scripts.inferencia_selecao_features import MODAL, colunas_selecionadas  # noqa: E402
from scripts.treinar_rs_sc import modelo_do_estudo  # noqa: E402
from src.feature_study.core.worker_lstm import valid_rows  # noqa: E402
from src.feature_study.data import era5_features as ef  # noqa: E402
from src.feature_study.data import station_dataset as sdb  # noqa: E402
from src.feature_study.diagnostics import regional as rg  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402

INMET = RAIZ / "dataset/raw/INMET_Stratified.nc"
SHP = RAIZ / "dataset/shp/shp_vento.shp"
A = RAIZ / "artifacts/feature_study"
OUT = A / "global"
BASE = OUT / "base_global.parquet"
JANELAS = OUT / "janelas.npy"
JANELA = 24
# região → (ERA5 bruto, grade diária no pico já calculada, clusters)
REGIOES = {
    "rs_sc": (RAIZ / "dataset/raw/era5_rs_sc", A / "rs_sc/_grade_diaria_pico.parquet", rg.RS_SC),
    "c12": (RAIZ / "dataset/raw/new_features/cluster12", A / "inferencia_c12/_grade_diaria_pico.parquet", (12,)),
    # arquivo por pontos: só as células dos polígonos, sem o oceano; vizinhança só com os pontos disponíveis
    "c5": (RAIZ / "dataset/raw/new_features/cluster5/vendaval_val12_global.nc",
           A / "inferencia_c5/_grade_diaria_pico.parquet", (5,)),
}
CHAVES_BASE = ["row_id", "estacao", "cluster_id", "regiao", "grid_lat", "grid_lon", "time", "season", "_split",
               "janela_ok", TARGET_VAR, "era5_gust_max"]


def colunas() -> list[str]:
    return [c for c in colunas_selecionadas() if not c.startswith("anor")]


def etapa_base() -> None:
    cols, split, partes, ests = colunas(), sdb.calendar_split(), [], []
    for reg, (raw, grade, clusters) in REGIOES.items():
        lat, lon = ef.grid_of(raw)
        st = sdb.stations_of(INMET, SHP.parent, clusters, lat, lon)
        b = sdb.build(pd.read_parquet(grade), st, INMET, cols, split=split)
        b["regiao"] = reg
        partes.append(b)
        ests.append(st.assign(regiao=reg))
    base = pd.concat(partes, ignore_index=True)
    base["row_id"] = np.arange(len(base))
    base["janela_ok"] = False                                     # preenchido na etapa `janelas`
    OUT.mkdir(parents=True, exist_ok=True)
    base[[*CHAVES_BASE, *cols]].to_parquet(BASE, index=False)
    pd.concat(ests, ignore_index=True).to_csv(OUT / "estacoes.csv", index=False)
    print(base.groupby(["cluster_id", "_split"]).size().unstack().to_string())


def etapa_adiciona_c5(anos) -> None:
    """Acrescenta o cluster 5 à base e às janelas sem refazer as outras regiões: as linhas antigas mantêm o
    `row_id` e as janelas. Idempotente: linhas do cluster 5 já presentes são refeitas."""
    from src.feature_study.diagnostics.inference import SEASON_OF_MONTH

    raw, grade_p, _ = REGIOES["c5"]
    cols, split = colunas(), sdb.calendar_split()
    lat, lon = ef.grid_of(raw)
    pontos = np.array(ef.points_of_cluster(raw, 5))
    ds = xr.open_dataset(raw)
    disp = np.column_stack([np.round(ds.lat.values.astype(float), 3), np.round(ds.lon.values.astype(float), 3)])
    st = sdb.stations_of(INMET, SHP.parent, (5,), lat, lon)
    d = np.array([ef.haversine(a, b, disp[:, 0], disp[:, 1]) for a, b in zip(st.latitude, st.longitude)])
    # estação cuja célula mais próxima não está no arquivo: a célula disponível mais próxima
    st["grid_lat"], st["grid_lon"] = disp[d.argmin(1), 0], disp[d.argmin(1), 1]
    st["dist_celula_km"] = d.min(1)
    print(st[["estacao", "grid_lat", "grid_lon", "dist_celula_km"]].round(2).to_string(), flush=True)
    cel = sorted({(float(a), float(b)) for a, b in pontos} | set(zip(st.grid_lat, st.grid_lon)))
    if not grade_p.exists():
        partes = []
        for y in anos:
            t0 = time.time()
            partes.append(ef.daily_peak_year(raw, cel, y))
            print(f"  grade c5 {y}: {time.time() - t0:.0f}s", flush=True)
        g = pd.concat(partes, ignore_index=True)
        g["season"] = g.time.dt.month.map(SEASON_OF_MONTH)
        grade_p.parent.mkdir(parents=True, exist_ok=True)
        g[["lat", "lon", "time", "season", *cols]].to_parquet(grade_p, index=False)
    novas = sdb.build(pd.read_parquet(grade_p), st, INMET, cols, split=split)
    novas["regiao"] = "c5"
    base = pd.read_parquet(BASE)
    base = base[base.cluster_id != 5]
    n0 = len(base)
    novas["row_id"] = np.arange(n0, n0 + len(novas))
    novas["janela_ok"] = False
    tudo = pd.concat([base, novas[base.columns]], ignore_index=True)
    velho = np.load(JANELAS, mmap_mode="r")
    tmp = JANELAS.with_suffix(".tmp.npy")
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=(len(tudo), JANELA, len(cols)))
    for i in range(0, n0, 50_000):
        X[i:min(i + 50_000, n0)] = velho[i:min(i + 50_000, n0)]
    X[n0:] = np.nan
    b = tudo.iloc[n0:].assign(pos=np.arange(n0, len(tudo)), lat=lambda x: x.grid_lat.round(3),
                              lon=lambda x: x.grid_lon.round(3))
    cel_b = sorted(set(zip(b.grid_lat, b.grid_lon)))
    for y in anos:
        t0 = time.time()
        meta, x = ef.peak_windows_year(raw, cel_b, y, cols, window=JANELA)
        meta = meta.assign(i=np.arange(len(meta)), lat=meta.lat.round(3), lon=meta.lon.round(3))
        j = b[["pos", "lat", "lon", "time"]].merge(meta[["lat", "lon", "time", "i"]], on=["lat", "lon", "time"]).sort_values("pos")
        X[j.pos.to_numpy()] = x[j.i.to_numpy()]
        print(f"  janelas c5 {y}: {len(j)} em {time.time() - t0:.0f}s", flush=True)
    X.flush()
    ok = valid_rows(np.asarray(X[n0:]))
    tudo.loc[tudo.index[n0:], "janela_ok"] = ok
    del velho, X
    tmp.replace(JANELAS)
    tudo.to_parquet(BASE, index=False)
    est = pd.read_csv(OUT / "estacoes.csv")
    pd.concat([est[est.cluster_id != 5], st.assign(regiao="c5")], ignore_index=True).to_csv(OUT / "estacoes.csv", index=False)
    print(tudo[tudo.janela_ok].groupby(["cluster_id", "_split"]).size().unstack().to_string())
    print(f"cluster 5: {ok.sum()} janelas completas de {len(ok)}; {tudo.estacao[n0:].nunique()} estações")


def etapa_janelas(anos) -> None:
    cols = colunas()
    base = pd.read_parquet(BASE)
    n = len(base)
    X = np.lib.format.open_memmap(JANELAS, mode="w+", dtype="float32", shape=(n, JANELA, len(cols)))
    X[:] = np.nan
    linhas = base[["regiao", "grid_lat", "grid_lon", "time"]].assign(
        pos=np.arange(n), lat=base.grid_lat.round(3), lon=base.grid_lon.round(3))
    for reg, (raw, _, _) in REGIOES.items():
        b = linhas[linhas.regiao == reg]
        cel = sorted(set(zip(b.grid_lat, b.grid_lon)))
        for y in anos:
            t0 = time.time()
            meta, x = ef.peak_windows_year(raw, cel, y, cols, window=JANELA)
            meta = meta.assign(i=np.arange(len(meta)), lat=meta.lat.round(3), lon=meta.lon.round(3))
            j = b.merge(meta[["lat", "lon", "time", "i"]], on=["lat", "lon", "time"]).sort_values("pos")
            X[j.pos.to_numpy()] = x[j.i.to_numpy()]               # várias estações na mesma célula: mesma janela
            print(f"  {reg} {y}: {len(j)} janelas em {time.time() - t0:.0f}s", flush=True)
        X.flush()
    ok = np.zeros(n, dtype=bool)
    for i in range(0, n, 50_000):
        ok[i:i + 50_000] = valid_rows(np.asarray(X[i:i + 50_000]))
    base["janela_ok"] = ok
    base.to_parquet(BASE, index=False)
    ult = np.asarray(X[ok][:, -1, :][:20000]) if ok.any() else np.zeros((0, len(cols)))
    ref = base.loc[ok, cols].to_numpy("float32")[:20000]
    print(f"janelas completas: {ok.sum()} de {n} | última hora da janela = linha do pico: "
          f"máx |dif| {float(np.nanmax(np.abs(ult - ref))):.2e}")


def etapa_janelas_grade(anos) -> None:
    """Janelas de 24 h de TODAS as células dos polígonos (para o mapa da LSTM), nos `anos` pedidos.
    Grava `janelas_grade.npy` (n, 24, F) e `grade_meta.parquet` (lat, lon, time, season, cluster_id, gust10fg)."""
    from src.feature_study.diagnostics.inference import SEASON_OF_MONTH

    cols = colunas()
    shp = RAIZ / "dataset/shp/shp_vento.shp"
    metas, xs = [], []
    for reg, (raw, _, clusters) in REGIOES.items():
        lat, lon = ef.grid_of(raw)
        cel_de = {c: ef.cells_in_cluster(shp, c, lat, lon) for c in clusters}
        cel = sorted({x for v in cel_de.values() for x in v})
        cl_de = {(round(a, 3), round(b, 3)): c for c, v in cel_de.items() for a, b in v}
        for y in anos:
            t0 = time.time()
            meta, x = ef.peak_windows_year(raw, cel, y, cols, window=JANELA)
            meta["cluster_id"] = [cl_de[(round(a, 3), round(b, 3))] for a, b in zip(meta.lat, meta.lon)]
            meta["season"] = meta.time.dt.month.map(SEASON_OF_MONTH)
            meta["gust10fg"] = x[:, -1, cols.index("gust10fg")]
            metas.append(meta)
            xs.append(x)
            print(f"  {reg} {y}: {len(meta)} janelas em {time.time() - t0:.0f}s", flush=True)
    np.save(OUT / "janelas_grade.npy", np.concatenate(xs))
    pd.concat(metas, ignore_index=True).to_parquet(OUT / "grade_meta.parquet", index=False)
    print("janelas da grade:", sum(len(m) for m in metas))


def _eval(base, preds, rotulo, avalia):
    m = base.cluster_id.isin(list(avalia)).to_numpy()
    return rg.evaluate(base[m], {k: v[m] for k, v in preds.items()}, rotulo=rotulo)


def etapa_ml(so_anual: bool = False) -> None:
    cols, champ = colunas(), champions(MODAL / "summary")
    base = pd.read_parquet(BASE)
    base = base[base.janela_ok].reset_index(drop=True)            # as mesmas linhas da LSTM
    metricas, prev = [], {"row_id": base.row_id.to_numpy()}
    t0 = time.time()

    def registrar(chave, desenho, treino, fit, avalia):
        b, m = rg.predict(base, fit, cols)
        mt = _eval(base, {"modelo": b, "Corrigido ML": m}, {"familia": "ML", "chave": chave, "desenho": desenho,
                                                            "treino": treino}, avalia)
        metricas.append(mt)
        prev[f"{chave}_bruto"], prev[f"{chave}_eqm"] = b, m
        pd.concat(metricas, ignore_index=True).to_csv(OUT / "metricas_ml.csv", index=False)
        r = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")].pivot_table(
            index="cluster", columns="versao", values="RMSE")
        print(f"  ML {chave} ({desenho} [{treino}]) RMSE teste:\n{r.round(3).to_string()}", flush=True)

    if so_anual:
        # acrescenta o GA ao que já foi rodado, sem refazer os outros desenhos
        antigas = pd.read_csv(OUT / "metricas_ml.csv")
        metricas.append(antigas[antigas.chave != "GA"])
        prev.update({k: v for k, v in pd.read_parquet(OUT / "previsoes_ml.parquet").items()
                     if k != "row_id" and not k.startswith("GA_")})
    else:
        registrar("A", "A: modelo do cluster 3 (estudo)", "3 (estudo)", modelo_do_estudo(cols), rg.GLOBAL)
    for d in rg.designs():
        if so_anual and not d.get("anual"):
            continue
        sub = base[base.cluster_id.isin(d["train"])]
        if d.get("anual"):
            cols_a = [*cols, *rg.DOY_COLS]
            fit = rg.fit_single(rg.with_doy(sub), cols_a)
            b, m = rg.predict(rg.with_doy(base).assign(season=rg.ANUAL), fit, cols_a)
            mt = _eval(base, {"modelo": b, "Corrigido ML": m}, {"familia": "ML", "chave": d["chave"],
                                                                "desenho": d["desenho"], "treino": d["treino"]},
                       d["avalia"])
            metricas.append(mt)
            prev[f"{d['chave']}_bruto"], prev[f"{d['chave']}_eqm"] = b, m
            pd.concat(metricas, ignore_index=True).to_csv(OUT / "metricas_ml.csv", index=False)
            r = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")].pivot_table(
                index="cluster", columns="versao", values="RMSE")
            print(f"  ML {d['chave']} ({d['desenho']}, {rg.MODELO_ANUAL}) RMSE teste:\n{r.round(3).to_string()}", flush=True)
            continue
        registrar(d["chave"], d["desenho"], d["treino"], rg.fit_set(sub, cols, champ), d["avalia"])
    pd.DataFrame(prev).to_parquet(OUT / "previsoes_ml.parquet", index=False)
    print(f"ML concluído em {time.time() - t0:.0f}s")


def etapa_avaliar() -> None:
    base = pd.read_parquet(BASE)
    base = base[base.janela_ok].reset_index(drop=True)
    pos = pd.Series(np.arange(len(base)), index=base.row_id.to_numpy())
    metricas, prev, faltam = [], {"row_id": base.row_id.to_numpy()}, []
    for d in rg.designs():
        b = np.full(len(base), np.nan, dtype="float32")
        m = b.copy()
        for s in ((rg.ANUAL,) if d.get("anual") else rg.SEASONS):
            p = OUT / "lstm" / f"{d['chave']}__{s}.parquet"
            if not p.exists():
                faltam.append(p.name)
                continue
            u = pd.read_parquet(p)
            u = u[u.row_id.isin(pos.index)]
            i = pos.loc[u.row_id].to_numpy()
            b[i], m[i] = u.bruto.to_numpy(), u.eqm.to_numpy()
        prev[f"{d['chave']}_bruto"], prev[f"{d['chave']}_eqm"] = b, m
        metricas.append(_eval(base, {"modelo": b, "Corrigido ML": m},
                              {"familia": "LSTM", "chave": d["chave"], "desenho": d["desenho"], "treino": d["treino"]},
                              d["avalia"]))
    if faltam:
        print(f"[AVISO] {len(faltam)} unidades da LSTM ausentes (rode o download do Modal): {faltam[:6]}…")
    lstm = pd.concat(metricas, ignore_index=True)
    lstm.to_csv(OUT / "metricas_lstm.csv", index=False)
    pd.DataFrame(prev).to_parquet(OUT / "previsoes_lstm.parquet", index=False)
    tudo = pd.concat([pd.read_csv(OUT / "metricas_ml.csv"), lstm], ignore_index=True)
    tudo.to_csv(OUT / "metricas_global.csv", index=False)
    t = tudo[(tudo.trimestre == "Todos") & (tudo.recorte == "teste 2000–2024") & (tudo.versao == "modelo")]
    print(t.pivot_table(index=["familia", "chave"], columns="cluster", values="RMSE").round(3).to_string())
    (OUT / "resumo_avaliacao.json").write_text(json.dumps({"unidades_lstm_ausentes": faltam}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "adiciona-c5", "janelas", "janelas-grade", "ml", "ml-anual", "avaliar"], required=True)
    ap.add_argument("--anos", default="2000-2025")
    a = ap.parse_args()
    i, f = (int(x) for x in a.anos.split("-"))
    {"base": etapa_base, "janelas": lambda: etapa_janelas(range(i, f + 1)), "ml": etapa_ml, "adiciona-c5": lambda: etapa_adiciona_c5(range(i, f + 1)),
     "ml-anual": lambda: etapa_ml(so_anual=True), "avaliar": etapa_avaliar,
     "janelas-grade": lambda: etapa_janelas_grade(range(i, f + 1))}[a.stage]()
