"""Gera `notebooks/feature_study_grupos/atual/inferencia_cluster<N>.ipynb` (padrão: cluster 12).

Generalização em dado completamente novo: o modelo do cluster 3 aplicado a outro cluster, lendo
`artifacts/feature_study/inferencia_c<N>/results_cluster<N>.nc` (gerado por `scripts/inferencia_cluster.py`).

    uv run python scripts/build_notebook_cluster.py --cluster 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", type=int, default=12)
CL = ap.parse_args().cluster

DESTINO = Path(f"notebooks/feature_study_grupos/atual/inferencia_cluster{CL}.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

if CL == 3:
    md("""
# Cluster 3 (região do treino): ERA5 × Corrigido ML

O mesmo notebook do cluster 12, aplicado ao **próprio cluster 3**, para comparar as duas regiões lado a lado.
As colunas vêm do `test_inference_cluster_3.parquet` (o pipeline original; o gerador
`src/feature_study/data/era5_features.py` foi conferido contra ele).

**Atenção: aqui o modelo NÃO é cego.** As 9 estações do cluster 3 com INMET em 2000–2024 foram usadas no treino
(meses fev/mar/mai/jun/ago/set/nov/dez) e na calibração do mapa. Por isso **todas as comparações com o INMET usam
só os dias que o modelo não viu**: os meses de teste (jan/abr/jul/out de 2000–2024) e 2025. Os mapas de fundo
usam a série inteira (2000–2025), que inclui dias de treino nas células das estações.

Duas versões:
- **ERA5**: rajada bruta (`gust10fg`) na hora do pico do dia;
- **Corrigido ML**: o melhor modelo de cada trimestre com as colunas selecionadas **sem `anor_*`** (49 colunas,
  arm `sel__val12_sem_anor`), seguido do **mapeamento de quantis** ajustado na validação (Empirical Quantile
  Mapping, EQM). Ver `documentation/11_mapeamento_de_quantis.md`.

Gerar: `uv run python scripts/inferencia_cluster.py --cluster 3 --grade artifacts/feature_study/inferencia/_grade_diaria_pico.parquet`
e depois este notebook.
""")
else:
    md(f"""
    # Generalização: o modelo do cluster 3 aplicado ao cluster {CL}

    O melhor modelo de cada trimestre, treinado **só com o cluster 3** e com as 58 colunas selecionadas, aplicado ao
    cluster {CL}. As colunas foram calculadas do ERA5 bruto do cluster {CL} pelas mesmas regras do cluster 3
    (`src/feature_study/data/era5_features.py`, conferidas contra o cluster 3). **Nada do cluster {CL} entrou em treino,
    validação ou calibração**: todos os anos (2000–2025) e todas as estações são dado novo.

    Duas versões:
    - **ERA5**: rajada bruta (`gust10fg`) na hora do pico do dia;
    - **Corrigido ML**: o melhor modelo de cada trimestre do cluster 3 com as colunas selecionadas **sem `anor_*`**
      (49 colunas, arm `sel__val12_sem_anor`), seguido do **mapeamento de quantis** ajustado na validação do
      cluster 3 (Empirical Quantile Mapping, EQM: leva cada previsão ao valor do INMET no mesmo quantil,
      preservando a ordem dos dias). O que é, por que existe e seus limites: `documentation/11_mapeamento_de_quantis.md`.

    Por que sem `anor`: no cluster 3 ele funciona como identificador de estação (9 valores, um por estação); em
    região nova sai da faixa do treino e leva o MLP do DJF a prever ~+12 m/s acima do INMET.

    Gerar: `uv run python scripts/inferencia_cluster.py --cluster {CL}` e depois este notebook.
    """)


py(f"""
import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
plt.rcParams.update({{"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10}})

CL = {CL}
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
DIR = RAIZ / f"artifacts/feature_study/inferencia_c{{CL}}"
ds = xr.open_dataset(DIR / f"results_cluster{{CL}}.nc")
est = pd.read_csv(DIR / "estacoes.csv")
inmet = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")["daily_wind_gust_max"]
poli = ds.no_poligono == 1
# Corrigido ML = modelo sem anor + mapeamento de quantis (documentation/11_mapeamento_de_quantis.md)
FONTES = {{"ERA5": ("era5_gust_max", "#c44"), "Corrigido ML": ("pred_mapeado_sem_anor", "#2a7ab0")}}
MAPAS = list(FONTES)
print("modelos (treinados no cluster 3):", json.loads(ds.attrs["modelos"]))
print(f"grade: {{int(poli.sum())}} células no polígono | {{ds.sizes['time']}} dias "
      f"({{str(ds.time.values[0])[:10]}} a {{str(ds.time.values[-1])[:10]}})")
print(f"estações INMET: {{len(est)}} | distância estação→célula: mediana {{est.dist_celula_km.median():.1f}} km")
print(est[["estacao", "latitude", "longitude", "grid_lat", "grid_lon", "dist_celula_km", "dias_inmet"]].round(3).to_string(index=False))
""")

md("""
## 0. O cluster novo está dentro do que o modelo viu?

Fração dos dias-célula em que cada coluna fica **fora da faixa [mín, máx] do treino do cluster 3**. Árvores
não extrapolam (saturam no limite visto); o MLP do DJF extrapola livremente. Coluna muito fora = previsão a
desconfiar.
""")

py("""
fora = pd.read_csv(DIR / "fora_da_faixa.csv", index_col=0).iloc[:, 0]
fora = fora[~fora.index.str.startswith("anor")]           # o Corrigido ML não usa anor_*
print(fora[fora > 0].round(3).to_string())
fig, ax = plt.subplots(figsize=(9, 0.22 * max(len(fora[fora > 0.001]), 3) + 1.2))
f = fora[fora > 0.001].sort_values()
ax.barh(f.index, f.values * 100, color="#c44")
ax.set_xlabel("% dos dias-célula fora da faixa do treino (cluster 3)"); ax.tick_params(axis="y", labelsize=7)
ax.set_title(f"Cluster {CL}: colunas fora da faixa do cluster 3")
plt.tight_layout(); plt.show()
""")

md("""
## Dados brutos: rajada ERA5 (`gust10fg`) horária do cluster

Primeiro a delimitação do cluster no `shp_vento` (o cluster é uma região de vento, não precisa ser contínuo:
pode ter várias partes separadas). Depois o arquivo bruto `sl/gust10fg_<ano>.nc`, horário, na **grade inteira
baixada** (a vizinhança de 75/250 km usa também as células fora do polígono). Contorno = polígono do cluster;
triângulos = estações INMET.
""")

py("""
import geopandas as gpd

RAW = RAIZ / f"dataset/raw/new_features/cluster{CL}/sl"
poly = gpd.read_file(RAIZ / "dataset/shp/shp_vento.shp").to_crs(4326)
poly = poly[poly.cluster.astype(int) == CL]
diarios, soma, n_h, maximo = [], None, 0, None
for f in sorted(RAW.glob("gust10fg_*.nc")):
    d_ = xr.open_dataset(f)
    g = d_[list(d_.data_vars)[0]]
    g = g.rename(valid_time="time") if "valid_time" in g.dims else g
    soma = g.sum("time") if soma is None else soma + g.sum("time")
    n_h += g.sizes["time"]
    mx = g.max("time")
    maximo = mx if maximo is None else np.maximum(maximo, mx)
    diarios.append(g.resample(time="1D").max().load())
bruto_d = xr.concat(diarios, "time")                        # máximo diário de cada célula, 2000–2025
media_h = soma / n_h
print(f"grade bruta: {bruto_d.sizes['latitude']} × {bruto_d.sizes['longitude']} células, {n_h} horas, "
      f"{bruto_d.sizes['time']} dias")
""")

py("""
# delimitação: todos os clusters do shp_vento, com o cluster alvo em vermelho
todos = gpd.read_file(RAIZ / "dataset/shp/shp_vento.shp").to_crs(4326)
g_ = poly.geometry.iloc[0]
partes = sorted(g_.geoms, key=lambda p: -p.area) if g_.geom_type == "MultiPolygon" else [g_]
print(f"cluster {CL}: {g_.geom_type} com {len(partes)} partes; maiores (graus²): "
      f"{[round(p.area, 3) for p in partes[:5]]}")
fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.25]})
todos.plot(ax=axs[0], column="cluster", cmap="tab20", edgecolor="k", lw=0.3)
poly.boundary.plot(ax=axs[0], color="red", lw=2)
for _, r_ in todos.iterrows():
    p_ = r_.geometry.representative_point()
    axs[0].text(p_.x, p_.y, r_.cluster, fontsize=8, ha="center", weight="bold" if int(r_.cluster) == CL else None)
axs[0].set_title("shp_vento: os 14 clusters (alvo em vermelho)")
axs[1].pcolormesh(ds.lon, ds.lat, poli.where(poli), cmap="Greens", vmin=0, vmax=1.5, shading="nearest", alpha=0.7)
poly.boundary.plot(ax=axs[1], color="red", lw=1.5)
todos[todos.cluster.astype(int) != CL].boundary.plot(ax=axs[1], color="#888", lw=0.5)
axs[1].scatter(est.longitude, est.latitude, marker="^", s=40, c="k", zorder=3)
for _, r_ in est.iterrows():
    axs[1].annotate(r_.estacao, (r_.longitude, r_.latitude), fontsize=7, xytext=(3, 3), textcoords="offset points")
axs[1].set_xlim(float(ds.lon.min()) - 0.5, float(ds.lon.max()) + 0.5)
axs[1].set_ylim(float(ds.lat.min()) - 0.5, float(ds.lat.max()) + 0.5)
axs[1].set_title(f"cluster {CL}: polígono, {int(poli.sum())} células usadas (verde) e estações INMET")
for ax in axs:
    ax.set_aspect("equal")
plt.tight_layout(); plt.show()

campos = {"média horária": media_h, "média do máximo diário": bruto_d.mean("time"),
          "P99 do máximo diário": bruto_d.quantile(0.99, "time"), "máximo absoluto": maximo}
fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), constrained_layout=True)
for ax, (nome, campo) in zip(axs, campos.items()):
    im = ax.pcolormesh(campo.longitude, campo.latitude, campo, cmap="viridis", shading="nearest")
    poly.boundary.plot(ax=ax, color="k", lw=1.2)
    ax.scatter(est.longitude, est.latitude, marker="^", s=28, c="w", edgecolors="k", zorder=3)
    ax.set_title(f"gust10fg bruto — {nome}"); ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.75, label="m/s")
fig.suptitle(f"Cluster {CL}: rajada ERA5 bruta, 2000–2025 (grade inteira baixada)", fontsize=13)
plt.show()
""")

py("""
# o dia com a maior rajada bruta dentro do polígono, hora a hora em 4 instantes
dentro = bruto_d.sel(latitude=ds.lat.values, longitude=ds.lon.values, method="nearest").where(poli.values)
dia = pd.Timestamp(dentro.max(["latitude", "longitude"]).idxmax("time").values)
g = xr.open_dataset(RAW / f"gust10fg_{dia.year}.nc")
g = g[list(g.data_vars)[0]]
g = (g.rename(valid_time="time") if "valid_time" in g.dims else g).sel(time=str(dia.date()))
hmax = int(g.max(["latitude", "longitude"]).argmax())
horas = sorted({max(hmax - 6, 0), max(hmax - 3, 0), hmax, min(hmax + 3, 23)})
vmax = float(g.max())
fig, axs = plt.subplots(1, len(horas), figsize=(4.8 * len(horas), 5), constrained_layout=True)
for ax, h in zip(np.atleast_1d(axs), horas):
    im = ax.pcolormesh(g.longitude, g.latitude, g.isel(time=h), cmap="magma", vmin=0, vmax=vmax, shading="nearest")
    poly.boundary.plot(ax=ax, color="w", lw=1.2)
    ax.set_title(f"{dia.date()} {h:02d} UTC"); ax.set_aspect("equal")
fig.colorbar(im, ax=axs, shrink=0.75, label="gust10fg (m/s)")
fig.suptitle(f"Dia de maior rajada bruta no polígono: {dia.date()} (máx. {vmax:.1f} m/s na grade)", fontsize=12)
plt.show()
""")

py("""
# série horária bruta na célula de uma estação, uma semana em torno do dia acima, com o máximo diário do INMET
r = est.set_index("estacao").loc[ex if "ex" in dir() else est.sort_values("dias_inmet").estacao.iloc[-1]]
ini, fim = dia - pd.Timedelta(days=3), dia + pd.Timedelta(days=5) - pd.Timedelta(hours=1)   # dias UTC inteiros
anos = sorted({ini.year, fim.year})
h = xr.concat([xr.open_dataset(RAW / f"gust10fg_{a}.nc") for a in anos], "valid_time")
h = h[list(h.data_vars)[0]].rename(valid_time="time").sel(latitude=r.grid_lat, longitude=r.grid_lon, method="nearest")
h = h.sel(time=slice(ini, fim))
o = inmet.sel(estacao=r.name).sel(time=slice(ini.normalize(), fim.normalize()))
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(h.time, h, color="#c44", lw=1, label="gust10fg bruto (horário)")
hd = h.resample(time="1D").max()                     # máximo do dia UTC (o que o estudo usa como ERA5)
ax.step(np.r_[hd.time.values, hd.time.values[-1] + np.timedelta64(1, "D")], np.r_[hd.values, hd.values[-1]],
        where="post", color="#c44", lw=1.6, ls="--", label="gust10fg: máximo do dia UTC")
ax.step(np.r_[o.time.values, o.time.values[-1] + np.timedelta64(1, "D")], np.r_[o.values, o.values[-1]],
        where="post", color="#111", lw=1.6, label="INMET: máximo diário")
ax.set_title(f"{r.name}: rajada ERA5 bruta hora a hora × INMET diário ({ini.date()} a {fim.date()})")
ax.set_ylabel("m/s"); ax.legend(); plt.show()
""")

md("""
## 1. Mapas: média e percentis 90, 95 e 99, com o INMET por cima

Fundo: a estatística da série diária de 2000–2025 em cada célula do polígono. Círculos: a mesma estatística do
INMET em cada estação (com ≥ 300 dias), na **mesma escala de cor**. Última linha: mapeado − ERA5.
""")

py("""
MIN_DIAS = 300
ESTAT = {"média": None, "P90": 0.90, "P95": 0.95, "P99": 0.99}


def stat(x, q):
    return x.mean("time") if q is None else x.quantile(q, "time")


ob = inmet.sel(estacao=est.estacao.tolist()).to_dataframe().reset_index().dropna(subset=["daily_wind_gust_max"])
ob = ob.drop(columns=["latitude", "longitude"], errors="ignore").merge(est, on="estacao")
gr = ds[[v for v, _ in FONTES.values()]].to_dataframe().reset_index().dropna()
gr["lat"], gr["lon"] = gr.lat.round(3), gr.lon.round(3)
ob["grid_lat"], ob["grid_lon"] = ob.grid_lat.round(3), ob.grid_lon.round(3)
par = ob.merge(gr, left_on=["grid_lat", "grid_lon", "time"], right_on=["lat", "lon", "time"])
par = par.rename(columns={"daily_wind_gust_max": "INMET", **{v: k for k, (v, _) in FONTES.items()}})
par["season"] = par.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
if CL == 3:   # no cluster 3 as estações estão no treino: só dias não vistos (meses de teste e 2025)
    par = par[par.time.dt.month.isin((1, 4, 7, 10)) | (par.time.dt.year >= 2025)]
    print("cluster 3: só dias não vistos pelo modelo (jan/abr/jul/out de 2000–2024 e 2025)")
n_dias = par.groupby("estacao").size()
parm = par[par.estacao.isin(n_dias[n_dias >= MIN_DIAS].index)]
print(f"pares estação-dia: {len(par)} | estações com ≥ {MIN_DIAS} dias: {parm.estacao.nunique()}")


def por_estacao(q):
    agg = (lambda x: x.mean()) if q is None else (lambda x: x.quantile(q))
    e = parm.groupby("estacao")[["INMET", *FONTES]].agg(agg)
    return e.join(est.set_index("estacao")[["latitude", "longitude"]])


NR = len(MAPAS) + 1
fig, axs = plt.subplots(NR, 4, figsize=(18, 4.6 * NR), constrained_layout=True)
for j, (nome, q) in enumerate(ESTAT.items()):
    campos = {k: stat(ds[FONTES[k][0]].where(poli), q) for k in MAPAS}
    e = por_estacao(q)
    vmin = float(min(*(x.min() for x in campos.values()), e.INMET.min()))
    vmax = float(max(*(x.max() for x in campos.values()), e.INMET.max()))
    for i, (rot, campo) in enumerate(campos.items()):
        im = axs[i, j].pcolormesh(ds.lon, ds.lat, campo, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
        axs[i, j].scatter(e.longitude, e.latitude, c=e.INMET, cmap="viridis", vmin=vmin, vmax=vmax, s=120,
                          edgecolors="white", linewidths=1.6, zorder=3)
        axs[i, j].set_title(f"{rot} + INMET\\n{nome} (média {float(campo.mean()):.1f})", fontsize=9)
    fig.colorbar(im, ax=axs[:NR - 1, j], shrink=0.6, label="m/s")
    d = campos[MAPAS[-1]] - campos["ERA5"]
    lim = float(np.abs(d).max())
    im = axs[NR - 1, j].pcolormesh(ds.lon, ds.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    axs[NR - 1, j].set_title(f"{MAPAS[-1]} − ERA5 — {nome} ({float(d.mean()):+.2f})")
    fig.colorbar(im, ax=axs[NR - 1, j], shrink=0.8, label="m/s")
for ax in axs.flat:
    ax.set_aspect("equal")
fig.suptitle(f"Cluster {CL} ({'região do treino; INMET só em dias não vistos' if CL == 3 else 'dado novo'}): {', '.join(MAPAS)} com o INMET por cima", fontsize=13)
plt.show()
""")

md("""
## 2. Contra o INMET, dia a dia

Todas as estações e todos os dias são novos para o modelo. RMSE, viés e correlação diários; POD e FAR acima
do P95 e do P99 do INMET de cada estação. **Ganho > 0 = erra menos que o ERA5.**
""")

py("""
def resumo(d):
    out = {"n": len(d), "estações": d.estacao.nunique()}
    for k in FONTES:
        e_ = d[k] - d.INMET
        out[f"RMSE_{k}"] = float(np.sqrt((e_ ** 2).mean()))
        out[f"viés_{k}"] = float(e_.mean())
        out[f"corr_{k}"] = float(np.corrcoef(d[k], d.INMET)[0, 1])
    for qn, qv in (("P95", 0.95), ("P99", 0.99)):
        lim = d.groupby("estacao").INMET.transform(lambda x: x.quantile(qv))
        ev = d.INMET >= lim
        for k in FONTES:
            al = d[k] >= lim
            out[f"POD_{qn}_{k}"] = float((ev & al).sum() / max(ev.sum(), 1))
            out[f"FAR_{qn}_{k}"] = float((~ev & al).sum() / max(al.sum(), 1))
    return pd.Series(out)


tab = pd.concat([resumo(par).rename("Todos"), *[resumo(g).rename(k) for k, g in par.groupby("season")]], axis=1).T
for k in list(FONTES)[1:]:
    tab[f"ganho_{k}"] = tab["RMSE_ERA5"] - tab[f"RMSE_{k}"]
blocos = {"erro": ["n", "estações", *[f"RMSE_{k}" for k in FONTES], *[f"ganho_{k}" for k in list(FONTES)[1:]],
                   *[f"viés_{k}" for k in FONTES], *[f"corr_{k}" for k in FONTES]],
          "extremos": [f"{m}_{q}_{k}" for q in ("P95", "P99") for m in ("POD", "FAR") for k in FONTES]}
for nome, cols_ in blocos.items():
    print(f"--- {nome}")
    print(tab[cols_].round(3).to_string())
""")

py("""
porest = par.groupby("estacao").apply(lambda d: pd.Series({
    "n": len(d), **{k: float(np.sqrt(((d[k] - d.INMET) ** 2).mean())) for k in FONTES}}))
MELHOR = MAPAS[-1]
porest["ganho"] = porest.ERA5 - porest[MELHOR]
porest = porest.sort_values("ganho")
fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(len(porest))
w_ = 0.8 / len(FONTES)
for k_, (k, (_, cor)) in enumerate(FONTES.items()):
    ax.bar(x + (k_ - (len(FONTES) - 1) / 2) * w_, porest[k], width=w_, color=cor, label=k)
ax.set_xticks(x, [f"{e}\\n(n={int(n)})" for e, n in zip(porest.index, porest.n)], fontsize=8)
ax.set_ylabel("RMSE diário (m/s)"); ax.legend()
ax.set_ylim(0, float(porest[["ERA5", MELHOR]].max().max()) * 1.6)
ax.set_title(f"RMSE por estação, cluster {CL} ({int((porest.ganho > 0).sum())} de {len(porest)} estações em que o {MELHOR} erra menos que o ERA5; eixo cortado)")
plt.tight_layout(); plt.show()
""")

md("""
## 3. Estatística por estação, com reta de ajuste

Para cada estação: a estatística do INMET contra a da sua célula, nos mesmos dias. Inclinação 1 = a grade
acompanha a diferença entre as estações.
""")

py("""
fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
lin = []
for ax, (nome, q) in zip(axs, ESTAT.items()):
    e = por_estacao(q)
    for k, (_, cor) in FONTES.items():
        a_, b_ = np.polyfit(e.INMET, e[k], 1)
        rm = float(np.sqrt(((e[k] - e.INMET) ** 2).mean()))
        xx = np.array([e.INMET.min(), e.INMET.max()])
        ax.plot(xx, a_ * xx + b_, color=cor, lw=1.4)
        ax.scatter(e.INMET, e[k], c=cor, s=35, label=f"{k} ({rm:.2f}; incl. {a_:.2f})", zorder=3)
        lin.append({"estatística": nome, "fonte": k, "RMSE_entre_estacoes": rm, "inclinação": a_,
                    "viés": float((e[k] - e.INMET).mean())})
    lo, hi = float(e[["INMET", *FONTES]].min().min()) - 0.5, float(e[["INMET", *FONTES]].max().max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(nome); ax.set_xlabel("INMET (m/s)"); ax.set_ylabel("célula, mesmos dias (m/s)"); ax.legend(fontsize=7)
fig.suptitle("Estatística por estação (entre parênteses: RMSE entre estações; inclinação, 1 = ideal)")
plt.show()
print(pd.DataFrame(lin).pivot(index="estatística", columns="fonte").round(2).reindex(list(ESTAT)).to_string())
""")

py("""
fig, ax = plt.subplots(figsize=(6.5, 6.5))
qs = np.linspace(0.5, 0.999, 200)
qi = np.quantile(par.INMET, qs)
for k, (_, cor) in FONTES.items():
    ax.plot(qi, np.quantile(par[k], qs), color=cor, lw=2, label=k)
ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k--", lw=0.8)
for q in (0.9, 0.95, 0.99):
    ax.axvline(np.quantile(par.INMET, q), color="#999", lw=0.6, ls=":")
ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil da grade (m/s)")
ax.set_title(f"Cluster {CL}: quantis da mediana ao P99,9\\n(abaixo da diagonal = corta os extremos)")
ax.legend(); plt.show()
""")

md("""
## 4. Séries: ERA5, padrão, mapeado e INMET

A estação com mais dias em 2025 e um mês de cada trimestre de 2018.
""")

py("""
def serie(estacao, ini, fim, ax):
    r = est.set_index("estacao").loc[estacao]
    cel = ds.sel(lat=r.grid_lat, lon=r.grid_lon, method="nearest").sel(time=slice(ini, fim))
    o = inmet.sel(estacao=estacao).sel(time=slice(ini, fim))
    for k, (v, cor) in FONTES.items():
        ax.plot(cel.time, cel[v], color=cor, lw=1.1 if k == "ERA5" else 1.3, label=k)
    ax.plot(o.time, o, color="#111", lw=1.1, marker=".", ms=3, label="INMET")
    ax.set_title(f"{estacao} — {ini[:7]} a {fim[:7]}"); ax.set_ylabel("m/s")


cont25 = par[par.time.dt.year == 2025].groupby("estacao").size()
ex = cont25.idxmax() if len(cont25) else n_dias.idxmax()
fig, axs = plt.subplots(5, 1, figsize=(16, 17))
serie(ex, "2025-01-01", "2025-12-31", axs[0])
for ax, mes in zip(axs[1:], ["2018-01", "2018-04", "2018-07", "2018-10"]):
    serie(ex, f"{mes}-01", str((pd.Timestamp(f"{mes}-01") + pd.offsets.MonthEnd(0)).date()), ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
axs[0].legend(ncol=4, loc="upper left")
plt.tight_layout(); plt.show()
""")

md(f"""
## Limites

1. **Clima diferente**: o modelo e o mapa de quantis aprenderam o cluster 3 (sul). Viés sistemático aqui é
   sinal de que a relação ERA5 → INMET muda de região, não de erro de código.
2. **Colunas fora da faixa** (seção 0): árvores saturam no limite visto; o MLP do DJF extrapola sem limite.
3. **`mcpr_r75_fracpos`, `mcpr_l75_250` e `grad_mslp_*` não puderam ser conferidos** contra o cluster 3 (sem o
   bruto de `mcpr`/`msl` dele); seguem a mesma regra das colunas conferidas.
4. **O Corrigido ML não usa `anor_*`**: no cluster 3 ele funciona como identificador de estação e, no
   cluster {CL}, seus valores ficam fora da faixa do treino (seção 0).
5. **O mapeamento de quantis foi ajustado com o INMET do cluster 3**: ele estica a distribuição para a do sul;
   aqui ele é aplicado sem nenhum ajuste local.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
