"""Gera `notebooks/feature_study_grupos/atual/inferencia_selecao_features.ipynb`.

ERA5 antes e depois da correção na grade do cluster 3 (melhor modelo por trimestre + colunas selecionadas),
lendo `artifacts/feature_study/inferencia/results_feature_selction.nc`, gerado por
`scripts/inferencia_selecao_features.py`. Nada é treinado no notebook.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/inferencia_selecao_features.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# ERA5 antes e depois da correção — grade do cluster 3

O melhor modelo de cada trimestre (rank 1 de `top_models.json`), treinado só com as 58 colunas selecionadas
(`config/features_maping.json`), aplicado às **74 células de 0,25°** de `test_inference_cluster_3.parquet`,
2000–2025. Uma linha por célula e dia, na hora de maior `gust10fg`: a previsão é a **rajada máxima diária**.

O que é dado novo para o modelo:
- **células sem estação** (65 de 74): nenhum alvo foi visto ali;
- **2025**: o treino vai até 2024;
- **12 estações que só reportam desde set/out de 2025**: nunca entraram no treino.

Nas células das 9 estações usadas no treino, os dias de **meses de treino** estão dentro da amostra: o erro
ali é otimista e não serve de prova. Gerar o `.nc`: `uv run python scripts/inferencia_selecao_features.py`.
""")

py("""
import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
ds = xr.open_dataset(RAIZ / "artifacts/feature_study/inferencia/results_feature_selction.nc")
g4 = pd.read_parquet(RAIZ / "dataset/raw/feature_study_cluster_3/features_grupo4_cluster3.parquet",
                     columns=["codigo_estacao", "latitude", "longitude", "grid_lat", "grid_lon"])
inmet = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")["daily_wind_gust_max"]

TREINO9 = ["A801", "A834", "A840", "A878", "A879", "A882", "A884", "A893", "B807"]
print("modelos por trimestre:", json.loads(ds.attrs["modelos"]))
print("RMSE de teste do reajuste (igual ao Modal):", json.loads(ds.attrs["rmse_teste_reajuste"]))
print(f"grade: {ds.sizes['lat']} lat × {ds.sizes['lon']} lon × {ds.sizes['time']} dias "
      f"({str(ds.time.values[0])[:10]} a {str(ds.time.values[-1])[:10]})")
print(f"dias-célula com previsão: {int(ds.pred_gust_max.notnull().sum())} | "
      f"sem previsão (coluna faltando, nada imputado): {int((ds.era5_gust_max.notnull() & ds.pred_gust_max.isnull()).sum())}")
""")

md("""
## 1. Mapas: média e percentis 90, 95 e 99

Para cada célula, estatística da série diária de 2000–2025 do ERA5 (rajada na hora do pico) e da previsão
corrigida. A última linha é a diferença (corrigido − ERA5): **vermelho = a correção aumenta o vento**.
Triângulos: as 9 estações do treino; círculos: as 12 que só existem em 2025.
""")

py("""
ESTAT = {"média": lambda x: x.mean("time"), "P90": lambda x: x.quantile(0.90, "time"),
         "P95": lambda x: x.quantile(0.95, "time"), "P99": lambda x: x.quantile(0.99, "time")}
valido = ds.pred_gust_max.notnull()
era5 = ds.era5_gust_max.where(valido)
pred = ds.pred_gust_max


def estacoes(ax):
    t = g4[g4.codigo_estacao.isin(TREINO9)]
    n = g4[~g4.codigo_estacao.isin(TREINO9)]
    ax.scatter(t.longitude, t.latitude, marker="^", s=30, c="k", label="estação do treino")
    ax.scatter(n.longitude, n.latitude, marker="o", s=18, facecolors="none", edgecolors="k", label="estação só em 2025")


def mapas(era5, pred, titulo):
    fig, axs = plt.subplots(3, 4, figsize=(18, 11.5), constrained_layout=True)
    for j, (nome, f) in enumerate(ESTAT.items()):
        a, b = f(era5), f(pred)
        vmin, vmax = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        for i, (rot, campo) in enumerate([("ERA5", a), ("corrigido", b)]):
            im = axs[i, j].pcolormesh(ds.lon, ds.lat, campo, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
            axs[i, j].set_title(f"{rot} — {nome} (média {float(campo.mean()):.1f} m/s)")
        fig.colorbar(im, ax=axs[:2, j], shrink=0.8, label="m/s")
        d = b - a
        lim = float(np.abs(d).max())
        im = axs[2, j].pcolormesh(ds.lon, ds.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        axs[2, j].set_title(f"corrigido − ERA5 — {nome} (média {float(d.mean()):+.2f})")
        fig.colorbar(im, ax=axs[2, j], shrink=0.8, label="m/s")
    for ax in axs.flat:
        estacoes(ax)
        ax.set_aspect("equal")
    axs[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(titulo, fontsize=13)
    plt.show()


mapas(era5, pred, "Rajada máxima diária, 2000–2025, todos os trimestres")
""")

md("""
### ERA5 + INMET e corrigido + INMET

Mesmos mapas, agora com o **INMET por cima**: cada círculo é uma estação, pintado com a mesma estatística e a
**mesma escala de cor** do mapa embaixo dele. Se o corrigido "absorveu" o INMET, a cor do círculo some no fundo
do corrigido e contrasta com o do ERA5. Só entram **dias que o modelo não viu** (meses de teste jan/abr/jul/out e 2025) e estações com pelo menos
300 desses dias (as 12 novas têm no máximo 38, pouco para um P99). O fundo é a grade em 2000–2025.

A última linha compara de forma justa: para cada estação, a estatística do INMET contra a da sua célula
(ERA5 e corrigido) **nos mesmos dias**. Pontos na diagonal = a grade reproduz a estação.
""")

py("""
MIN_DIAS = 300
ob = inmet.sel(estacao=inmet.estacao.isin(g4.codigo_estacao)).to_dataframe().reset_index().dropna(subset=["daily_wind_gust_max"])
ob = ob.drop(columns=["latitude", "longitude"], errors="ignore").merge(
    g4.rename(columns={"codigo_estacao": "estacao"}), on="estacao")
ob["grid_lat"], ob["grid_lon"] = ob.grid_lat.round(2), ob.grid_lon.round(2)
gr = ds[["era5_gust_max", "pred_gust_max"]].to_dataframe().reset_index().dropna()
gr["lat"], gr["lon"] = gr.lat.round(2), gr.lon.round(2)
par = ob.merge(gr, left_on=["grid_lat", "grid_lon", "time"], right_on=["lat", "lon", "time"])
# só dias que o modelo NÃO viu: meses de teste (jan/abr/jul/out) e 2025. Nos meses de treino o modelo já viu
# esse INMET, e a concordância ali não prova nada.
par = par[par.time.dt.month.isin((1, 4, 7, 10)) | (par.time.dt.year >= 2025)]
n_dias = par.groupby("estacao").size()
par = par[par.estacao.isin(n_dias[n_dias >= MIN_DIAS].index)]
QS = {"média": None, "P90": 0.90, "P95": 0.95, "P99": 0.99}


def est_por_estacao(q):
    agg = (lambda x: x.mean()) if q is None else (lambda x: x.quantile(q))
    t = par.groupby("estacao").agg(INMET=("daily_wind_gust_max", agg), ERA5=("era5_gust_max", agg),
                                   corrigido=("pred_gust_max", agg), lat=("latitude", "first"),
                                   lon=("longitude", "first"), n=("time", "size"))
    return t


fig, axs = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
linhas = []
for j, (nome, f) in enumerate(ESTAT.items()):
    a, b = f(era5), f(pred)
    e = est_por_estacao(QS[nome])
    vmin = float(min(a.min(), b.min(), e.INMET.min()))
    vmax = float(max(a.max(), b.max(), e.INMET.max()))
    for i, (rot, campo) in enumerate([("ERA5 + INMET", a), ("corrigido + INMET", b)]):
        im = axs[i, j].pcolormesh(ds.lon, ds.lat, campo, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
        axs[i, j].scatter(e.lon, e.lat, c=e.INMET, cmap="viridis", vmin=vmin, vmax=vmax, s=170,
                          edgecolors="white", linewidths=1.8, zorder=3)
        axs[i, j].set_title(f"{rot} — {nome}")
        axs[i, j].set_aspect("equal")
    fig.colorbar(im, ax=axs[:2, j], shrink=0.8, label="m/s")
    ax = axs[2, j]
    for col, cor in [("ERA5", "#c44"), ("corrigido", "#2a7ab0")]:
        rm = float(np.sqrt(((e[col] - e.INMET) ** 2).mean()))
        ax.scatter(e.INMET, e[col], c=cor, s=45, label=f"{col} (RMSE entre estações {rm:.2f})", zorder=3)
        linhas.append({"estatística": nome, "fonte": col, "RMSE_entre_estacoes": rm,
                       "viés_médio": float((e[col] - e.INMET).mean())})
    for est, r in e.iterrows():
        ax.annotate(est, (r.INMET, r.corrigido), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo, hi = float(e[["INMET", "ERA5", "corrigido"]].min().min()) - 0.5, float(e[["INMET", "ERA5", "corrigido"]].max().max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"INMET — {nome} (m/s)"); ax.set_ylabel("grade, mesmos dias (m/s)")
    ax.set_title(f"{nome}: estação × célula")
    ax.legend(fontsize=7, loc="upper left")
fig.suptitle(f"INMET por cima da grade — só dias não vistos pelo modelo ({len(e)} estações com ≥ {MIN_DIAS} dias)", fontsize=13)
plt.show()

print("dias por estação:", n_dias[n_dias >= MIN_DIAS].to_dict())
print(pd.DataFrame(linhas).pivot(index="estatística", columns="fonte").round(2).reindex(list(QS)).to_string())
""")

md("""
### Por trimestre climático

Mesma estatística por trimestre, só a diferença (corrigido − ERA5), para ver onde e quando a correção age.
""")

py("""
fig, axs = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
dif = {}
for i, s in enumerate(["DJF", "MAM", "JJA", "SON"]):
    m = ds.time.dt.season == s
    for j, (nome, f) in enumerate(ESTAT.items()):
        dif[(s, nome)] = f(pred.where(m)) - f(era5.where(m))
lim = max(float(np.abs(v).max()) for v in dif.values())
for (s, nome), d in dif.items():
    i, j = ["DJF", "MAM", "JJA", "SON"].index(s), list(ESTAT).index(nome)
    im = axs[i, j].pcolormesh(ds.lon, ds.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    axs[i, j].set_title(f"{s} {nome}: {float(d.mean()):+.2f} m/s")
    estacoes(axs[i, j])
    axs[i, j].set_aspect("equal")
fig.colorbar(im, ax=axs, shrink=0.6, label="corrigido − ERA5 (m/s)")
plt.show()

tab = pd.DataFrame({(s, n): {"ERA5": float(ESTAT[n](era5.where(ds.time.dt.season == s)).mean()),
                              "corrigido": float(ESTAT[n](pred.where(ds.time.dt.season == s)).mean())}
                    for s in ["DJF", "MAM", "JJA", "SON"] for n in ESTAT}).T
tab["diferença"] = tab.corrigido - tab.ERA5
print("Média espacial de cada estatística (m/s):")
print(tab.round(2).to_string())
""")

md("""
## 2. Contra o INMET nas células das estações

Cada estação é comparada com a célula da grade que a contém (`grid_lat`, `grid_lon`). Três recortes:
- **teste 2000–2024**: meses de teste (jan/abr/jul/out) das 9 estações do treino;
- **2025**: ano nunca visto, incluindo as 12 estações novas;
- **treino**: dentro da amostra, só como referência (otimista).

**Ganho > 0 = a correção erra menos que o ERA5.** Linhas sem previsão (coluna faltando) ficam de fora dos dois.
""")

py("""
TESTE_MESES = (1, 4, 7, 10)
obs = inmet.sel(estacao=inmet.estacao.isin(g4.codigo_estacao)).to_dataframe().reset_index().dropna(subset=["daily_wind_gust_max"])
obs = obs.merge(g4.rename(columns={"codigo_estacao": "estacao"}), on="estacao")
grade = ds[["era5_gust_max", "pred_gust_max"]].to_dataframe().reset_index().dropna()
grade["lat"], grade["lon"] = grade.lat.round(2), grade.lon.round(2)
obs["grid_lat"], obs["grid_lon"] = obs.grid_lat.round(2), obs.grid_lon.round(2)
j = obs.merge(grade, left_on=["grid_lat", "grid_lon", "time"], right_on=["lat", "lon", "time"])
j["season"] = j.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
j["recorte"] = np.select([j.time.dt.year >= 2025, j.time.dt.month.isin(TESTE_MESES)],
                         ["2025 (nunca visto)", "teste 2000–2024"], "treino (dentro da amostra)")
j["grupo_est"] = np.where(j.estacao.isin(TREINO9), "estação do treino", "estação nova (2025)")
y = "daily_wind_gust_max"


def resumo(d):
    e, p = d.era5_gust_max - d[y], d.pred_gust_max - d[y]
    return pd.Series({"n": len(d), "estações": d.estacao.nunique(),
                      "RMSE_ERA5": np.sqrt((e ** 2).mean()), "RMSE_corr": np.sqrt((p ** 2).mean()),
                      "viés_ERA5": e.mean(), "viés_corr": p.mean()})


t = pd.concat([j.groupby(["recorte", "grupo_est"]).apply(resumo).assign(trimestre="Todos").reset_index(),
               j.groupby(["recorte", "grupo_est", "season"]).apply(resumo).reset_index().rename(columns={"season": "trimestre"})])
t["ganho"] = t.RMSE_ERA5 - t.RMSE_corr
ORDEM = ["teste 2000–2024", "2025 (nunca visto)", "treino (dentro da amostra)"]
t["_o"] = t.recorte.map(ORDEM.index)
t = t.sort_values(["_o", "grupo_est", "trimestre"]).drop(columns="_o")
print(t.round(3).to_string(index=False))
""")

py("""
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
for ax, rec in zip(axs, ORDEM):
    d = t[(t.recorte == rec) & (t.trimestre != "Todos")]
    for k, (g, dd) in enumerate(d.groupby("grupo_est")):
        x = np.arange(len(dd)) + k * 0.4
        ax.bar(x, dd.ganho, width=0.38, label=g, color=["#2a7ab0", "#d98c3f"][k])
        ax.set_xticks(np.arange(len(dd)) + 0.2, dd.trimestre)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(rec)
    ax.set_ylabel("ganho de RMSE sobre o ERA5 (m/s)")
    ax.legend(fontsize=8)
plt.show()
""")

md("""
## 3. Séries: ERA5, corrigido e INMET

Séries diárias nas células das estações. Primeiro 2025 inteiro (dado novo) para uma estação do treino e uma
estação nova; depois uma janela de teste (janeiro, abril, julho e outubro de um ano qualquer).
""")

py("""
CORES = {"INMET": "#111111", "ERA5": "#c44", "corrigido": "#2a7ab0"}


def serie(estacao, ini, fim, ax):
    r = g4[g4.codigo_estacao == estacao].iloc[0]
    cel = ds.sel(lat=r.grid_lat, lon=r.grid_lon, method="nearest").sel(time=slice(ini, fim))
    o = inmet.sel(estacao=estacao).sel(time=slice(ini, fim))
    ax.plot(cel.time, cel.era5_gust_max, color=CORES["ERA5"], lw=1, label="ERA5")
    ax.plot(cel.time, cel.pred_gust_max, color=CORES["corrigido"], lw=1.2, label="corrigido")
    ax.plot(o.time, o, color=CORES["INMET"], lw=1.1, marker=".", ms=3, label="INMET")   # falha do INMET = quebra na linha
    ax.set_title(f"{estacao} ({'treino' if estacao in TREINO9 else 'nova, só 2025'}) — célula {r.grid_lat}, {r.grid_lon}")
    ax.set_ylabel("rajada máx. diária (m/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))


# estações com mais INMET em 2025
cont = j[j.time.dt.year >= 2025].groupby(["grupo_est", "estacao"]).size()
exemplo_treino = cont["estação do treino"].idxmax()
exemplo_nova = cont["estação nova (2025)"].idxmax() if "estação nova (2025)" in cont.index.get_level_values(0) else None
print("dias de INMET em 2025:", cont.to_dict())

fig, axs = plt.subplots(2 if exemplo_nova else 1, 1, figsize=(16, 7.5 if exemplo_nova else 4), squeeze=False)
serie(exemplo_treino, "2025-01-01", "2025-12-31", axs[0, 0])
if exemplo_nova:
    serie(exemplo_nova, "2025-08-15", "2025-12-31", axs[1, 0])
axs[0, 0].legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.show()
""")

py("""
fig, axs = plt.subplots(4, 1, figsize=(16, 12))
for ax, mes in zip(axs, ["2018-01", "2018-04", "2018-07", "2018-10"]):
    serie(exemplo_treino, f"{mes}-01", pd.Timestamp(f"{mes}-01") + pd.offsets.MonthEnd(0), ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
axs[0].legend(ncol=3, loc="upper left")
plt.suptitle("Meses de teste de 2018 (nunca usados no treino)", y=1.0)
plt.tight_layout(); plt.show()
""")

md("""
### Todas as estações em 2025

Uma linha por estação com INMET em 2025, com o RMSE do ERA5 e da correção no título.
""")

py("""
ests = cont.reset_index().sort_values(["grupo_est", 0], ascending=[True, False]).estacao.tolist()
fig, axs = plt.subplots(len(ests), 1, figsize=(16, 2.4 * len(ests)), sharex=True)
for ax, e in zip(np.atleast_1d(axs), ests):
    serie(e, "2025-01-01", "2025-12-31", ax)
    d = j[(j.estacao == e) & (j.time.dt.year >= 2025)]
    ax.set_title(ax.get_title() + f" | n={len(d)} RMSE ERA5 {np.sqrt(((d.era5_gust_max - d[y]) ** 2).mean()):.2f}"
                 f" → corrigido {np.sqrt(((d.pred_gust_max - d[y]) ** 2).mean()):.2f}", fontsize=9)
np.atleast_1d(axs)[0].legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.show()
""")

md("""
### Uma célula sem estação

Sem INMET, só dá para ver se a correção é plausível: a célula com a maior correção média.
""")

py("""
dm = (pred - era5).mean("time")
k = np.unravel_index(int(np.nanargmax(np.abs(dm.values))), dm.shape)
lat0, lon0 = float(ds.lat[k[0]]), float(ds.lon[k[1]])
cel = ds.sel(lat=lat0, lon=lon0, time=slice("2025-01-01", "2025-12-31"))
fig, ax = plt.subplots(figsize=(16, 3.8))
ax.plot(cel.time, cel.era5_gust_max, color=CORES["ERA5"], lw=1, label="ERA5")
ax.plot(cel.time, cel.pred_gust_max, color=CORES["corrigido"], lw=1.2, label="corrigido")
ax.set_title(f"Célula {lat0}, {lon0} (sem estação), 2025 — correção média {float(dm[k]):+.2f} m/s em 2000–2025")
ax.legend(); plt.show()
""")

md("""
## 4. Otimizado no Optuna, mirando os extremos

O mesmo modelo de cada trimestre, com hiperparâmetros buscados no Optuna (`scripts/otimizar_extremos.py`).
Objetivo na **validação**: 0,5·RMSE + 0,5·RMSE_P90. O espaço inclui peso extra para os dias acima do P90 do
treino e perdas assimétricas (Expectile/Quantile no CatBoost, quantile no HistGradientBoosting). O teste só
foi usado no fim, para a comparação abaixo.

**Mapeado** = modelo padrão + mapeamento de quantis para o INMET, ajustado na validação
(`scripts/calibrar_quantis.py`): estica a distribuição comprimida do modelo sem mudar a ordem dos dias. É a
versão recomendada e a que aparece nos mapas abaixo.
""")

py("""
DIR = RAIZ / "artifacts/feature_study/inferencia"
ds_opt = xr.open_dataset(DIR / "results_feature_selction_optuna.nc")
# busca ampla (200 tentativas, espaço estendido), se já tiver rodado
AMPLO = (DIR / "results_feature_selction_optuna_amplo.nc").exists()
ds_amp = xr.open_dataset(DIR / "results_feature_selction_optuna_amplo.nc") if AMPLO else None
# padrão + mapeamento de quantis para o INMET (scripts/calibrar_quantis.py), se já tiver rodado
QMAP = (DIR / "results_feature_selction_qmap.nc").exists()
ds_qm = xr.open_dataset(DIR / "results_feature_selction_qmap.nc") if QMAP else None
# os mapas mostram a versão recomendada: o mapeado, se existir
ds_best = ds_qm if QMAP else (ds_amp if AMPLO else ds_opt)
pred_opt = ds_best.pred_gust_max.where(valido)
ROT_BEST = "mapeado" if QMAP else ("otimizado amplo" if AMPLO else "otimizado")
for suf, rot in [("", "40 tentativas"), *([("_amplo", "amplo, 200 tentativas")] if AMPLO else [])]:
    print(f"--- {rot}")
    for s_, v in json.loads((DIR / f"optuna{suf}_params.json").read_text()).items():
        print(f"{s_} {v['model']}: peso_cauda {v['peso_cauda']:.2f} acima de {v['lim_cauda_treino']:.1f} m/s | {v['params']}")
comp = pd.read_csv(DIR / "optuna_comparacao.csv")
if AMPLO:
    ca = pd.read_csv(DIR / "optuna_amplo_comparacao.csv")
    comp = pd.concat([comp, ca[ca.versao == "otimizado"].assign(versao="otimizado amplo")])
if QMAP:
    cq = pd.read_csv(DIR / "qmap_comparacao.csv")
    comp = pd.concat([comp, cq[cq.versao == "mapeado"]])
comp["versao"] = pd.Categorical(comp.versao, ["padrão", "otimizado", "otimizado amplo", "mapeado"])
comp = comp.sort_values(["season", "split", "versao"])
cols_ = ["RMSE", "RMSE_P90", "vies_P99", "P99_pred", "P99_obs", "POD_P95", "FAR_P95", "POD_P99", "FAR_P99"]
print("\\nTESTE do estudo (9 estações, meses jan/abr/jul/out), padrão × otimizado:")
print(comp[comp.split == "teste"].set_index(["season", "model", "versao"])[cols_].round(3).to_string())
""")

py("""
fig, axs = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
for j, (nome, f) in enumerate(ESTAT.items()):
    a, b, o = f(era5), f(pred), f(pred_opt)
    e = est_por_estacao(QS[nome])
    vmin = float(min(a.min(), o.min(), e.INMET.min())); vmax = float(max(a.max(), o.max(), e.INMET.max()))
    for i, (rot, campo) in enumerate([("ERA5 + INMET", a), (f"{ROT_BEST} + INMET", o)]):
        im = axs[i, j].pcolormesh(ds.lon, ds.lat, campo, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
        axs[i, j].scatter(e.lon, e.lat, c=e.INMET, cmap="viridis", vmin=vmin, vmax=vmax, s=170,
                          edgecolors="white", linewidths=1.8, zorder=3)
        axs[i, j].set_title(f"{rot} — {nome} (média {float(campo.mean()):.1f})")
    fig.colorbar(im, ax=axs[:2, j], shrink=0.8, label="m/s")
    d = o - b
    lim = float(np.abs(d).max())
    im = axs[2, j].pcolormesh(ds.lon, ds.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    axs[2, j].set_title(f"{ROT_BEST} − padrão — {nome} ({float(d.mean()):+.2f})")
    fig.colorbar(im, ax=axs[2, j], shrink=0.8, label="m/s")
    for ax in axs[:, j]:
        ax.set_aspect("equal")
fig.suptitle(f"{ROT_BEST} no Optuna: mapas com o INMET por cima (círculos = dias não vistos)", fontsize=13)
plt.show()
""")

md("""
### Contra o INMET, só dias não vistos (meses de teste e 2025)

Por estação (estatística da estação × da célula, mesmos dias) e por dia (RMSE, POD e FAR acima dos limiares
de P95 e P99 de cada estação).
""")

py("""
go = ds_opt[["pred_gust_max"]].to_dataframe().reset_index().dropna().rename(columns={"pred_gust_max": "otimizado"})
go["lat"], go["lon"] = go.lat.round(2), go.lon.round(2)
par3 = par.merge(go, on=["lat", "lon", "time"]).rename(columns={"pred_gust_max": "padrão", "era5_gust_max": "ERA5",
                                                                 "daily_wind_gust_max": "INMET"})
FONTES = {"ERA5": "#c44", "padrão": "#2a7ab0", "otimizado": "#1b9e77"}
if AMPLO:
    ga = ds_amp[["pred_gust_max"]].to_dataframe().reset_index().dropna().rename(columns={"pred_gust_max": "otimizado amplo"})
    ga["lat"], ga["lon"] = ga.lat.round(2), ga.lon.round(2)
    par3 = par3.merge(ga, on=["lat", "lon", "time"])
    FONTES["otimizado amplo"] = "#e08a00"
if QMAP:
    gq = ds_qm[["pred_gust_max"]].to_dataframe().reset_index().dropna().rename(columns={"pred_gust_max": "mapeado"})
    gq["lat"], gq["lon"] = gq.lat.round(2), gq.lon.round(2)
    par3 = par3.merge(gq, on=["lat", "lon", "time"])
    FONTES["mapeado"] = "#7b3294"

fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
lin = []
for ax, (nome, q) in zip(axs, QS.items()):
    agg = (lambda x: x.mean()) if q is None else (lambda x: x.quantile(q))
    e = par3.groupby("estacao")[["INMET", *FONTES]].agg(agg)
    for fnt, cor in FONTES.items():
        rm = float(np.sqrt(((e[fnt] - e.INMET) ** 2).mean()))
        a_, b_ = np.polyfit(e.INMET, e[fnt], 1)          # reta de ajuste: inclinação 1 = acompanha o INMET
        xx = np.array([e.INMET.min(), e.INMET.max()])
        ax.plot(xx, a_ * xx + b_, color=cor, lw=1.4, alpha=0.9)
        ax.scatter(e.INMET, e[fnt], c=cor, s=40, label=f"{fnt} ({rm:.2f}; incl. {a_:.2f})", zorder=3)
        lin.append({"estatística": nome, "fonte": fnt, "inclinação": a_, "intercepto": b_})
        lin[-1].update({"RMSE_entre_estacoes": rm, "viés": float((e[fnt] - e.INMET).mean())})
    lo, hi = float(e.min().min()) - 0.5, float(e.max().max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(nome); ax.set_xlabel("INMET (m/s)"); ax.set_ylabel("célula, mesmos dias (m/s)"); ax.legend(fontsize=7)
fig.suptitle("Estatística por estação: INMET × célula (entre parênteses: RMSE entre estações; inclinação da reta, 1 = ideal)")
plt.show()
print(pd.DataFrame(lin).pivot(index="estatística", columns="fonte").round(2).reindex(list(QS)).to_string())


def por_dia(d):
    out = {"n": len(d)}
    for fnt in FONTES:
        out[f"RMSE_{fnt}"] = float(np.sqrt(((d[fnt] - d.INMET) ** 2).mean()))
    for qn, qv in (("P95", 0.95), ("P99", 0.99)):
        lim = d.groupby("estacao").INMET.transform(lambda x: x.quantile(qv))
        ev = d.INMET >= lim
        for fnt in FONTES:
            al = d[fnt] >= lim
            out[f"POD_{qn}_{fnt}"] = float((ev & al).sum() / max(ev.sum(), 1))
            out[f"FAR_{qn}_{fnt}"] = float((~ev & al).sum() / max(al.sum(), 1))
    return pd.Series(out)


par3["season"] = par3.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
tab3 = pd.concat([por_dia(par3).rename("Todos"), *[por_dia(g).rename(k) for k, g in par3.groupby("season")]], axis=1).T
print("\\nPor dia, dias não vistos (limiar = P95/P99 do INMET de cada estação):")
print(tab3.round(3).to_string())
""")

py("""
fig, ax = plt.subplots(figsize=(6.5, 6.5))
qs = np.linspace(0.5, 0.999, 200)
qi = np.quantile(par3.INMET, qs)
for fnt, cor in FONTES.items():
    ax.plot(qi, np.quantile(par3[fnt], qs), color=cor, lw=2, label=fnt)
ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k--", lw=0.8)
for q in (0.9, 0.95, 0.99):
    ax.axvline(np.quantile(par3.INMET, q), color="#999", lw=0.6, ls=":")
ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil da grade (m/s)")
ax.set_title("Quantis da mediana ao P99,9 — dias não vistos\\n(abaixo da diagonal = corta os extremos)")
ax.legend(); plt.show()
""")

py("""
def serie3(estacao, ini, fim, ax):
    r = g4[g4.codigo_estacao == estacao].iloc[0]
    sel = dict(lat=r.grid_lat, lon=r.grid_lon, method="nearest")
    a = ds.sel(**sel).sel(time=slice(ini, fim)); o = ds_opt.sel(**sel).sel(time=slice(ini, fim))
    ob_ = inmet.sel(estacao=estacao).sel(time=slice(ini, fim))
    ax.plot(a.time, a.era5_gust_max, color=FONTES["ERA5"], lw=1, label="ERA5")
    ax.plot(a.time, a.pred_gust_max, color=FONTES["padrão"], lw=1.1, label="padrão")
    ax.plot(o.time, o.pred_gust_max, color=FONTES["otimizado"], lw=1.3, label="otimizado")
    if AMPLO:
        m_ = ds_amp.sel(**sel).sel(time=slice(ini, fim))
        ax.plot(m_.time, m_.pred_gust_max, color=FONTES["otimizado amplo"], lw=1.3, label="otimizado amplo")
    if QMAP:
        q_ = ds_qm.sel(**sel).sel(time=slice(ini, fim))
        ax.plot(q_.time, q_.pred_gust_max, color=FONTES["mapeado"], lw=1.6, label="mapeado")
    ax.plot(ob_.time, ob_, color="#111", lw=1.1, marker=".", ms=3, label="INMET")
    ax.set_title(f"{estacao} — {ini[:7]} a {fim[:7]}"); ax.set_ylabel("m/s")


fig, axs = plt.subplots(3, 1, figsize=(16, 11))
serie3(exemplo_treino, "2025-01-01", "2025-12-31", axs[0])
serie3(exemplo_treino, "2018-07-01", "2018-07-31", axs[1])
serie3(exemplo_treino, "2018-10-01", "2018-10-31", axs[2])
axs[0].legend(ncol=6, loc="upper left")
plt.tight_layout(); plt.show()
""")

md("""
## Limites

1. **Fora das estações só há plausibilidade, não acerto.** O modelo aprendeu com 9 estações.
2. **`anor_*` identifica estação** (portão das estáticas permutadas). Em células sem estação esses valores são
   novos para o modelo; desconfie de células onde a correção destoa das vizinhas.
3. **A hora do pico vem do ERA5.** É a rajada máxima diária, não uma série horária.
4. **As 12 estações novas têm poucos dias** (desde set/out de 2025): o ganho delas tem muita incerteza.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
