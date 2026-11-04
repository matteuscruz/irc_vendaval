"""Gera `notebooks/feature_study_grupos/atual/regional_rs_sc.ipynb`: o modelo do cluster 3 em regiões novas e os
modelos novos de RS/SC (por cluster, regional e regional-LOCO), junto com o resultado do cluster 12.

Lê `artifacts/feature_study/rs_sc/` (gerado por `scripts/treinar_rs_sc.py`) e
`artifacts/feature_study/inferencia_c12/` (`scripts/inferencia_cluster.py --cluster 12`). Nada é treinado aqui.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/regional_rs_sc.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# RS/SC: o modelo do cluster 3 em regiões novas e modelos treinados nos locais novos

ERA5 bruto de RS/SC (`dataset/raw/era5_rs_sc`, 8 variáveis, 2000–2025) → as 49 colunas do **Corrigido ML**
(selecionadas, sem `anor`), uma linha por (estação ou célula, dia) na hora do pico da rajada ERA5. Clusters
avaliados: **1, 2, 3 e 4** (os 5 e 6 só aparecem em parte na grade) e o **12**, do teste anterior.

| desenho | treina com | o que mede |
|---|---|---|
| **A** modelo do cluster 3 | cluster 3 (estudo) | generalização do modelo atual para região nova, sem treino |
| **B** por cluster | só as estações do cluster | o melhor possível com dado local; a matriz treino × teste mostra a transferência |
| **C** regional | estações dos clusters 1–4 juntas | um modelo único para RS/SC |
| **L** regional-LOCO | os outros três clusters | o regional numa região que ele **nunca viu** (cluster inteiro de fora) |

Em todos: os campeões do cluster 3 por trimestre (MLP DJF, ExtraTrees MAM, CatBoost JJA, HistGB SON), seed 42,
EQM ajustado na validação do próprio treino (`documentation/11_mapeamento_de_quantis.md`). Avaliação **só em dias
que nenhum treino viu**: meses de teste (jan/abr/jul/out de 2000–2024) e 2025. O split é só do calendário, então
os dias de teste são os mesmos para todos os clusters.
""")

py("""
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.diagnostics import regional as rg

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
RS = RAIZ / "artifacts/feature_study/rs_sc"
C12 = RAIZ / "artifacts/feature_study/inferencia_c12"
TARGET = "daily_wind_gust_max"

mt = pd.read_csv(RS / "metricas.csv")
pe = pd.read_parquet(RS / "previsoes_estacoes.parquet")
est = pd.read_csv(RS / "estacoes.csv")
par = json.loads((RS / "params.json").read_text())
inmet = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")[TARGET]
shp = gpd.read_file(RAIZ / "dataset/shp/shp_vento.shp").to_crs(4326)
print("campeões:", par["campeoes"])
print("estações por cluster (com alvo):", est[est.dias_inmet > 0].groupby("cluster_id").size().to_dict())
print("linhas de treino por desenho (soma dos trimestres):",
      {k: sum(v[s]["n_train"] for s in v) for k, v in par["modelos"].items()})
""")

md("## 1. Onde estão os clusters e as estações")

py("""
e12 = pd.read_csv(C12 / "estacoes.csv")
fig, ax = plt.subplots(figsize=(9, 10))
shp.plot(ax=ax, color="#eeeeee", edgecolor="#999", lw=0.4)
cores = {1: "#2a7ab0", 2: "#d98c3f", 3: "#5aa469", 4: "#8a6fb0", 12: "#c44"}
for cid, cor in cores.items():
    shp[shp.cluster.astype(int) == cid].plot(ax=ax, color=cor, alpha=0.45, edgecolor=cor)
    p = shp[shp.cluster.astype(int) == cid].geometry.iloc[0].representative_point()
    ax.text(p.x, p.y, str(cid), fontsize=13, weight="bold", ha="center")
ax.scatter(est.longitude, est.latitude, s=14, c="k", marker="^", label="estações RS/SC (clusters 1–4)")
ax.scatter(e12.longitude, e12.latitude, s=14, c="#c44", marker="^", label="estações do cluster 12")
ax.add_patch(plt.Rectangle((-57.75, -34), 9.75, 8.25, fill=False, ls="--", lw=1.2, label="grade ERA5 RS/SC"))
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Clusters avaliados (cor) e estações INMET"); plt.show()
""")

md("""
## 2. O modelo do cluster 3 (desenho A) em regiões novas, junto com o cluster 12

Sem treino novo. O cluster 3 entra como referência (região do treino, só dias não vistos). Mesmas métricas e mesmos
recortes nos cinco clusters.
""")

py("""
# cluster 12: as mesmas métricas, a partir do .nc do teste anterior e do INMET, nos mesmos recortes
ds12 = xr.open_dataset(C12 / "results_cluster12.nc")
g12 = ds12[["era5_gust_max", "pred_padrao_sem_anor", "pred_mapeado_sem_anor"]].to_dataframe().reset_index().dropna()
g12["lat"], g12["lon"] = g12.lat.round(3), g12.lon.round(3)
o12 = inmet.sel(estacao=e12.estacao.tolist()).to_dataframe().reset_index()[["estacao", "time", TARGET]].dropna()
o12 = o12.merge(e12.assign(grid_lat=e12.grid_lat.round(3), grid_lon=e12.grid_lon.round(3))[["estacao", "grid_lat", "grid_lon"]],
                on="estacao").merge(g12, left_on=["grid_lat", "grid_lon", "time"], right_on=["lat", "lon", "time"])
o12["season"] = o12.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
o12["_split"] = np.where(o12.time.dt.year >= 2025, "novo",
                         np.where(o12.time.dt.month.isin((1, 4, 7, 10)), "test", "train"))
o12["cluster_id"] = 12
mt12 = rg.evaluate(o12, {"modelo": o12.pred_padrao_sem_anor.to_numpy(), "Corrigido ML": o12.pred_mapeado_sem_anor.to_numpy()},
                   rotulo={"desenho": "A: modelo do cluster 3 (estudo)", "treino": "3 (estudo)"})
A = pd.concat([mt[mt.desenho.str.startswith("A")], mt12], ignore_index=True)

def quadro(d, recorte="teste 2000–2024"):
    q = d[(d.trimestre == "Todos") & (d.recorte == recorte)].pivot_table(
        index="cluster", columns="versao", values=["RMSE", "corr", "POD_P99", "FAR_P99", "n"])
    out = pd.DataFrame({"n": q["n"]["ERA5"], "RMSE ERA5": q["RMSE"]["ERA5"], "RMSE Corrigido": q["RMSE"]["Corrigido ML"],
                        "ganho": q["RMSE"]["ERA5"] - q["RMSE"]["Corrigido ML"],
                        "corr ERA5": q["corr"]["ERA5"], "corr Corrigido": q["corr"]["Corrigido ML"],
                        "POD P99 ERA5": q["POD_P99"]["ERA5"], "POD P99 Corrigido": q["POD_P99"]["Corrigido ML"]})
    return out

for rec in ("teste 2000–2024", "2025"):
    print(f"--- desenho A, {rec} (ganho > 0 = Corrigido ML erra menos que o ERA5)")
    print(quadro(A, rec).round(3).to_string())

qa = quadro(A)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
x = np.arange(len(qa))
axs[0].bar(x, qa.ganho, color=["#5aa469" if v > 0 else "#c44" for v in qa.ganho])
axs[0].axhline(0, color="k", lw=0.7); axs[0].set_title("ganho de RMSE sobre o ERA5 (m/s)")
for ax, (a, b, t) in zip(axs[1:], [("corr ERA5", "corr Corrigido", "correlação com o INMET"),
                                   ("POD P99 ERA5", "POD P99 Corrigido", "POD no P99")]):
    ax.bar(x - 0.2, qa[a], 0.4, color="#c44", label="ERA5"); ax.bar(x + 0.2, qa[b], 0.4, color="#2a7ab0", label="Corrigido ML")
    ax.set_title(t); ax.legend(fontsize=8)
for ax in axs:
    ax.set_xticks(x, [f"cluster {i}" + (" (treino)" if i == 3 else "") for i in qa.index], fontsize=8)
fig.suptitle("Desenho A: o modelo do cluster 3 aplicado a cada cluster (meses de teste 2000–2024)")
plt.show()
""")

md("""
## 3. Um modelo por cluster (desenho B): a matriz treino × teste

Cada linha é o cluster usado no treino; cada coluna, o cluster avaliado. A diagonal é o modelo local; fora
dela, a transferência. Valor = ganho de RMSE do Corrigido ML sobre o ERA5 (verde = melhor que o ERA5).
""")

py("""
B = mt[mt.desenho.str.startswith("B") & (mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")]
fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
for ax, (versao, met, titulo, cmap, sinal) in zip(axs, [
        ("Corrigido ML", "RMSE", "ganho de RMSE sobre o ERA5 — Corrigido ML", "RdYlGn", 1),
        ("modelo", "RMSE", "ganho de RMSE sobre o ERA5 — modelo sem EQM", "RdYlGn", 1),
        ("Corrigido ML", "POD_P99", "POD P99 do Corrigido ML menos o do ERA5", "RdYlGn", -1)]):
    v = B[B.versao == versao].pivot_table(index="treino", columns="cluster", values=met)
    e = B[B.versao == "ERA5"].pivot_table(index="treino", columns="cluster", values=met)
    m = (e - v) if sinal == 1 else (v - e)
    lim = float(np.nanmax(np.abs(m.values)))
    im = ax.imshow(m.values, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    weight="bold" if m.index[i] == str(m.columns[j]) else None)
    ax.set_xticks(range(m.shape[1]), [f"avalia {c_}" for c_ in m.columns]); ax.set_yticks(range(m.shape[0]), [f"treina {t}" for t in m.index])
    ax.set_title(titulo, fontsize=10); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Desenho B: um modelo por cluster, avaliado em todos (meses de teste 2000–2024; diagonal em negrito = local)")
plt.show()
""")

md("""
## 4. Comparação por cluster: A, B (local), C (regional) e L (regional sem o cluster)

Para cada cluster: o RMSE e o POD P99 do Corrigido ML em cada desenho, contra o ERA5. **L** é a pergunta da
generalização: o regional numa região que ficou inteira de fora do treino.
""")

py("""
def desenho_curto(r):
    return {"A": "A cl.3", "B": "B local", "C": "C regional", "L": "L LOCO"}[r.desenho[0]]

sel = mt[(mt.trimestre == "Todos")].copy()
sel = sel[~(sel.desenho.str.startswith("B") & (sel.treino != sel.cluster.astype(str)))]       # B só o local
sel["d"] = sel.apply(desenho_curto, axis=1)
ORD = ["A cl.3", "B local", "C regional", "L LOCO"]
for rec in ("teste 2000–2024", "2025"):
    s_ = sel[sel.recorte == rec]
    era = s_[s_.versao == "ERA5"].groupby("cluster")[["RMSE", "POD_P99", "corr"]].first()
    cm = s_[s_.versao == "Corrigido ML"].pivot_table(index="cluster", columns="d", values="RMSE")[ORD]
    sm = s_[s_.versao == "modelo"].pivot_table(index="cluster", columns="d", values="RMSE")[ORD]
    pod = s_[s_.versao == "Corrigido ML"].pivot_table(index="cluster", columns="d", values="POD_P99")[ORD]
    print(f"=== {rec}: RMSE (ERA5 | Corrigido ML por desenho | modelo sem EQM por desenho)")
    print(pd.concat({"ERA5": era.RMSE, **{f"CML {k}": cm[k] for k in ORD}, **{f"mod {k}": sm[k] for k in ORD}}, axis=1).round(3).to_string())
    print(f"--- {rec}: POD P99 (ERA5 | Corrigido ML por desenho)")
    print(pd.concat({"ERA5": era.POD_P99, **{k: pod[k] for k in ORD}}, axis=1).round(3).to_string())

s_ = sel[sel.recorte == "teste 2000–2024"]
fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), constrained_layout=True)
cor_d = {"ERA5": "#c44", "A cl.3": "#999999", "B local": "#2a7ab0", "C regional": "#5aa469", "L LOCO": "#e08a00"}
cls = sorted(s_.cluster.unique())
for ax, met, tit in zip(axs, ["RMSE", "POD_P99"], ["RMSE diário (m/s)", "POD no P99"]):
    w = 0.16
    era = s_[s_.versao == "ERA5"].groupby("cluster")[met].first()
    ax.bar(np.arange(len(cls)) - 2 * w, era.reindex(cls), w, color=cor_d["ERA5"], label="ERA5")
    for k, d_ in enumerate(ORD):
        v = s_[(s_.versao == "Corrigido ML") & (s_.d == d_)].set_index("cluster")[met].reindex(cls)
        ax.bar(np.arange(len(cls)) + (k - 1) * w, v, w, color=cor_d[d_], label=f"Corrigido ML — {d_}")
    ax.set_xticks(range(len(cls)), [f"cluster {i}" for i in cls]); ax.set_title(tit); ax.legend(fontsize=7)
fig.suptitle("Corrigido ML por desenho em cada cluster (meses de teste 2000–2024)")
plt.show()
""")

py("""
print("Regional (C) e LOCO (L), Corrigido ML, ganho de RMSE sobre o ERA5 por trimestre (teste 2000–2024):")
for d_ in ("C", "L"):
    s_ = mt[mt.desenho.str.startswith(d_) & (mt.recorte == "teste 2000–2024")]
    g = (s_[s_.versao == "ERA5"].pivot_table(index="cluster", columns="trimestre", values="RMSE")
         - s_[s_.versao == "Corrigido ML"].pivot_table(index="cluster", columns="trimestre", values="RMSE"))
    print(f"--- {d_}"); print(g[["Todos", "DJF", "MAM", "JJA", "SON"]].round(3).to_string())
""")

md("""
## 5. Mapas por cluster: ERA5, regional (C) e regional sem o cluster (L)

Média e P99 da série diária de 2000–2025 em cada célula do polígono. Círculos: o INMET de cada estação **nos dias
não vistos** (meses de teste e 2025), na mesma escala de cor.
""")

py("""
TESTE = pe._split.isin(["test", "novo"])
for cid in (1, 2, 3, 4):
    ds = xr.open_dataset(RS / f"results_rs_sc_c{cid}.nc")
    e_ = est[(est.cluster_id == cid) & (est.dias_inmet > 0)].set_index("estacao")
    ob = pe[TESTE & (pe.cluster_id == cid)].groupby("estacao")[TARGET]
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    for i, (nome, f) in enumerate([("média", lambda x: x.mean("time")), ("P99", lambda x: x.quantile(0.99, "time"))]):
        campos = {"ERA5": f(ds.era5_gust_max), "regional (C)": f(ds.pred_C), "sem o cluster (L)": f(ds.pred_L)}
        q_ = ob.mean() if i == 0 else ob.quantile(0.99)
        vmin = float(min(*(v.min() for v in campos.values()), q_.min())); vmax = float(max(*(v.max() for v in campos.values()), q_.max()))
        for j, (rot, campo) in enumerate(campos.items()):
            im = axs[i, j].pcolormesh(ds.lon, ds.lat, campo, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
            axs[i, j].scatter(e_.loc[q_.index, "longitude"], e_.loc[q_.index, "latitude"], c=q_.values, cmap="viridis",
                              vmin=vmin, vmax=vmax, s=90, edgecolors="white", linewidths=1.4, zorder=3)
            axs[i, j].set_title(f"{rot} — {nome} ({float(campo.mean()):.1f})", fontsize=9)
        fig.colorbar(im, ax=axs[i, :3], shrink=0.8, label="m/s")
        d = campos["regional (C)"] - campos["ERA5"]; lim = float(np.abs(d).max())
        im = axs[i, 3].pcolormesh(ds.lon, ds.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        axs[i, 3].set_title(f"regional − ERA5 — {nome} ({float(d.mean()):+.2f})", fontsize=9)
        fig.colorbar(im, ax=axs[i, 3], shrink=0.8, label="m/s")
    for ax in axs.flat:
        ax.set_aspect("equal")
    fig.suptitle(f"Cluster {cid}: ERA5 × Corrigido ML regional (INMET por cima, dias não vistos)", fontsize=12)
    plt.show()
""")

md("## 6. Quantis por cluster (dias não vistos)")

py("""
VERS = {"ERA5": ("era5_gust_max", "#c44"), "A cl.3": ("pred_A", "#999999"), "B local": ("pred_B", "#2a7ab0"),
        "C regional": ("pred_C", "#5aa469"), "L LOCO": ("pred_L", "#e08a00")}
fig, axs = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
qs = np.linspace(0.5, 0.999, 200)
for ax, cid in zip(axs, (1, 2, 3, 4)):
    d = pe[TESTE & (pe.cluster_id == cid)].dropna(subset=[v for v, _ in VERS.values()])
    qi = np.quantile(d[TARGET], qs)
    for k, (v, cor) in VERS.items():
        ax.plot(qi, np.quantile(d[v], qs), color=cor, lw=2, label=k)
    ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k--", lw=0.8)
    ax.set_title(f"cluster {cid} (n={len(d)})"); ax.set_xlabel("quantil do INMET (m/s)")
axs[0].set_ylabel("mesmo quantil da previsão (m/s)"); axs[0].legend(fontsize=8)
fig.suptitle("Quantis da mediana ao P99,9 — abaixo da diagonal = corta os extremos"); plt.show()
""")

md("""
## Limites

1. **Campeões do cluster 3.** O modelo de cada trimestre foi escolhido no cluster 3 e não foi retriado nos clusters
   novos; outro modelo pode ser melhor em outra região.
2. **Cluster 1 tem só 4 estações** com dados: o desenho B dele é frágil e a média entre estações (inclinação) é
   pouco informativa.
3. **Cluster 4**: o polígono passa um pouco da borda norte da grade (−25,6° contra −25,75°): a vizinhança de 250 km
   das células do norte fica cortada.
4. **Vizinhança depende da grade baixada.** As colunas `_r75`/`_r250` usam todas as células da grade bruta; na grade
   RS/SC o cluster 3 tem mais vizinhos do que na grade original dele, então as colunas de vizinhança das células perto
   da borda antiga diferem um pouco das do `test_inference_cluster_3.parquet`.
5. **`mcpr_r75_fracpos`, `mcpr_l75_250` e `grad_mslp_*`** seguem a definição dos dicionários, sem conferência
   contra o pipeline original.
6. **Uma seed** (42) por ajuste: diferenças pequenas entre desenhos (poucos centésimos de m/s) estão dentro do ruído.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
