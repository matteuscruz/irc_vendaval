"""Gera `notebooks/feature_study_grupos/atual/comparacao_abordagens.ipynb`: todas as abordagens testadas, lado a lado.

Lê apenas o que já foi gravado (nada é treinado):
  - estudo de features (Modal + local): ML grupo 1, ML selecionadas, LSTM base e selecionadas (teste do estudo);
  - `artifacts/feature_study/inferencia/`: padrão, EQM, Optuna 40 e Optuna 200 (cluster 3, grade);
  - `artifacts/feature_study/inferencia_c3/`: sem `anor` e Corrigido ML (sem `anor` + EQM);
  - `artifacts/feature_study/rs_sc/`: local (B), regional (C) e LOCO (L) em RS/SC, e as métricas por cluster;
  - `artifacts/feature_study/inferencia_c12/`: o modelo do cluster 3 no cluster 12.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/comparacao_abordagens.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Comparação das abordagens

Todas as abordagens testadas para corrigir a rajada do ERA5 contra o INMET, **nos mesmos dias, nas mesmas estações e
com as mesmas métricas**. Nada é treinado neste notebook: tudo vem do que já foi gravado.

| abordagem | o que é | onde foi feita |
|---|---|---|
| **ERA5** | rajada bruta (`gust10fg`) na hora do pico do dia | — |
| **ML grupo 1** | melhor modelo por trimestre só com o grupo 1 (arm `base`) | estudo de features |
| **ML selecionadas** | o mesmo modelo com as 58 colunas selecionadas (`sel__val12`) | estudo de features |
| **LSTM base / selecionadas** | LSTM (janela de 24 h) com o grupo 1 / com as selecionadas | estudo de features |
| **ML + EQM** | ML selecionadas + mapeamento de quantis | `calibrar_quantis.py` |
| **Optuna 40 / 200** | ML selecionadas com hiperparâmetros buscados para os extremos (40 e 200 tentativas) | `otimizar_extremos.py` |
| **ML sem anor** | ML com as 49 colunas sem `anor` | `inferencia_cluster.py` |
| **Corrigido ML** | ML sem anor + EQM (a versão recomendada para região nova) | `inferencia_cluster.py` |
| **Local (B)** | Corrigido ML treinado só com as estações do próprio cluster | `treinar_rs_sc.py` |
| **Regional (C)** | Corrigido ML treinado com os clusters 1–4 de RS/SC juntos | `treinar_rs_sc.py` |
| **LOCO (L)** | regional treinado SEM o cluster avaliado (região nunca vista) | `treinar_rs_sc.py` |

Na seção 1, B, C e L aparecem **com EQM** (é o que foi salvo por estação-dia); as versões sem EQM deles estão na
seção 2, por cluster.

Todas usam os mesmos modelos por trimestre (os campeões do estudo: MLP DJF, ExtraTrees MAM, CatBoost JJA, HistGB SON),
exceto a LSTM. Avaliação **só em dias que nenhum modelo viu**: meses de teste (jan/abr/jul/out de 2000–2024) e 2025.
""")

py("""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.feature_study.diagnostics import predictor_curves as pc
from src.feature_study.diagnostics import regional as rg

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
A = RAIZ / "artifacts/feature_study"
TARGET = "daily_wind_gust_max"
TESTE = (1, 4, 7, 10)
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON"}
""")

md("""
## 1. Cluster 3: todas as abordagens nos mesmos dias

As 9 estações do cluster 3 com INMET. Cada abordagem é lida na célula ERA5 da estação e casada por (estação, dia).
Nos meses de teste entram todas; em 2025 a LSTM e os ML do estudo não existem (o teste do estudo vai até 2024).
""")

py("""
g4 = pd.read_parquet(RAIZ / "dataset/raw/feature_study_cluster_3/features_grupo4_cluster3.parquet",
                     columns=["codigo_estacao", "grid_lat", "grid_lon"]).rename(columns={"codigo_estacao": "estacao"})
g4["lat"], g4["lon"] = g4.grid_lat.round(2), g4.grid_lon.round(2)


def da_grade(path, var, rotulo):
    d = xr.open_dataset(path)[var].to_dataframe().reset_index().dropna(subset=[var])
    d["lat"], d["lon"] = d.lat.round(2), d.lon.round(2)
    return g4[["estacao", "lat", "lon"]].merge(d, on=["lat", "lon"])[["estacao", "time", var]].rename(columns={var: rotulo})


INF = A / "inferencia"
fontes = [
    da_grade(INF / "results_feature_selction.nc", "era5_gust_max", "ERA5"),
    da_grade(INF / "results_feature_selction.nc", "pred_gust_max", "ML selecionadas (grade)"),
    da_grade(INF / "results_feature_selction_qmap.nc", "pred_gust_max", "ML + EQM"),
    da_grade(INF / "results_feature_selction_optuna.nc", "pred_gust_max", "Optuna 40"),
    da_grade(INF / "results_feature_selction_optuna_amplo.nc", "pred_gust_max", "Optuna 200"),
    da_grade(A / "inferencia_c3/results_cluster3.nc", "pred_padrao_sem_anor", "ML sem anor"),
    da_grade(A / "inferencia_c3/results_cluster3.nc", "pred_mapeado_sem_anor", "Corrigido ML"),
]
pe = pd.read_parquet(A / "rs_sc/previsoes_estacoes.parquet")
pe3 = pe[pe.cluster_id == 3][["estacao", "time", "pred_B", "pred_C", "pred_L"]].rename(
    columns={"pred_B": "Local (B) + EQM", "pred_C": "Regional (C) + EQM", "pred_L": "LOCO (L) + EQM"})
# só as versões com EQM foram salvas por estação-dia; as sem EQM de B/C/L estão na seção 2 (por cluster)

obs = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")[TARGET].sel(estacao=g4.estacao.tolist())
obs = obs.to_dataframe().reset_index()[["estacao", "time", TARGET]].dropna().rename(columns={TARGET: "INMET"})
c3 = obs
for f in fontes:
    c3 = c3.merge(f, on=["estacao", "time"], how="inner")
c3 = c3.merge(pe3, on=["estacao", "time"], how="left")

# estudo de features: ML grupo 1 / selecionadas e LSTM, nas linhas de teste do estudo
MODAL, LOCAL = A / "cluster3_groups_modal", A / "cluster3_groups"
est = pc.load_predictors(MODAL, LOCAL, pc.champions(MODAL / "summary"))
tst = pd.read_parquet(MODAL / "data/test.parquet", columns=["row_id", "estacao", "time"])
est = est.merge(tst, on="row_id").rename(columns={"ML, base": "ML grupo 1", "ML, selecionadas": "ML selecionadas",
                                                  "LSTM, base": "LSTM base", "LSTM, selecionadas": "LSTM selecionadas"})
c3 = c3.merge(est[["estacao", "time", "ML grupo 1", "ML selecionadas", "LSTM base", "LSTM selecionadas"]],
              on=["estacao", "time"], how="left")
c3["season"] = c3.time.dt.month.map(SEASON)
c3["recorte"] = np.select([c3.time.dt.year >= 2025, c3.time.dt.month.isin(TESTE)], ["2025", "teste 2000–2024"], "treino")
c3 = c3[c3.recorte != "treino"].reset_index(drop=True)

ABORD = ["ERA5", "ML grupo 1", "ML selecionadas", "LSTM base", "LSTM selecionadas", "ML + EQM", "Optuna 40",
         "Optuna 200", "ML sem anor", "Corrigido ML", "Local (B) + EQM", "Regional (C) + EQM", "LOCO (L) + EQM"]
FAMILIA = {"ERA5": "ERA5", "ML grupo 1": "estudo", "ML selecionadas": "estudo", "LSTM base": "LSTM",
           "LSTM selecionadas": "LSTM", "ML + EQM": "EQM", "Optuna 40": "Optuna", "Optuna 200": "Optuna",
           "ML sem anor": "sem anor", "Corrigido ML": "EQM", "Local (B) + EQM": "treino RS/SC",
           "Regional (C) + EQM": "treino RS/SC", "LOCO (L) + EQM": "treino RS/SC"}
COR = {"ERA5": "#c44", "estudo": "#2a7ab0", "LSTM": "#17808a", "EQM": "#7b3294", "Optuna": "#1b9e77",
       "sem anor": "#8a6fb0", "treino RS/SC": "#e08a00"}
t3 = c3[c3.recorte == "teste 2000–2024"]
print("meses de teste 2000–2024:", len(t3), "estação-dias,", t3.estacao.nunique(), "estações |",
      "linhas com todas as abordagens:", int(t3[ABORD].notna().all(axis=1).sum()))
print("2025:", int((c3.recorte == "2025").sum()), "estação-dias")
""")

py("""
def metricas(d, col):
    ok = d[col].notna() & d.INMET.notna()
    d = d[ok]
    y, p = d.INMET.to_numpy(float), d[col].to_numpy(float)
    out = {"n": len(d), "RMSE": np.sqrt(np.mean((p - y) ** 2)), "viés": np.mean(p - y), "corr": np.corrcoef(p, y)[0, 1],
           "desvio prev/obs": np.std(p) / np.std(y), "P99 prev − obs": np.quantile(p, 0.99) - np.quantile(y, 0.99)}
    for nome, q in (("P95", 0.95), ("P99", 0.99)):
        lim = d.groupby("estacao").INMET.transform(lambda x: x.quantile(q)).to_numpy()
        ev, al = y >= lim, p >= lim
        out[f"POD {nome}"] = (ev & al).sum() / max(ev.sum(), 1)
        out[f"FAR {nome}"] = (~ev & al).sum() / max(al.sum(), 1)
    e = pd.DataFrame({"e": d.estacao, "y": y, "p": p}).groupby("e").mean()
    out["inclinação estações"] = np.polyfit(e.y, e.p, 1)[0]
    return out


# mesmas linhas para todas: só os dias em que todas as abordagens existem
comum = t3.dropna(subset=ABORD)
tab = pd.DataFrame({a: metricas(comum, a) for a in ABORD}).T
tab["ganho RMSE"] = tab.loc["ERA5", "RMSE"] - tab.RMSE
print(f"Cluster 3, meses de teste 2000–2024, {len(comum)} estação-dias comuns a todas as abordagens")
print(tab[["RMSE", "ganho RMSE", "viés", "corr", "desvio prev/obs", "P99 prev − obs", "POD P95", "FAR P95",
           "POD P99", "FAR P99", "inclinação estações"]].round(3).to_string())
""")

py("""
fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
ordem = tab.sort_values("RMSE").index
cores = [COR[FAMILIA[a]] for a in ordem]
axs[0].barh(ordem, tab.loc[ordem, "RMSE"], color=cores); axs[0].invert_yaxis()
axs[0].axvline(tab.loc["ERA5", "RMSE"], color="#c44", ls="--", lw=1); axs[0].set_xlabel("RMSE diário (m/s)")
axs[0].set_title("erro diário (menor = melhor)")
ordem2 = tab.sort_values("POD P99", ascending=False).index
axs[1].barh(ordem2, tab.loc[ordem2, "POD P99"], color=[COR[FAMILIA[a]] for a in ordem2]); axs[1].invert_yaxis()
axs[1].axvline(tab.loc["ERA5", "POD P99"], color="#c44", ls="--", lw=1); axs[1].set_xlabel("POD no P99")
axs[1].set_title("extremos detectados (maior = melhor)")
for a in ABORD:
    axs[2].scatter(tab.loc[a, "RMSE"], tab.loc[a, "POD P99"], s=80, color=COR[FAMILIA[a]], edgecolor="k", zorder=3)
    axs[2].annotate(a, (tab.loc[a, "RMSE"], tab.loc[a, "POD P99"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
axs[2].set_xlabel("RMSE diário (m/s) → pior"); axs[2].set_ylabel("POD P99 → melhor")
axs[2].set_title("a troca: acerto diário × extremos (canto superior esquerdo = ideal)")
fig.suptitle("Cluster 3: todas as abordagens nos mesmos dias de teste (cor = família da abordagem)", fontsize=12)
plt.show()
""")

md("""
### Por trimestre

Ganho de RMSE sobre o ERA5 (verde = melhor que o ERA5) e POD no P99, em cada trimestre, nas mesmas linhas.
""")

py("""
gan, pod = {}, {}
for s in ("DJF", "MAM", "JJA", "SON"):
    d = comum[comum.season == s]
    m = {a: metricas(d, a) for a in ABORD}
    gan[s] = {a: m["ERA5"]["RMSE"] - m[a]["RMSE"] for a in ABORD}
    pod[s] = {a: m[a]["POD P99"] for a in ABORD}
gan, pod = pd.DataFrame(gan).loc[ABORD], pd.DataFrame(pod).loc[ABORD]
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
for ax, (m, tit, cmap, centro) in zip(axs, [(gan, "ganho de RMSE sobre o ERA5 (m/s)", "RdYlGn", 0),
                                            (pod, "POD no P99", "Greens", None)]):
    lim = float(np.abs(m.values).max())
    im = ax.imshow(m.values, cmap=cmap, vmin=-lim if centro == 0 else 0, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:+.2f}" if centro == 0 else f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4), m.columns); ax.set_yticks(range(len(m)), m.index); ax.grid(False); ax.set_title(tit)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle("Cluster 3, por trimestre (meses de teste 2000–2024)"); plt.show()
""")

py("""
fig, ax = plt.subplots(figsize=(7, 7))
qs = np.linspace(0.5, 0.999, 200)
qi = np.quantile(comum.INMET, qs)
DEST = ["ERA5", "ML selecionadas", "LSTM base", "ML + EQM", "Optuna 40", "Corrigido ML", "Regional (C) + EQM"]
for a in DEST:
    ax.plot(qi, np.quantile(comum[a], qs), lw=2.2 if a in ("ERA5", "Corrigido ML") else 1.4, color=COR[FAMILIA[a]],
            ls="--" if a in ("LSTM base", "Optuna 40") else "-", label=a)
ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k:", lw=1)
ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil da abordagem (m/s)")
ax.set_title("Quantis da mediana ao P99,9 (abaixo da diagonal = corta os extremos)"); ax.legend(fontsize=8)
plt.show()
""")

py("""
n25 = c3[c3.recorte == "2025"]
AB25 = [a for a in ABORD if n25[a].notna().any()]
comum25 = n25.dropna(subset=AB25)
t25 = pd.DataFrame({a: metricas(comum25, a) for a in AB25}).T
t25["ganho RMSE"] = t25.loc["ERA5", "RMSE"] - t25.RMSE
print(f"Cluster 3, 2025 (ano nunca visto), {len(comum25)} estação-dias; sem LSTM e ML do estudo (o teste deles vai até 2024)")
print(t25[["RMSE", "ganho RMSE", "corr", "POD P95", "FAR P95", "POD P99", "FAR P99"]].round(3).to_string())
""")

md("""
## 2. Generalização: as abordagens em outros clusters

Métricas por cluster, nos meses de teste de 2000–2024. **A** = o modelo do cluster 3 (Corrigido ML) sem treino novo;
**B** = treinado no próprio cluster; **C** = regional RS/SC; **L** = regional sem o cluster avaliado (região nunca
vista). O cluster 12 só tem A (não há modelo treinado lá). Valores com e sem EQM.
""")

py("""
mt = pd.read_csv(A / "rs_sc/metricas.csv")
mt = mt[(mt.trimestre == "Todos")]
mt = mt[~(mt.desenho.str.startswith("B") & (mt.treino != mt.cluster.astype(str)))]          # B: só o local
mt["abordagem"] = mt.desenho.str[0].map({"A": "A cluster 3", "B": "B local", "C": "C regional", "L": "L LOCO"})

# cluster 12, mesmas métricas (desenho A), a partir do .nc e do INMET
C12 = A / "inferencia_c12"
e12 = pd.read_csv(C12 / "estacoes.csv")
g12 = xr.open_dataset(C12 / "results_cluster12.nc")[["era5_gust_max", "pred_padrao_sem_anor", "pred_mapeado_sem_anor"]]
g12 = g12.to_dataframe().reset_index().dropna()
g12["lat"], g12["lon"] = g12.lat.round(3), g12.lon.round(3)
o12 = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")[TARGET].sel(estacao=e12.estacao.tolist())
o12 = o12.to_dataframe().reset_index()[["estacao", "time", TARGET]].dropna()
o12 = o12.merge(e12.assign(lat=e12.grid_lat.round(3), lon=e12.grid_lon.round(3))[["estacao", "lat", "lon"]], on="estacao")
o12 = o12.merge(g12, on=["lat", "lon", "time"])
o12["season"] = o12.time.dt.month.map(SEASON)
o12["_split"] = np.where(o12.time.dt.year >= 2025, "novo", np.where(o12.time.dt.month.isin(TESTE), "test", "train"))
o12["cluster_id"] = 12
m12 = rg.evaluate(o12, {"modelo": o12.pred_padrao_sem_anor.to_numpy(), "Corrigido ML": o12.pred_mapeado_sem_anor.to_numpy()},
                  rotulo={"desenho": "A", "treino": "3"})
m12 = m12[m12.trimestre == "Todos"].assign(abordagem="A cluster 3")
G = pd.concat([mt, m12], ignore_index=True)
ORD = ["A cluster 3", "B local", "C regional", "L LOCO"]


def matriz(recorte, versao, met):
    s = G[(G.recorte == recorte)]
    era = s[s.versao == "ERA5"].groupby("cluster")[met].first()
    v = s[s.versao == versao].pivot_table(index="abordagem", columns="cluster", values=met).reindex(ORD)
    return v, era


for rec in ("teste 2000–2024", "2025"):
    for versao in ("modelo", "Corrigido ML"):
        v, era = matriz(rec, versao, "RMSE")
        print(f"--- {rec}, RMSE, {'sem EQM' if versao == 'modelo' else 'com EQM (Corrigido ML)'}; primeira linha = ERA5")
        print(pd.concat([era.to_frame("ERA5").T, v]).round(3).to_string())
""")

py("""
fig, axs = plt.subplots(1, 3, figsize=(19, 4.6), constrained_layout=True)
for ax, (versao, met, tit, sinal) in zip(axs, [("modelo", "RMSE", "ganho de RMSE sobre o ERA5 — sem EQM", 1),
                                              ("Corrigido ML", "RMSE", "ganho de RMSE sobre o ERA5 — com EQM", 1),
                                              ("Corrigido ML", "POD_P99", "POD P99 menos o do ERA5 — com EQM", -1)]):
    v, era = matriz("teste 2000–2024", versao, met)
    m = (era - v) if sinal == 1 else (v - era)
    lim = float(np.nanmax(np.abs(m.values)))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m.values[i, j]):
                ax.text(j, i, f"{m.values[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(m.shape[1]), [f"cluster {x}" for x in m.columns]); ax.set_yticks(range(len(ORD)), ORD)
    ax.set_title(tit, fontsize=10); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Generalização por cluster (meses de teste 2000–2024; verde = melhor que o ERA5; cluster 12 só tem A)")
plt.show()
""")

md("## 3. Quadro-resumo")

py("""
linhas = []
for a in ABORD:
    linhas.append({"abordagem": a, "família": FAMILIA[a], "RMSE cluster 3": tab.loc[a, "RMSE"],
                   "ganho cluster 3": tab.loc[a, "ganho RMSE"], "POD P99 cluster 3": tab.loc[a, "POD P99"],
                   "FAR P99 cluster 3": tab.loc[a, "FAR P99"], "corr cluster 3": tab.loc[a, "corr"]})
res = pd.DataFrame(linhas).set_index("abordagem")
gen = {}
for ab in ORD:
    for versao, nome in (("modelo", "sem EQM"), ("Corrigido ML", "com EQM")):
        v, era = matriz("teste 2000–2024", versao, "RMSE")
        gen[f"{ab} {nome}"] = (era - v.loc[ab]).round(2).to_dict()
print("Cluster 3 (mesmos dias de teste):")
print(res.round(3).to_string())
print()
print("Ganho de RMSE sobre o ERA5 por cluster (meses de teste 2000–2024):")
print(pd.DataFrame(gen).T.to_string())
""")

md("""
## Leitura (números da execução de 24/10/2026; confira contra as tabelas acima se reexecutar)

**Cluster 3, mesmos 13.571 dias de teste:**
- **Erro diário:** todos os modelos sem EQM ganham do ERA5 (RMSE 2,66). Os melhores são **LSTM base (2,29)**,
  **ML grupo 1 (2,29)** e **ML selecionadas (2,32)**. Tirar o `anor` custa ~0,09 (ML sem anor 2,40).
- **Extremos (P99):** nenhum modelo sem EQM chega ao ERA5 (POD 0,21). ML selecionadas fica em 0,08, a LSTM em 0,00.
  O **EQM** sobe o POD (ML + EQM e Corrigido ML 0,11, local 0,13) e leva a distribuição para perto da do INMET
  (desvio prev/obs de 0,77–0,81 para 0,91), ao custo de +0,1 a +0,2 m/s de RMSE.
- **Regional (C) + EQM** é a única abordagem que alcança o ERA5 no P99 (POD 0,23 contra 0,21), mas com RMSE pior que
  o ERA5 (2,70) e mais falsos alarmes (FAR 0,82). O regional mistura clusters com mais vento, e o EQM dele, ajustado
  na distribuição de todos, infla o cluster 3 (viés +0,24).
- **Optuna 40 e 200** ficam atrás do EQM simples nos dois critérios. O de 200 sobreajustou a validação.
- **2025**, ano nunca visto, repete o quadro. ML + EQM, ML sem anor e local têm os melhores RMSE (2,29–2,32 contra 2,63).

**Generalização (seção 2, sem EQM):**
- O **local (B)** e o **regional (C)** ganham do ERA5 em todos os clusters de RS/SC: 0,18 a 0,69 m/s. O maior ganho é
  no cluster 4. O regional fica a poucos centésimos do local, e no cluster 2 é igual.
- O **modelo do cluster 3 (A)** e o **LOCO (L)**, ambos em região não vista, ganham pouco ou perdem: −0,14 a +0,25.
  O cluster 12 perde. A relação ERA5 → INMET é regional.
- **Com EQM**, o ganho de RMSE some na maior parte dos clusters, exceto no 4. O EQM é uma escolha para percentis e
  extremos, não para o erro diário.

**Recomendação:** para o erro do dia a dia, o modelo **sem EQM treinado com estações da região**, seja local ou
regional. Para mapas de percentis altos e risco, o mesmo modelo **com EQM**. Aplicar um modelo de uma região em
outra sem treino local não melhora o ERA5.
""")


nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
