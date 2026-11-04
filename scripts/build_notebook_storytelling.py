"""Gera `notebooks/feature_study_grupos/atual/storytelling_rajadas.ipynb`: a história completa do projeto, com plots.

Do problema (o ERA5 erra a rajada) à recomendação (modelo global anual), lendo só resultados já gravados:
  - `artifacts/feature_study/global/`  base comum, previsões e métricas de ML e LSTM (clusters 1–5 e 12);
  - `artifacts/feature_study/cluster3_groups_modal/summary/main/effects.csv`  contribuição dos grupos;
  - `artifacts/feature_study/inferencia_c12/results_cluster12.nc`  o efeito do `anor` em região nova.
Nada é treinado.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/storytelling_rajadas.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Corrigindo a rajada de vento do ERA5 com aprendizado de máquina

### Do problema ao modelo recomendado

Este notebook conta a história do projeto do começo ao fim, com os dados e os resultados. Cada capítulo responde
a uma pergunta e prepara a seguinte:

1. **O problema**: quanto o ERA5 erra a rajada que as estações do INMET medem?
2. **O desenho**: como montamos os dados para treinar e testar de forma honesta?
3. **Que informação ajuda**: quais variáveis do ERA5 carregam informação sobre o erro?
4. **A primeira correção**: o modelo melhora o dia a dia, mas corta os extremos. Como resolver?
5. **Generalização**: um modelo treinado numa região funciona em outra?
6. **ML × LSTM**: um modelo sequencial ajuda?
7. **Um modelo para tudo**: o melhor resultado.
8. **Recomendação e limites.**

Tudo é avaliado **só em dias que nenhum modelo viu**: os meses de teste (jan/abr/jul/out de 2000–2024) e o ano de
2025 inteiro.
""")

py("""
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
FS = 1.6                                  # fator de escala das fontes em todas as figuras
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10 * FS, "figure.dpi": 100})
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
A = RAIZ / "artifacts/feature_study"
G = A / "global"
TARGET = "daily_wind_gust_max"
CLS = [1, 2, 3, 4, 5, 12]
COR_CL = {1: "#2a7ab0", 2: "#d98c3f", 3: "#5aa469", 4: "#8a6fb0", 5: "#b8860b", 12: "#c44e52"}
C_ERA5, C_ML, C_LSTM, C_INMET = "#c44", "#2a7ab0", "#17808a", "#111111"

base = pd.read_parquet(G / "base_global.parquet", columns=["row_id", "estacao", "cluster_id", "time", "season",
                                                          "_split", "janela_ok", TARGET, "era5_gust_max"])
base = base[base.janela_ok].reset_index(drop=True)
pml = pd.read_parquet(G / "previsoes_ml.parquet")
pls = pd.read_parquet(G / "previsoes_lstm.parquet")
d = (base.merge(pml[["row_id", "A_bruto", "A_eqm", "G_bruto", "G_eqm", "GA_bruto", "GA_eqm"]].add_prefix("ml_")
                .rename(columns={"ml_row_id": "row_id"}), on="row_id")
         .merge(pls[["row_id", "G_bruto", "G_eqm", "GA_bruto", "GA_eqm"]].add_prefix("ls_").rename(columns={"ls_row_id": "row_id"}),
                on="row_id"))
# melhor regional por cluster e trimestre: o modelo local (B) do próprio cluster, um por trimestre (ML)
for pre, prev in (("ml", pml), ("ls", pls)):
    d[f"{pre}_local_eqm"] = np.nan
    d[f"{pre}_local_bruto"] = np.nan
    for cl in CLS:
        k = (d.cluster_id == cl).to_numpy()
        u = d[["row_id"]][k].merge(prev[["row_id", f"B{cl}_eqm", f"B{cl}_bruto"]], on="row_id", how="left")
        d.loc[k, f"{pre}_local_eqm"], d.loc[k, f"{pre}_local_bruto"] = u[f"B{cl}_eqm"].to_numpy(), u[f"B{cl}_bruto"].to_numpy()
# as 6 opções de modelo (coluna final, coluna da saída bruta, nome, cor), usadas em todas as figuras por estação-dia
MODS = [("ml_local_eqm", "ml_local_bruto", "ML local", "#9ecae1"),
        ("ml_G_eqm", "ml_G_bruto", "ML global por trimestre", "#4f93c8"),
        ("ml_GA_eqm", "ml_GA_bruto", "ML global anual", "#1f4e85"),
        ("ls_local_eqm", "ls_local_bruto", "LSTM local", "#99d8c9"),
        ("ls_G_eqm", "ls_G_bruto", "LSTM global por trimestre", "#3fae9a"),
        ("ls_GA_eqm", "ls_GA_bruto", "LSTM global anual", "#0b5f66")]
EST4 = ["DJF", "MAM", "JJA", "SON"]
SAZ = {"ml_local", "ml_G", "ls_local", "ls_G"}         # modelos com um ajuste por trimestre


def rot_est(ce, nome, e):
    # modelo sazonal: o ajuste daquele trimestre ("ML local DJF"); modelo anual: o mesmo modelo nos dias do trimestre
    return f"{nome} {e}" if ce.rsplit("_", 1)[0] in SAZ else f"{nome} {e}"
d["recorte"] = np.where(d._split == "novo", "2025", np.where(d._split == "test", "teste", "treino/val"))
nv = d[d._split.isin(["test", "novo"])].reset_index(drop=True)          # dias não vistos
mt = pd.read_csv(G / "metricas_global.csv")
est = pd.read_csv(G / "estacoes.csv")
shp = gpd.read_file(RAIZ / "dataset/shp/shp_vento.shp").to_crs(4326)


def rmse(p, y):
    return float(np.sqrt(np.nanmean((np.asarray(p) - np.asarray(y)) ** 2)))


print(f"{d.estacao.nunique()} estações, {len(d):,} estação-dias ({len(nv):,} não vistos), clusters {CLS}")
""")

# ── 1 ──────────────────────────────────────────────────────────────────────
md("""
## 1. O problema: o ERA5 erra a rajada

O ERA5 é a reanálise global do ECMWF: dá o vento em toda parte, hora a hora, desde 1940. Mas a grade tem ~25 km, e a
**rajada** é um fenômeno local e curto. As estações do INMET medem a rajada real, só que em poucos pontos.

O objetivo do projeto: **aprender com as estações a corrigir a rajada do ERA5**, para ter uma rajada corrigida em
toda a grade. Trabalhamos com **6 clusters** de regime de vento (o mapa do `shp_vento`): os quatro do RS/SC, o
cluster 5 (norte de SC, PR e litoral de SP e RJ) e o cluster 12 (leste de MG, ES e norte do RJ).
""")

py("""
fig, ax = plt.subplots(figsize=(8.5, 9.5))
shp.plot(ax=ax, color="#eeeeee", edgecolor="#bbbbbb", lw=0.4)
for cl, cor in COR_CL.items():
    shp[shp.cluster.astype(int) == cl].plot(ax=ax, color=cor, alpha=0.45, edgecolor=cor)
    p = shp[shp.cluster.astype(int) == cl].geometry.iloc[0].representative_point()
    ax.text(p.x, p.y, str(cl), fontsize=14 * FS, weight="bold", ha="center", va="center")
e_ = est[est.estacao.isin(d.estacao.unique())]
ax.scatter(e_.longitude, e_.latitude, s=16, c=[COR_CL[c_] for c_ in e_.cluster_id], edgecolor="k", lw=0.4,
           marker="^", zorder=3, label=f"estações INMET ({len(e_)})")
ax.set_aspect("equal"); ax.legend(loc="lower right")
ax.set_title("Clusters de vento estudados e as estações INMET usadas")
plt.show()
""")

md("""
**Quanto o ERA5 erra?** Cada ponto é um dia numa estação: a rajada máxima diária medida (INMET) contra a do ERA5 na
mesma célula. O viés médio é pequeno (de −0,6 a +0,6 m/s conforme o cluster), mas a dispersão é grande (RMSE de
2,3 a 3,4 m/s), e **os dias de vento forte ficam abaixo da diagonal**: o ERA5 suaviza os picos.

*Todos os modelos do notebook já incluem o mapeamento de quantis, explicado no capítulo 4.*
""")

py("""
paineis = [("era5_gust_max", "ERA5"), *[(ce, nome) for ce, _, nome, _ in MODS]]
fig, axs = plt.subplots(2, len(paineis), figsize=(6.4 * len(paineis), 12.5), constrained_layout=True)
qs = np.linspace(0.5, 0.999, 200)
for j_, (col, rot) in enumerate(paineis):
    ax = axs[0, j_]
    ok = nv[col].notna()
    hb = ax.hexbin(nv.loc[ok, TARGET], nv.loc[ok, col], gridsize=70, bins="log", cmap="viridis", mincnt=1,
                   extent=(0, 30, 0, 30), vmin=1, vmax=1000)
    ax.plot([0, 30], [0, 30], "w--", lw=1)
    ax.set_xlabel("rajada INMET (m/s)"); ax.set_ylabel(f"rajada {rot.split(' (')[0]} (m/s)")
    ax.set_title(f"{rot}\\nRMSE {rmse(nv.loc[ok, col], nv.loc[ok, TARGET]):.2f}, "
                 f"corr {np.corrcoef(nv.loc[ok, col], nv.loc[ok, TARGET])[0, 1]:.2f}", fontsize=10 * FS)
    ax2 = axs[1, j_]
    for cl in CLS:
        s = nv[(nv.cluster_id == cl) & ok]
        ax2.plot(np.quantile(s[TARGET], qs), np.quantile(s[col], qs), color=COR_CL[cl], lw=2, label=f"cluster {cl}")
        ax2.scatter(np.quantile(s[TARGET], 0.99), np.quantile(s[col], 0.99), color=COR_CL[cl], s=45, edgecolor="k",
                    zorder=3)
    ax2.plot([8, 33], [8, 33], "k:", lw=1); ax2.set_xlim(8, 34); ax2.set_ylim(8, 34)
    ax2.set_xlabel("quantil do INMET (m/s)"); ax2.set_ylabel(f"mesmo quantil, {rot.split(' (')[0]} (m/s)")
    ax2.set_title(f"{rot.split(' (')[0]}: quantis por cluster (ponto = P99)\\nabaixo da diagonal = corta os picos",
                  fontsize=10 * FS)
    ax2.legend(fontsize=8 * FS, loc="upper left")
fig.colorbar(hb, ax=axs[0, :], label="dias (escala log)", shrink=0.8)
fig.suptitle(f"Dias não vistos (n={len(nv):,}): ERA5 e as 6 opções de modelo")
plt.show()

print("RMSE por cluster:")
print(nv.groupby("cluster_id").apply(lambda s: pd.Series({rot: rmse(s[c_], s[TARGET]) for c_, rot in paineis}))
      .round(2).to_string())

tab = nv.groupby("cluster_id").apply(lambda s: pd.Series({
    "estações": s.estacao.nunique(), "dias": len(s), "RMSE ERA5": rmse(s.era5_gust_max, s[TARGET]),
    "viés ERA5": (s.era5_gust_max - s[TARGET]).mean(), "corr": np.corrcoef(s.era5_gust_max, s[TARGET])[0, 1],
    "P99 INMET": s[TARGET].quantile(0.99), "P99 ERA5": s.era5_gust_max.quantile(0.99)}))
print(tab.round(2).to_string())
""")

md("**Por trimestre.** Cada modelo sazonal no seu trimestre (ex.: *ML local DJF* é o ajuste de verão); os anuais avaliados no mesmo trimestre:")

py("""
fig, axs = plt.subplots(1, 4, figsize=(26, 6.5), constrained_layout=True)
for ax, e in zip(axs, EST4):
    s = nv[nv.season == e]
    qi_ = np.quantile(s[TARGET], qs)
    ax.plot(qi_, np.quantile(s.era5_gust_max, qs), color=C_ERA5, lw=2.4, label=f"ERA5 {e}")
    for ce, _, nome, cor in MODS:
        ok = s[ce].notna()
        ax.plot(np.quantile(s.loc[ok, TARGET], qs), np.quantile(s.loc[ok, ce], qs), color=cor, lw=1.8,
                ls="-" if ce.startswith("ls") else "--", label=rot_est(ce, nome, e))
    ax.plot([8, 33], [8, 33], "k:", lw=1); ax.set_xlim(8, 34); ax.set_ylim(8, 34)
    ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil (m/s)")
    ax.set_title(f"{e} (n={len(s):,})"); ax.legend(fontsize=7 * FS)
fig.suptitle("Quantis por trimestre, todos os clusters, dias não vistos", fontsize=12 * FS)
plt.show()
print("RMSE por trimestre e cluster (m/s):")
print(nv.groupby(["season", "cluster_id"]).apply(lambda s: pd.Series({rot: rmse(s[c_], s[TARGET]) for c_, rot in paineis}))
      .unstack("cluster_id").T.round(2).reindex(columns=EST4).to_string())
""")

md("""
**Regressão dia a dia.** Cada ponto é um estação-dia não visto (teste 2000–2024 + 2025): previsão × INMET. A
inclinação seria 1 se a previsão acompanhasse a variação diária observada; abaixo de 1, ela achata (superestima os dias
calmos e subestima os ventosos). Nos painéis de modelo, a reta tracejada vermelha é a do ERA5 nos mesmos dias.
Embaixo, o viés médio de cada estação (entre parênteses, o do ERA5).
""")

py("""
VERS_E = [("era5_gust_max", "ERA5"), *[(ce, nome) for ce, _, nome, _ in MODS]]
pe_ = nv.groupby(["estacao", "cluster_id"])[[TARGET, *[c_ for c_, _ in VERS_E]]].mean().reset_index()
pe_ = pe_.merge(est[["estacao", "latitude", "longitude"]].drop_duplicates("estacao"), on="estacao")
vies = {c_: pe_[c_] - pe_[TARGET] for c_, _ in VERS_E}
# escala comum, cortada no P95 do |viés|: uma estação com viés enorme achataria as outras
lim = float(np.nanpercentile(np.abs(np.concatenate([v.values for v in vies.values()])), 95))
fig, axs = plt.subplots(2, len(VERS_E), figsize=(6.4 * len(VERS_E), 12), constrained_layout=True)
xx = np.array([0, 35])


def reta_diaria(ax, s_, col, rot):
    # regressão sobre os estação-dias (não sobre médias de estação): hexbin de densidade + reta MQO;
    # nos painéis de modelo, a reta do ERA5 × INMET nos mesmos dias entra tracejada para comparação
    ok = s_[[TARGET, col, "era5_gust_max"]].notna().all(axis=1)
    y_ = s_.loc[ok, TARGET].to_numpy()
    ax.hexbin(y_, s_.loc[ok, col].to_numpy(), gridsize=60, bins="log", cmap="Greys", mincnt=1, extent=(0, 35, 0, 35))
    ax.plot(xx, xx, "k:", lw=1)
    for c_, nome_, cor_, ls_ in ([(col, "ERA5", C_ERA5, "-")] if col == "era5_gust_max" else
                                 [(col, "modelo", C_LSTM, "-"), ("era5_gust_max", "ERA5", C_ERA5, "--")]):
        p_ = s_.loc[ok, c_].to_numpy()
        a0, b0 = np.polyfit(y_, p_, 1)
        r0 = np.corrcoef(y_, p_)[0, 1]
        ax.plot(xx, a0 * xx + b0, color=cor_, lw=2, ls=ls_,
                label=f"{nome_}: y = {a0:.2f}x + {b0:.1f}  (r = {r0:.2f}, R² = {r0**2:.2f})")
        if c_ == col:
            a_ = a0
    ax.set_xlim(0, 35); ax.set_ylim(0, 35); ax.set_aspect("equal"); ax.legend(fontsize=8 * FS, loc="upper left")
    ax.set_xlabel("rajada diária, INMET (m/s)"); ax.set_ylabel(f"{rot.split(' (')[0]} (m/s)")
    ax.set_title(f"{rot}\\nn = {ok.sum():,} dias, inclinação {a_:.2f} (1 = acompanha)", fontsize=10 * FS)


for j_, (col, rot) in enumerate(VERS_E):
    reta_diaria(axs[0, j_], nv, col, rot)
    ax2 = axs[1, j_]
    shp[shp.cluster.astype(int).isin(CLS)].plot(ax=ax2, color="#f2f2f2", edgecolor="#999", lw=0.5)
    sc = ax2.scatter(pe_.longitude, pe_.latitude, c=vies[col].clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                     s=60, edgecolor="k", lw=0.4)
    for _, r_ in pe_[vies[col].abs() > lim].iterrows():
        rot_ = f"{vies[col][_]:+.1f}" + ("" if j_ == 0 else f" (ERA5 {vies['era5_gust_max'][_]:+.1f})")
        ax2.annotate(f"{r_.estacao} {rot_}", (r_.longitude, r_.latitude), fontsize=7 * FS, xytext=(4, 4),
                     textcoords="offset points")
    ax2.set_aspect("equal")
    ref_ = "" if j_ == 0 else f" (ERA5 {vies['era5_gust_max'].abs().median():.2f})"
    ax2.set_title(f"viés médio por estação, {rot.split(' (')[0]}: |viés| mediano {vies[col].abs().median():.2f} m/s"
                  + ref_, fontsize=10 * FS)
fig.colorbar(sc, ax=axs[1, :], shrink=0.7, label=f"viés médio − INMET (m/s; escala cortada em ±{lim:.1f})")
fig.suptitle("Dia a dia (em cima) e viés médio por estação (embaixo), todos os dias não vistos", fontsize=12 * FS)
plt.show()
""")

md("**Regressão dia a dia por trimestre** (linhas = trimestres; reta tracejada vermelha = ERA5 nos mesmos dias):")

py("""
fig, axs = plt.subplots(4, len(VERS_E), figsize=(4.6 * len(VERS_E), 19), constrained_layout=True)
for i_, e in enumerate(EST4):
    for j_, (col, rot) in enumerate(VERS_E):
        reta_diaria(axs[i_, j_], nv[nv.season == e], col,
                    f"ERA5 {e}" if col == "era5_gust_max" else rot_est(col, rot, e))
fig.suptitle("Regressão dia a dia por trimestre, todos os clusters, dias não vistos", fontsize=13 * FS)
plt.show()
""")

md("O mesmo, **cluster a cluster** (escala de cor do viés igual à da figura acima):")

py("""
for cl in CLS:
    pc = pe_[pe_.cluster_id == cl]
    fig, axs = plt.subplots(2, len(VERS_E), figsize=(6 * len(VERS_E), 10), constrained_layout=True)
    poly = shp[shp.cluster.astype(int) == cl]
    for j_, (col, rot) in enumerate(VERS_E):
        reta_diaria(axs[0, j_], nv[nv.cluster_id == cl], col, rot)
        ax2 = axs[1, j_]
        poly.plot(ax=ax2, color="#f2f2f2", edgecolor="#999", lw=0.6)
        v_ = (pc[col] - pc[TARGET])
        sc = ax2.scatter(pc.longitude, pc.latitude, c=v_.clip(-lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, s=80,
                         edgecolor="k", lw=0.4)
        v0 = pc.era5_gust_max - pc[TARGET]
        for _, r_ in pc.iterrows():
            rot_ = f"{r_[col] - r_[TARGET]:+.1f}" + ("" if j_ == 0 else f" (ERA5 {v0[_]:+.1f})")
            ax2.annotate(f"{r_.estacao} {rot_}", (r_.longitude, r_.latitude), fontsize=6 * FS,
                         xytext=(4, 3), textcoords="offset points")
        ref_ = "" if j_ == 0 else f"; ERA5 {v0.abs().median():.2f}"
        ax2.set_aspect("equal")
        ax2.set_title(f"viés por estação (|viés| mediano {v_.abs().median():.2f} m/s{ref_})", fontsize=10 * FS)
    fig.colorbar(sc, ax=axs[1, :], shrink=0.7, label=f"viés médio − INMET (m/s; escala ±{lim:.1f})")
    fig.suptitle(f"Cluster {cl}: dia a dia (em cima) e viés por estação ({len(pc)} estações, embaixo)", fontsize=13 * FS)
    plt.show()
""")

# ── 2 ──────────────────────────────────────────────────────────────────────
md("""
## 2. O desenho: treinar e testar de forma honesta

- **Uma linha por (estação, dia):** o alvo é a rajada máxima diária do INMET; as variáveis do ERA5 são tomadas na
  **hora em que a rajada do ERA5 é máxima** naquele dia (a hora do pico). A LSTM vê as 24 horas até esse pico.
- **Split por blocos de mês**, igual para todas as regiões: **teste** = janeiro, abril, julho e outubro de
  2000–2024 (um mês por trimestre climático); **validação** = blocos (ano, mês) sorteados nos outros meses;
  **treino** = o resto; **2025** fica fora de tudo. Um dia de folga entre os blocos evita que a informação vaze.
- **Nada é imputado:** dia com dado faltando sai.
""")

py("""
import sys
sys.path.insert(0, str(RAIZ))
from src.feature_study.data.station_dataset import calendar_split

sp = calendar_split()
dias = pd.date_range("2000-01-01", "2025-12-31", freq="MS")
lab = np.where(dias.year >= 2025, "2025", sp.label(dias))
cod = {"train": 0, "val": 1, "test": 2, "2025": 3}
mat = np.array([cod[x] for x in lab]).reshape(26, 12)
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(9, 7))
ax.imshow(mat, cmap=ListedColormap(["#c6dbef", "#fdae6b", "#2a7ab0", "#888888"]), aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(12), ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"])
ax.set_yticks(range(26), range(2000, 2026), fontsize=7 * FS); ax.grid(False)
for nome, cor in [("treino", "#c6dbef"), ("validação", "#fdae6b"), ("teste", "#2a7ab0"), ("2025 (nunca visto)", "#888888")]:
    ax.bar(0, 0, color=cor, label=nome)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
ax.set_title("Split por blocos de mês (o mesmo para todas as regiões)")
plt.show()
print(d.groupby(["cluster_id", "_split"]).size().unstack().rename(columns={"novo": "2025"}).to_string())
""")

md("""
### Como cada estratégia cria os seus modelos

Todas usam as mesmas linhas, o mesmo split e as mesmas 49 colunas; muda **com quais estações** se treina e **quantos
modelos** saem. Cada modelo ajusta o seu mapeamento de quantis na validação das estações com que foi treinado.

| estratégia | treina com | modelos | aplicado a |
|---|---|---|---|
| **estudo (cluster 3)** | cluster 3, estudo de features (só ML) | 4 (um por trimestre) | todos os clusters |
| **local** | as estações do próprio cluster | 6 clusters × 4 trimestres = 24 | o próprio cluster (e os outros, para teste) |
| **regional RS/SC** | clusters 1–4 juntos | 4 (um por trimestre) | todos |
| **global por trimestre** | clusters 1–5 + 12 juntos | 4 (um por trimestre) | todos |
| **global anual** | clusters 1–5 + 12 juntos, com o dia do ano como coluna | **1** | todos |
| **LOCO** (deixa um fora) | todos os clusters menos um | 6 × 4 = 24 | só o cluster deixado de fora (região nova) |

No ML, cada trimestre usa o campeão do estudo do cluster 3 (MLP no DJF, ExtraTrees no MAM, CatBoost no JJA, HistGB no
SON); o global anual usa CatBoost. Na LSTM a arquitetura é sempre a mesma (LSTM(96), 24 h até o pico).
""")

py("""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EST = [("local", {c_: [c_] for c_ in CLS}, "× 4 trimestres", "cada cluster treina o seu"),
       ("regional RS/SC", {"RS/SC": [1, 2, 3, 4]}, "× 4 trimestres", "o 5 e o 12 ficam fora do treino, mas recebem o modelo"),
       ("global por trimestre", {"global": CLS}, "× 4 trimestres", "um por trimestre para todos"),
       ("global anual", {"global": CLS}, "× 1 (ano todo)", "dia do ano como coluna"),
       ("LOCO", {f"sem {c_}": [x for x in CLS if x != c_] for c_ in CLS}, "× 4 trimestres",
        "testado só no cluster que ficou fora")]
fig, axs = plt.subplots(len(EST), 1, figsize=(15, 3.0 * len(EST)), constrained_layout=True)
xc = {c_: 0.5 + 1.0 * i for i, c_ in enumerate(CLS)}
for ax, (nome, mods, mult, nota) in zip(axs, EST):
    ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(0, 2.85, nome, fontsize=12 * FS, weight="bold", va="top")
    usados = {c_ for f_ in mods.values() for c_ in f_}
    for c_, x_ in xc.items():                     # estações de cada cluster (cinza = fora do treino)
        ax.add_patch(FancyBboxPatch((x_ - 0.42, 1.0), 0.84, 0.8, boxstyle="round,pad=0.03",
                                    fc=COR_CL[c_] if c_ in usados else "#cccccc", ec="k", alpha=0.85))
        ax.text(x_, 1.4, f"cluster\\n{c_}", ha="center", va="center", fontsize=8 * FS, color="w", weight="bold")
    n_ = len(mods)
    xs_ = np.linspace(7.0, 11.8, n_) if n_ > 1 else [9.4]
    for (rot_, fonte), xm in zip(mods.items(), xs_):                     # modelos e de onde vêm
        ax.add_patch(FancyBboxPatch((xm - 0.5, 1.0), 1.0, 0.8, boxstyle="round,pad=0.03", fc="#f2f2f2", ec="k"))
        ax.text(xm, 1.4, f"modelo\\n{rot_}", ha="center", va="center", fontsize=7.5 * FS)
        for c_ in fonte:
            ax.add_patch(FancyArrowPatch((xc[c_] + 0.3, 1.8), (xm - 0.3, 1.8), connectionstyle="arc3,rad=-0.18",
                                         arrowstyle="-|>", mutation_scale=8, lw=0.7,
                                         color=COR_CL[c_] if n_ > 1 else "#555", alpha=0.7))
    ax.text(13.9, 1.4, mult, ha="right", va="center", fontsize=10 * FS, weight="bold")
    ax.text(9.4, 0.55, nota, ha="center", fontsize=9 * FS, style="italic")
fig.suptitle("Estratégias de treino: estações (esquerda) → modelos (direita)", fontsize=13 * FS)
plt.show()
""")

# ── 3 ──────────────────────────────────────────────────────────────────────
md("""
## 3. Que informação do ERA5 ajuda?

As variáveis candidatas foram organizadas em 4 grupos (spec técnico v1.0): **1** vento local, camada limite e tempo;
**2** convecção; **3** escala sinótica; **4** relevo. O estudo de features (cluster 3, 5 seeds, 10 modelos,
bootstrap pareado em blocos) mediu quanto cada grupo acrescenta. A faixa cinza é o mínimo relevante (1 % do erro).
""")

py("""
ef = pd.read_csv(A / "cluster3_groups_modal/summary/main/effects.csv")
ef = ef[(ef.family == "all") & (ef.metric == "rmse")].set_index("comparison")
itens = [("drop_grp__grupo1", "tirar o grupo 1 (vento local)"), ("add_grp__grupo2", "somar o grupo 2 (convecção)"),
         ("add_grp__grupo3", "somar o grupo 3 (sinótico)"), ("add_grp__grupo4", "somar o grupo 4 (relevo)"),
         ("ctrl__noise", "controle: coluna de ruído")]
cor_v = {"acrescenta": C_ML, "sem informação": "#bbbbbb", "inconclusivo": "#e0a040", "prejudica": C_ERA5}
fig, ax = plt.subplots(figsize=(10, 3.8))
for yi, (k, rot) in enumerate(itens[::-1]):
    r = ef.loc[k]
    ax.errorbar(r.effect, yi, xerr=[[r.effect - r.ci_lo], [r.ci_hi - r.effect]], fmt="o", ms=8, capsize=4, lw=2,
                color=cor_v.get(r.verdict, "k"))
    ax.text(r.ci_hi + 0.006, yi, f"{r.effect:+.3f} m/s ({r.verdict})", va="center", fontsize=9 * FS)
s_ = float(ef.sesoi.iloc[0])
ax.axvspan(-s_, s_, color="#dddddd", zorder=0); ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(itens)), [r for _, r in itens[::-1]]); ax.set_xlabel("efeito no RMSE (m/s; positivo = o grupo ajuda)")
ax.set_xlim(right=0.35); ax.set_title("Só o grupo 1 (vento local) carrega informação; os outros ficam no ruído")
plt.show()
""")

md("""
Mesmo assim, uma **seleção por validação** (permutação por variável, acima do piso de ruído, em ≥ 3 de 4 trimestres)
encontrou **12 variáveis** que valem a pena, com 58 colunas. O desempenho é igual ao do conjunto completo de 167
colunas, com um terço das variáveis:

| grupo | variáveis selecionadas |
|---|---|
| 1 | rajada ERA5 (`gust10fg`, ponto, defasagens e vizinhança 75/250 km), vento a 10 m e 100 m, componente norte-sul, hora solar |
| 2 | CAPE e precipitação convectiva (ponto, defasagens, vizinhança, contrastes de escala) |
| 3 | gradiente de pressão ao nível do mar |
| 4 | `anor` (ângulo do relevo subgrade) |

**O `anor` era uma armadilha.** Ele é estático, e no cluster 3 tem só 9 valores, um por estação: o modelo o usava
como um código de estação. Numa região nova, os valores saem da faixa do treino e o modelo explode. Por isso **o
modelo usa as 49 colunas sem `anor`**. Lista completa: `documentation/12_features_selecionadas.md`.
""")

py("""
ds12 = xr.open_dataset(A / "inferencia_c12/results_cluster12.nc")
e12 = pd.read_csv(A / "inferencia_c12/estacoes.csv")
g12 = ds12[["era5_gust_max", "pred_padrao", "pred_padrao_sem_anor"]].to_dataframe().reset_index().dropna()
g12["lat"], g12["lon"] = g12.lat.round(3), g12.lon.round(3)
o12 = xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc")[TARGET].sel(estacao=e12.estacao.tolist())
o12 = o12.to_dataframe().reset_index()[["estacao", "time", TARGET]].dropna()
o12 = o12.merge(e12.assign(lat=e12.grid_lat.round(3), lon=e12.grid_lon.round(3))[["estacao", "lat", "lon"]], on="estacao")
o12 = o12.merge(g12, on=["lat", "lon", "time"])
o12 = o12[o12.time.dt.month.isin((1, 4, 7, 10)) | (o12.time.dt.year >= 2025)]
o12["season"] = o12.time.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
r = o12.groupby("season").apply(lambda s: pd.Series({"ERA5": rmse(s.era5_gust_max, s[TARGET]),
                                                     "modelo com anor": rmse(s.pred_padrao, s[TARGET]),
                                                     "modelo sem anor": rmse(s.pred_padrao_sem_anor, s[TARGET])}))
r = r.reindex(["DJF", "MAM", "JJA", "SON"])
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(4); w = 0.27
for k, (col, cor) in enumerate([("ERA5", C_ERA5), ("modelo com anor", "#999999"), ("modelo sem anor", C_ML)]):
    ax.bar(x + (k - 1) * w, r[col], w, color=cor, label=col)
    for xi, v in zip(x + (k - 1) * w, r[col]):
        ax.text(xi, min(v, 6.8) + 0.08, f"{v:.1f}", ha="center", fontsize=8 * FS)
ax.set_ylim(0, 7.3); ax.set_xticks(x, r.index); ax.set_ylabel("RMSE (m/s; eixo cortado em 7)")
ax.set_title("Cluster 12 (região nova): com `anor`, o MLP do verão (DJF) explode; sem ele, o modelo volta ao normal")
ax.legend(fontsize=8 * FS); plt.show()
""")

# ── 4 ──────────────────────────────────────────────────────────────────────
md("""
## 4. A primeira correção, e o problema dos extremos

O melhor modelo de ML por trimestre (escolhido entre 39 na validação), treinado com as estações do próprio cluster,
erra menos que o ERA5 no dia a dia. Mas ele **comprime a distribuição**: minimizando o erro quadrático, aposta no
meio quando não sabe se o dia vai ter pico, e os extremos ficam baixos.

**A correção: mapeamento de quantis (Empirical Quantile Mapping, EQM).** Cada previsão vai para o valor do INMET no
mesmo quantil, `f(p) = Q_INMET(F_modelo(p))`, ajustado na validação. A ordem dos dias não muda; só a escala é
esticada de volta. Detalhes: `documentation/11_mapeamento_de_quantis.md`.
""")

py("""
s = nv[nv.cluster_id == 3]
qi = np.quantile(s[TARGET], qs)
FAM = [(nome, cb, ce, cor) for ce, cb, nome, cor in MODS]
fig, axs = plt.subplots(len(FAM), 2, figsize=(15, 5.3 * len(FAM)), constrained_layout=True)
for i_, (nome, cb, ce, cor) in enumerate(FAM):
    ax = axs[i_, 0]
    for col, rot, c_, ls in [("era5_gust_max", "ERA5", C_ERA5, "-"), (cb, f"{nome}, saída bruta", cor, "-"),
                             (ce, f"{nome}, final", cor, "--")]:
        ax.plot(qi, np.quantile(s[col], qs), color=c_, ls=ls, lw=2.2, label=rot)
    ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k:", lw=1)
    ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil (m/s)")
    ax.set_title(f"Cluster 3, {nome}: a saída bruta comprime a cauda; a final a devolve", fontsize=10 * FS); ax.legend(fontsize=8 * FS)
    ax = axs[i_, 1]
    for col, rot, c_, ls in [(TARGET, "INMET", C_INMET, "-"), ("era5_gust_max", "ERA5", C_ERA5, "-"),
                             (cb, "saída bruta", cor, "-"), (ce, "final", cor, "--")]:
        ax.hist(s[col].dropna(), bins=np.arange(0, 30, 0.5), histtype="step", lw=1.8, color=c_, ls=ls, density=True,
                label=rot)
    ax.set_yscale("log"); ax.set_xlabel("rajada diária (m/s)"); ax.set_ylabel("densidade (log)")
    ax.set_title(f"{nome}: a cauda final volta para perto da do INMET", fontsize=10 * FS); ax.legend(fontsize=8 * FS)
plt.show()
for nome, cb, ce, _ in [("ERA5", "era5_gust_max", None, None), *FAM]:
    for col, rot in ([(cb, nome)] if ce is None else [(cb, f"{nome}, saída bruta"), (ce, f"{nome}, final")]):
        print(f"{rot:40s} RMSE {rmse(s[col], s[TARGET]):.2f} | desvio {s[col].std():.2f} (INMET {s[TARGET].std():.2f}) | "
              f"P99 {s[col].quantile(0.99):.1f} (INMET {s[TARGET].quantile(0.99):.1f})")
""")

py("""
fig, axs = plt.subplots(2, 4, figsize=(26, 12), constrained_layout=True)
s3 = nv[nv.cluster_id == 3]
for j_, e in enumerate(EST4):
    s = s3[s3.season == e]
    qi_ = np.quantile(s[TARGET], qs)
    for i_, (k_, tit_) in enumerate([(1, "saída bruta"), (0, "final")]):
        ax = axs[i_, j_]
        ax.plot(qi_, np.quantile(s.era5_gust_max, qs), color=C_ERA5, lw=2.4, label="ERA5")
        for m_ in MODS:
            col = m_[k_]
            ax.plot(qi_, np.quantile(s[col], qs), color=m_[3], lw=1.8, ls="-" if col.startswith("ls") else "--",
                    label=rot_est(m_[0], m_[2], e))
        ax.plot([qi_[0], qi_[-1]], [qi_[0], qi_[-1]], "k:", lw=1)
        ax.set_title(f"Cluster 3, {e}, {tit_}", fontsize=10 * FS); ax.legend(fontsize=7 * FS)
        ax.set_xlabel("quantil do INMET (m/s)"); ax.set_ylabel("mesmo quantil (m/s)")
fig.suptitle("Cluster 3 por trimestre: cada modelo sazonal no seu trimestre, saída bruta (em cima) e final (embaixo)",
             fontsize=12 * FS)
plt.show()
""")

md("""
A troca é inevitável: esticar a distribuição também estica os erros, então **o EQM custa um pouco de RMSE e devolve a
capacidade de pegar extremos**. Tentamos resolver isso dentro do modelo (busca de hiperparâmetros com Optuna, peso
extra nos dias de pico, perdas assimétricas): sobreajustou a validação e ficou pior que o EQM simples nos dois
critérios.
""")

md("""
### O EQM sozinho funciona, sem o modelo?

Se o EQM devolve os extremos, por que não aplicá-lo direto à rajada do ERA5? Testamos: um mapa de quantis
ERA5 → INMET por cluster, ajustado na mesma validação, sem modelo nenhum.
""")

py("""
from src.feature_study.diagnostics import quantile_map as qm

d["eqm_so"] = np.nan
for cl in CLS:
    k = (d.cluster_id == cl).to_numpy()
    kv = k & (d._split == "val").to_numpy()
    mapa = qm.fit(d.era5_gust_max.to_numpy()[kv], d[TARGET].to_numpy()[kv])
    d.loc[k, "eqm_so"] = qm.apply(d.era5_gust_max.to_numpy()[k], mapa)
nv = d[d._split.isin(["test", "novo"])].reset_index(drop=True)


def pod_far(s, col, q):
    lim_ = s.groupby("estacao")[TARGET].transform(lambda x: x.quantile(q))
    ev, al = s[TARGET] >= lim_, s[col] >= lim_
    return (ev & al).sum() / max(ev.sum(), 1), (~ev & al).sum() / max(al.sum(), 1)


VERS = [("era5_gust_max", "ERA5", C_ERA5), ("eqm_so", "ERA5 reescalado (sem modelo)", "#e0a040"),
        *[(ce, nome, cor) for ce, _, nome, cor in MODS]]
lin = []
for cl in CLS:
    s = nv[nv.cluster_id == cl]
    for col, rot, _ in VERS:
        po, fa = pod_far(s, col, 0.99)
        lin.append({"cluster": cl, "versao": rot, "RMSE": rmse(s[col], s[TARGET]),
                    "correlação": np.corrcoef(s[col], s[TARGET])[0, 1], "POD P99": po, "FAR P99": fa})
q = pd.DataFrame(lin)
fig, axs = plt.subplots(2, 2, figsize=(19, 9), constrained_layout=True)
for ax, met, tit in zip(axs.flat, ["RMSE", "correlação", "POD P99", "FAR P99"],
                        ["RMSE (menor = melhor)", "correlação com o INMET", "POD no P99 (maior = melhor)",
                         "FAR no P99 (menor = melhor)"]):
    x = np.arange(len(CLS)); w = 0.8 / len(VERS)
    for k, (_, rot, cor) in enumerate(VERS):
        v = q[q.versao == rot].set_index("cluster")[met].reindex(CLS)
        ax.bar(x + (k - (len(VERS) - 1) / 2) * w, v, w, color=cor, label=rot)
    ax.set_xticks(x, [f"cl. {c_}" for c_ in CLS]); ax.set_title(tit)
    if met == "RMSE":
        ax.set_ylim(1.5, None)
    if met == "correlação":
        ax.set_ylim(0.5, 0.8)
axs[0, 0].legend(fontsize=7 * FS, ncol=2)
fig.suptitle("Só reescalar o ERA5 não basta: mais extremos marcados, mais alarmes falsos, mesmo acerto diário",
             fontsize=12 * FS)
plt.show()
print(q.pivot_table(index="versao", columns="cluster", values=["RMSE", "correlação"]).round(2).to_string())
""")

py("""
fig, axs = plt.subplots(4, 2, figsize=(20, 18), constrained_layout=True)
for i_, e in enumerate(EST4):
    lin_ = []
    for cl in CLS:
        s = nv[(nv.cluster_id == cl) & (nv.season == e)]
        for col, rot, cor in VERS:
            lin_.append({"cluster": cl, "col": col, "RMSE": rmse(s[col], s[TARGET]), "POD P99": pod_far(s, col, 0.99)[0]})
    q_ = pd.DataFrame(lin_)
    for ax, met in zip(axs[i_], ["RMSE", "POD P99"]):
        x = np.arange(len(CLS)); w = 0.8 / len(VERS)
        for k, (col, rot, cor) in enumerate(VERS):
            lab = rot if col in ("era5_gust_max", "eqm_so") else rot_est(col, rot, e)
            ax.bar(x + (k - (len(VERS) - 1) / 2) * w, q_[q_.col == col].set_index("cluster")[met].reindex(CLS), w,
                   color=cor, label=lab)
        ax.set_xticks(x, [f"cl. {c_}" for c_ in CLS]); ax.set_title(f"{e}: {met}")
        if met == "RMSE":
            ax.set_ylim(1.0, None)
        ax.legend(fontsize=6.5 * FS, ncol=2)
fig.suptitle("ERA5 reescalado × modelos, por trimestre (dias não vistos)", fontsize=12 * FS)
plt.show()
""")

md("""
**Não.** O EQM sozinho **piora o RMSE do ERA5** em todos os clusters e **não muda a correlação**: ele só reescala os
valores, e o ERA5 continua errando os mesmos dias. Ele marca mais dias como extremos (o POD sobe), mas a maioria
desses alarmes é falsa (o FAR sobe junto).

O ganho tem duas partes: **o modelo** aprende quais dias são fortes e quais são fracos (a correlação sobe de
0,55–0,71 para 0,65–0,76); **o EQM** devolve a escala que o modelo comprimiu. Só os dois juntos ganham do ERA5 no erro
diário e nos extremos. Por isso, **daqui em diante, todo modelo já inclui o EQM**.
""")

# ── 5 ──────────────────────────────────────────────────────────────────────
md("""
## 5. Generalização: um modelo de uma região funciona em outra?

Testamos o modelo do cluster 3 em regiões que ele nunca viu, e treinamos modelos com as estações de cada cluster
(local), de RS/SC juntos (regional), de todos (global) e de todos menos o avaliado (LOCO, a região nunca vista).
""")

py("""
B = mt[(mt.familia == "ML") & mt.chave.str.match(r"^B\\d") & (mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")]
v = B[B.versao == "modelo"].pivot_table(index="treino", columns="cluster", values="RMSE")
e = B[B.versao == "ERA5"].pivot_table(index="treino", columns="cluster", values="RMSE")
m = (e - v).reindex(index=[str(x) for x in CLS], columns=CLS)
fig, axs = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.3]})
lim = float(np.nanmax(np.abs(m.values)))
im = axs[0].imshow(m.values, cmap="RdYlGn", vmin=-lim, vmax=lim)
for a in range(len(CLS)):
    for b in range(len(CLS)):
        axs[0].text(b, a, f"{m.values[a, b]:+.2f}", ha="center", va="center", fontsize=9 * FS,
                    weight="bold" if a == b else None)
axs[0].set_xticks(range(len(CLS)), [f"avalia {x}" for x in CLS]); axs[0].set_yticks(range(len(CLS)), [f"treina {x}" for x in CLS])
axs[0].grid(False); fig.colorbar(im, ax=axs[0], shrink=0.8, label="ganho de RMSE sobre o ERA5 (m/s)")
axs[0].set_title("Modelo local: ganha no próprio cluster (diagonal), perde nos outros")
t = mt[(mt.familia == "ML") & (mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024") & (mt.versao == "modelo")]
t = t[~(t.chave.str.match(r"^B\\d") & (t.chave.str[1:] != t.cluster.astype(str)))]
era = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024") & (mt.versao == "ERA5")].groupby("cluster").RMSE.first()
grupos = [("A", "modelo do cluster 3, sem treino local", "#bbbbbb"), ("L", "LOCO: região nunca vista", "#e0a040"),
          ("B", "local: estações do próprio cluster", C_ML)]
x = np.arange(len(CLS)); w = 0.2
axs[1].bar(x - 1.5 * w, era.reindex(CLS), w, color=C_ERA5, label="ERA5")
for k, (ch, rot, cor) in enumerate(grupos):
    vv = t[t.chave.str[0] == ch].groupby("cluster").RMSE.first().reindex(CLS) if ch != "A" else t[t.chave == "A"].set_index("cluster").RMSE.reindex(CLS)
    axs[1].bar(x + (k - 0.5) * w, vv, w, color=cor, label=rot)
axs[1].set_xticks(x, [f"cluster {c_}" for c_ in CLS]); axs[1].set_ylim(1.5, None); axs[1].set_ylabel("RMSE (m/s)")
axs[1].set_title("Sem estações da região, o modelo não melhora o ERA5"); axs[1].legend(fontsize=8 * FS)
plt.show()
""")

md("""
**A relação entre o ERA5 e o INMET é regional.** Um modelo aprende os vícios do ERA5 *naquela* região; levado para
outra, fica igual ou pior que o ERA5. Mas **basta incluir estações da região no treino** para o ganho aparecer em
todos os clusters, e um modelo regional ou global (várias regiões juntas) fica a poucos centésimos do local.
""")

# ── 6 ──────────────────────────────────────────────────────────────────────
md("""
## 6. ML × LSTM, de igual para igual

A LSTM vê as 24 horas até o pico, em vez de um instante só. Rodamos ML e LSTM com **as mesmas linhas, o mesmo split,
as mesmas 49 colunas**, em todos os desenhos (a LSTM em GPU, no Modal). Diferença de RMSE LSTM − ML
(azul = a LSTM erra menos):
""")

py("""
DES = {"B": "local", "C": "regional RS/SC", "G": "global por trimestre", "GA": "global anual", "L": "LOCO (região nova)"}
t = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024") & (mt.versao == "Corrigido ML")].copy()
t = t[~(t.chave.str.match(r"^B\\d") & (t.chave.str[1:] != t.cluster.astype(str)))]
t["d"] = t.chave.where(~t.chave.str.match(r"^[BL]\\d"), t.chave.str[0]).map(DES)
p = t.pivot_table(index=["d", "cluster"], columns="familia", values="RMSE").dropna()
dif = (p.LSTM - p.ML).unstack("cluster").reindex(index=list(DES.values()), columns=CLS).dropna(how="all")
fig, ax = plt.subplots(figsize=(10, 4))
lim = float(np.nanmax(np.abs(dif.values)))
im = ax.imshow(dif.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for a in range(dif.shape[0]):
    for b in range(dif.shape[1]):
        if np.isfinite(dif.values[a, b]):
            ax.text(b, a, f"{dif.values[a, b]:+.2f}", ha="center", va="center", fontsize=9 * FS)
ax.set_xticks(range(len(CLS)), [f"cluster {x}" for x in CLS]); ax.set_yticks(range(len(dif)), dif.index); ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="RMSE LSTM − ML (m/s)")
ax.set_title("Com estações da região, empatam; no modelo anual a LSTM ganha; em região nova o ML generaliza melhor")
plt.show()
""")

# ── 7 ──────────────────────────────────────────────────────────────────────
md("""
## 7. Um modelo para tudo: o melhor resultado

Até aqui, cada trimestre (DJF, MAM, JJA, SON) tinha o seu modelo. O último teste foi **um único modelo para os 6
clusters e o ano inteiro**, com o dia do ano como variável (seno e cosseno). Cada trimestre passa a
aprender com os dados dos outros: são 283 mil linhas de treino em vez de ~71 mil por modelo.

Essa é a **LSTM global anual**.
""")

py("""
t = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")]
era = t[t.versao == "ERA5"].groupby("cluster").RMSE.first().reindex(CLS)
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
x = np.arange(len(CLS)); w = 0.16
ax.bar(x - 2 * w, era, w, color=C_ERA5, label="ERA5")
for k, (fam, ch, cor, rot) in enumerate([("ML", "G", "#9ecae1", "ML por trimestre"), ("ML", "GA", C_ML, "ML anual"),
                                         ("LSTM", "G", "#99d8c9", "LSTM por trimestre"), ("LSTM", "GA", C_LSTM, "LSTM anual")]):
    v = t[(t.familia == fam) & (t.chave == ch) & (t.versao == "Corrigido ML")].set_index("cluster").RMSE.reindex(CLS)
    ax.bar(x + (k - 1) * w, v, w, color=cor, label=rot)
    if ch == "GA" and fam == "LSTM":
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=7.5 * FS, weight="bold")
ax.set_xticks(x, [f"cluster {c_}" for c_ in CLS]); ax.set_ylim(1.5, 3.6); ax.set_ylabel("RMSE (m/s)"); ax.legend(fontsize=8 * FS)
fig.suptitle("Global: um modelo por trimestre × um modelo para o ano todo (meses de teste 2000–2024)", fontsize=12 * FS)
plt.show()
""")

py("""
ts_ = mt[(mt.trimestre != "Todos") & (mt.recorte == "teste 2000–2024")]
fig, axs = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
for ax, e in zip(axs.flat, EST4):
    t_ = ts_[ts_.trimestre == e]
    x = np.arange(len(CLS)); w = 0.16
    ax.bar(x - 2 * w, t_[t_.versao == "ERA5"].groupby("cluster").RMSE.first().reindex(CLS), w, color=C_ERA5, label="ERA5")
    for k, (fam, ch, cor, rot) in enumerate([("ML", "G", "#9ecae1", f"ML global {e}"),
                                             ("ML", "GA", C_ML, f"ML anual {e}"),
                                             ("LSTM", "G", "#99d8c9", f"LSTM global {e}"),
                                             ("LSTM", "GA", C_LSTM, f"LSTM anual {e}")]):
        v = t_[(t_.familia == fam) & (t_.chave == ch) & (t_.versao == "Corrigido ML")].set_index("cluster").RMSE.reindex(CLS)
        ax.bar(x + (k - 1) * w, v, w, color=cor, label=rot)
    ax.set_xticks(x, [f"cluster {c_}" for c_ in CLS]); ax.set_ylim(1.0, None); ax.set_ylabel("RMSE (m/s)")
    ax.set_title(e); ax.legend(fontsize=7 * FS)
fig.suptitle("Por trimestre: o modelo global daquele trimestre × o anual nos mesmos dias (teste 2000–2024)", fontsize=12 * FS)
plt.show()
""")

md("""
**Todas as opções lado a lado.** ML e LSTM em três escalas (local = um modelo por cluster e trimestre; global por
trimestre; global anual), contra o ERA5. Erro do dia a dia (RMSE) e detecção de extremos (POD no P99),
nos meses de teste de 2000–2024 e em 2025:
""")

py("""
tt = mt[mt.trimestre == "Todos"]
num = pd.to_numeric(tt.chave.str.extract(r"^B(\\d+)$")[0], errors="coerce")
NOME7 = {"B": "local", "G": "global por trimestre", "GA": "global anual"}
op = pd.concat([tt[tt.chave.isin(["G", "GA"])], tt[num == tt.cluster].assign(chave="B")])
op = op[op.versao == "Corrigido ML"].assign(opcao=lambda x: x.familia + " " + x.chave.map(NOME7))
op = pd.concat([tt[tt.versao == "ERA5"].drop_duplicates(["cluster", "recorte"]).assign(opcao="ERA5", familia="ERA5"), op])
ORD7 = ["ERA5", *[f"{f} {n}" for f in ("ML", "LSTM") for n in NOME7.values()]]
for rec in ("teste 2000–2024", "2025"):
    print(f"\\n{rec}: RMSE (m/s) | POD no P99")
    r_ = op[op.recorte == rec].pivot_table(index="opcao", columns="cluster", values="RMSE").reindex(ORD7)
    q_ = op[op.recorte == rec].pivot_table(index="opcao", columns="cluster", values="POD_P99").reindex(ORD7)
    print(r_.round(2).astype(str).add(" | ").add(q_.round(2).astype(str)).to_string())

COR7 = {"ERA5": C_ERA5, "ML": C_ML, "LSTM": C_LSTM}
fig, axs = plt.subplots(1, 2, figsize=(17, 5.4), constrained_layout=True)
for ax, met, tit in ((axs[0], "RMSE", "RMSE (m/s) — menor é melhor"), (axs[1], "POD_P99", "POD no P99 — maior é melhor")):
    d = op[op.recorte == "teste 2000–2024"].pivot_table(index="opcao", columns="cluster", values=met).reindex(columns=CLS)
    ops = ORD7
    w = 0.8 / len(ops)
    for i, o in enumerate(ops):
        f_ = o.split()[0]
        ax.bar(np.arange(len(CLS)) + (i - (len(ops) - 1) / 2) * w, d.loc[o], w, color=COR7[f_], edgecolor="black",
               lw=0.4, alpha=0.4 if "local" in o else 0.7 if "trimestre" in o else 1.0,
               hatch="//" if f_ == "LSTM" and "anual" in o else None, label=o)
    ax.set_xticks(range(len(CLS)), [f"cluster {k}" for k in CLS]); ax.set_title(tit); ax.legend(fontsize=8 * FS, ncol=2)
axs[0].set_ylim(1.5, 3.6)
fig.suptitle("Todas as opções, meses de teste 2000–2024 (hachurado = LSTM global anual)", fontsize=12 * FS)
plt.show()
""")

py("""
tq = mt[mt.trimestre.isin(EST4) & (mt.recorte == "teste 2000–2024")]
numq = pd.to_numeric(tq.chave.str.extract(r"^B(\\d+)$")[0], errors="coerce")
opq = pd.concat([tq[tq.chave.isin(["G", "GA"])], tq[numq == tq.cluster].assign(chave="B")])
opq = opq[opq.versao == "Corrigido ML"].assign(opcao=lambda x: x.familia + " " + x.chave.map(NOME7))
opq = pd.concat([tq[tq.versao == "ERA5"].drop_duplicates(["cluster", "trimestre"]).assign(opcao="ERA5", familia="ERA5"), opq])
fig, axs = plt.subplots(4, 2, figsize=(19, 19), constrained_layout=True)
for i_, e in enumerate(EST4):
    for ax, met in zip(axs[i_], ("RMSE", "POD_P99")):
        d_ = opq[opq.trimestre == e].pivot_table(index="opcao", columns="cluster", values=met).reindex(index=ORD7, columns=CLS)
        w = 0.8 / len(ORD7)
        for i, o in enumerate(ORD7):
            f_ = o.split()[0]
            lab = (f"{o} {e}" if ("local" in o or "trimestre" in o) else f"{o} {e}")
            ax.bar(np.arange(len(CLS)) + (i - (len(ORD7) - 1) / 2) * w, d_.loc[o], w, color=COR7[f_], edgecolor="black",
                   lw=0.4, alpha=0.4 if "local" in o else 0.7 if "trimestre" in o else 1.0,
                   hatch="//" if f_ == "LSTM" and "anual" in o else None, label=lab)
        ax.set_xticks(range(len(CLS)), [f"cluster {k}" for k in CLS]); ax.set_title(f"{e}: {met}")
        if met == "RMSE":
            ax.set_ylim(1.0, None)
        ax.legend(fontsize=6.5 * FS, ncol=2)
fig.suptitle("Todas as opções por trimestre, meses de teste 2000–2024", fontsize=13 * FS)
plt.show()
for met in ("RMSE", "POD_P99"):
    print(f"\\n{met}, teste 2000–2024:")
    print(opq.pivot_table(index=["trimestre", "opcao"], columns="cluster", values=met).reindex(columns=CLS).round(2).to_string())
""")

md("""
- **Erro do dia a dia:** a LSTM global anual tem o menor RMSE em 5 dos 6 clusters, no teste e em 2025. No cluster 3
  o ML treinado só no próprio cluster fica um pouco à frente, e em 2025 a LSTM global anual chega a errar mais que o ERA5 ali.
- **Extremos:** a LSTM global anual detecta o P99 tanto quanto ou mais que o ERA5 (0,13–0,27), **exceto no cluster 5**
  (0,09 contra 0,14 do ERA5). **O ML global anual (CatBoost) detecta um pouco mais** nos clusters do Sul (até 0,33) e dispensa GPU e janelas de 24 h: é a alternativa quando a
  prioridade é a cauda.
""")

md("""
**Onde a correção ajuda?** Ganho de RMSE de cada modelo sobre o ERA5 em cada estação, nos dias não vistos (verde =
erra menos que o ERA5).
""")

py("""
g_ = nv.groupby(["estacao", "cluster_id"]).apply(lambda s: pd.Series({
    "n": len(s), **{nome: rmse(s.era5_gust_max, s[TARGET]) - rmse(s[ce], s[TARGET]) for ce, _, nome, _ in MODS}})
    ).reset_index()
g_ = g_[g_.n >= 200].merge(est[["estacao", "latitude", "longitude"]].drop_duplicates("estacao"), on="estacao")
fig, axs = plt.subplots(2, 3, figsize=(21, 15), constrained_layout=True)
# escala comum, limitada ao P95 do |ganho|: uma estação com ganho muito alto (ERA5 muito ruim na célula) achataria as outras
lim = float(np.nanpercentile(np.abs(g_[[m_[2] for m_ in MODS]].to_numpy()), 95))
for ax, (_, _, nome, _) in zip(axs.flat, MODS):
    shp[shp.cluster.astype(int).isin(CLS)].plot(ax=ax, color="#f2f2f2", edgecolor="#999", lw=0.5)
    sc = ax.scatter(g_.longitude, g_.latitude, c=g_[nome].clip(-lim, lim), cmap="RdYlGn", vmin=-lim, vmax=lim, s=55,
                    edgecolor="k", lw=0.4)
    for _, r_ in g_[g_[nome].abs() > lim].iterrows():
        ax.annotate(f"{r_.estacao} {r_[nome]:+.1f}", (r_.longitude, r_.latitude), fontsize=6 * FS, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(f"{nome}: melhor que o ERA5 em {int((g_[nome] > 0).sum())} de {len(g_)} estações\\n"
                 f"(ganho mediano {g_[nome].median():.2f} m/s)", fontsize=10 * FS)
fig.colorbar(sc, ax=axs, shrink=0.6, label=f"ganho de RMSE sobre o ERA5 (m/s; escala cortada em ±{lim:.1f})")
plt.show()
""")

py("""
ge_ = nv.groupby(["estacao", "season"]).apply(lambda s: pd.Series({
    "n": len(s), **{ce: rmse(s.era5_gust_max, s[TARGET]) - rmse(s[ce], s[TARGET]) for ce, _, _, _ in MODS}})
    ).reset_index()
ge_ = ge_[ge_.n >= 50].merge(est[["estacao", "latitude", "longitude"]].drop_duplicates("estacao"), on="estacao")
fig, axs = plt.subplots(4, len(MODS), figsize=(4.2 * len(MODS), 17.5), constrained_layout=True)
for i_, e in enumerate(EST4):
    g2 = ge_[ge_.season == e]
    for j_, (ce, _, nome, _) in enumerate(MODS):
        ax = axs[i_, j_]
        shp[shp.cluster.astype(int).isin(CLS)].plot(ax=ax, color="#f2f2f2", edgecolor="#999", lw=0.4)
        sc = ax.scatter(g2.longitude, g2.latitude, c=g2[ce].clip(-lim, lim), cmap="RdYlGn", vmin=-lim, vmax=lim, s=22,
                        edgecolor="k", lw=0.3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{rot_est(ce, nome, e)}\\n{int((g2[ce] > 0).sum())}/{len(g2)} melhores, mediana {g2[ce].median():.2f}",
                     fontsize=8 * FS)
fig.colorbar(sc, ax=axs, shrink=0.5, label=f"ganho de RMSE sobre o ERA5 (m/s; escala ±{lim:.1f})")
fig.suptitle("Ganho sobre o ERA5 por estação e trimestre", fontsize=13 * FS)
plt.show()
""")

md("""
**E os extremos?** POD = fração dos dias extremos do INMET (acima do P95/P99 de cada estação) que a previsão também
põe acima do limiar; FAR = fração dos alarmes que eram falsos.
""")

py("""
VPF = [("era5_gust_max", "ERA5", C_ERA5), *[(ce, nome, cor) for ce, _, nome, cor in MODS]]
fig, axs = plt.subplots(2, 1, figsize=(16, 9.5), constrained_layout=True)
for ax, q in zip(axs, (0.95, 0.99)):
    x = np.arange(len(CLS)); w = 0.8 / len(VPF)
    for k, (col, rot, cor) in enumerate(VPF):
        pf = [pod_far(nv[nv.cluster_id == cl], col, q) for cl in CLS]
        xk = x + (k - (len(VPF) - 1) / 2) * w
        ax.bar(xk, [p_[0] for p_ in pf], w, color=cor, label=f"{rot} (POD)")
        ax.scatter(xk, [p_[1] for p_ in pf], color="k", marker="_", s=80, zorder=3)
    ax.set_xticks(x, [f"cluster {c_}" for c_ in CLS]); ax.set_ylim(0, 1)
    ax.set_title(f"P{int(q * 100)}: barras = POD (maior = melhor); traços = FAR (menor = melhor)"); ax.legend(fontsize=7 * FS, ncol=4)
plt.show()
""")

py("""
fig, axs = plt.subplots(4, 1, figsize=(16, 17), constrained_layout=True)
for ax, e in zip(axs, EST4):
    x = np.arange(len(CLS)); w = 0.8 / len(VPF)
    for k, (col, rot, cor) in enumerate(VPF):
        pf = [pod_far(nv[(nv.cluster_id == cl) & (nv.season == e)], col, 0.99) for cl in CLS]
        xk = x + (k - (len(VPF) - 1) / 2) * w
        ax.bar(xk, [p_[0] for p_ in pf], w, color=cor,
               label=f"ERA5 {e}" if col == "era5_gust_max" else rot_est(col, rot, e))
        ax.scatter(xk, [p_[1] for p_ in pf], color="k", marker="_", s=80, zorder=3)
    ax.set_xticks(x, [f"cluster {c_}" for c_ in CLS]); ax.set_ylim(0, 1)
    ax.set_title(f"P99 em {e} (limiar anual da estação): barras = POD; traços = FAR"); ax.legend(fontsize=7 * FS, ncol=4)
plt.show()
""")

md("""
**Diagrama de Taylor.** Resume três medidas num ponto: a **correlação** com o INMET (ângulo), o **desvio-padrão**
relativo ao do INMET (distância à origem; 1 = mesma variabilidade) e o **erro centrado** (distância à estrela, que é o
INMET). Quanto mais perto da estrela, melhor.
""")

py("""
TAY = [("era5_gust_max", "ERA5", C_ERA5), *[(ce, nome, cor) for ce, _, nome, cor in MODS]]


def taylor(ax, s, titulo):
    y = s[TARGET].to_numpy(float)
    sd0 = y.std()
    rs = np.linspace(0, 1.6, 200)
    for rc in (0.25, 0.5, 0.75, 1.0):                                  # contornos de erro centrado (normalizado)
        th = np.linspace(0, np.pi / 2, 200)
        x0, y0 = 1 + rc * np.cos(np.linspace(0, np.pi, 200)), rc * np.sin(np.linspace(0, np.pi, 200))
        r_, t_ = np.hypot(x0, y0), np.arctan2(y0, x0)
        ok = (r_ <= 1.6) & (t_ <= np.pi / 2)
        ax.plot(t_[ok], r_[ok], color="#bbbbbb", lw=0.7, ls="--")
    ax.plot(np.linspace(0, np.pi / 2, 100), np.ones(100), color="k", lw=0.8, ls=":")
    ax.plot(0, 1, marker="*", ms=16, color=C_INMET, zorder=5)
    for col, rot, cor in TAY:
        ok = s[col].notna()
        p_ = s.loc[ok, col].to_numpy(float)
        r = np.corrcoef(p_, y[ok.to_numpy()])[0, 1]
        ax.plot(np.arccos(r), p_.std() / sd0, marker="o", ms=9, color=cor, mec="k", mew=0.6, ls="", zorder=4)
    ax.set_thetamin(0); ax.set_thetamax(90); ax.set_rlim(0, 1.6)
    cor_ticks = np.array([0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    ax.set_thetagrids(np.degrees(np.arccos(cor_ticks)), [f"{c_:.2f}" for c_ in cor_ticks], fontsize=7 * FS)
    ax.set_title(titulo, fontsize=10 * FS, pad=12)
    ax.text(np.radians(50), 1.85, "correlação", fontsize=8 * FS, ha="center")


fig, axs = plt.subplots(2, 4, figsize=(24, 12), subplot_kw={"projection": "polar"}, constrained_layout=True)
axs.flat[-1].set_visible(False)
for ax, (rot, s_) in zip(axs.flat, [("todos os clusters", nv), *[(f"cluster {cl}", nv[nv.cluster_id == cl]) for cl in CLS]]):
    taylor(ax, s_, f"{rot} (n={len(s_):,})")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="*", color=C_INMET, ls="", ms=14, label="INMET (referência)")]
h += [Line2D([], [], marker="o", color=c_, mec="k", ls="", ms=9, label=r_) for _, r_, c_ in TAY]
fig.legend(handles=h, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.03), frameon=False)
fig.suptitle("Diagrama de Taylor, dias não vistos: raio = desvio relativo ao INMET; ângulo = correlação; "
             "tracejado = erro centrado", fontsize=12 * FS)
plt.show()
""")

py("""
fig, axs = plt.subplots(1, 4, figsize=(26, 7), subplot_kw={"projection": "polar"}, constrained_layout=True)
for ax, e in zip(axs, EST4):
    s_ = nv[nv.season == e]
    taylor(ax, s_, f"{e}, todos os clusters (n={len(s_):,})")
h = [Line2D([], [], marker="*", color=C_INMET, ls="", ms=14, label="INMET (referência)")]
h += [Line2D([], [], marker="o", color=c_, mec="k", ls="", ms=9,
             label=r_) for col, r_, c_ in TAY]
fig.legend(handles=h, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.suptitle("Taylor por trimestre: os modelos sazonais são o ajuste daquele trimestre", fontsize=12 * FS)
plt.show()
""")

md("""
**Leitura:** o ERA5 tem a menor correlação. Os modelos giram para a direita (correlação maior) com o raio perto de 1
(a mesma variabilidade do INMET). Compare a distância de cada modelo à estrela em cada cluster.
""")

md("""
### Mapa de extrapolação: a correção em toda a grade

O objetivo do projeto é ter a rajada corrigida **onde não há estação**. Abaixo, **cada opção de modelo**
aplicada a todas as células dos clusters, 2000–2025: ML local, ML global por trimestre, ML global anual, LSTM local e
LSTM global anual, ao lado do ERA5 (linha de cima) e da correção em relação ao ERA5 (linha de baixo).
Círculos: o INMET de cada estação nos dias não vistos, na mesma escala de cor.

As previsões estão em NetCDF, uma variável por modelo, em
`artifacts/feature_study/global/predicoes/predicoes_rajada_{rs_sc,c5,c12}.nc` (geradas por
`scripts/exportar_predicoes_nc.py`), para reproduzir os mapas ou alimentar o dashboard.
""")

py("""
mapas = {r_: G / "predicoes" / f"predicoes_rajada_{r_}.nc" for r_ in ("rs_sc", "c5", "c12")}
if not all(p_.exists() for p_ in mapas.values()):
    print("previsões ainda não exportadas: rode `uv run python scripts/exportar_predicoes_nc.py`")
else:
    obs_est = nv.groupby("estacao")[TARGET].agg(media="mean", p99=lambda x: x.quantile(0.99)).join(
        est.drop_duplicates("estacao").set_index("estacao")[["latitude", "longitude", "cluster_id"]])
    dss = {r_: xr.open_dataset(p_) for r_, p_ in mapas.items()}
    MOD = [("ml_local", "ML local"), ("ml_global_trimestral", "ML global por trimestre"),
           ("ml_global_anual", "ML global anual"), ("lstm_local", "LSTM local"), ("lstm_global_anual", "LSTM global anual")]
    for nome, f, col_o in [("média", lambda x: x.mean("time"), "media"), ("P99", lambda x: x.quantile(0.99, "time"), "p99")]:
        cf = {r_: {v: f(ds[v]) for v in ["era5", *[m_ for m_, _ in MOD]]} for r_, ds in dss.items()}
        todos = [x for d_ in cf.values() for x in d_.values()]
        vmin = float(min(min(float(x.min()) for x in todos), obs_est[col_o].min()))
        vmax = float(max(max(float(x.max()) for x in todos), obs_est[col_o].max()))
        lim = float(np.nanpercentile(np.abs(np.concatenate(
            [(d_[m_] - d_["era5"]).values.ravel() for d_ in cf.values() for m_, _ in MOD])), 99))
        n = len(MOD) + 1
        fig, axs = plt.subplots(2, n, figsize=(5.2 * n, 11), constrained_layout=True)
        for j_, (v, rot) in enumerate([("era5", "ERA5"), *MOD]):
            for i_ in range(2):
                ax = axs[i_, j_]
                shp.plot(ax=ax, color="#f4f4f4", edgecolor="#cccccc", lw=0.3)
                if i_ == 1 and v == "era5":
                    ax.axis("off"); continue
                for r_, d_ in cf.items():
                    ds = dss[r_]
                    if i_ == 0:
                        im = ax.pcolormesh(ds.lon, ds.lat, d_[v], cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest")
                    else:
                        im2 = ax.pcolormesh(ds.lon, ds.lat, d_[v] - d_["era5"], cmap="RdBu_r", vmin=-lim,
                                            vmax=lim, shading="nearest")
                shp[shp.cluster.astype(int).isin(CLS)].boundary.plot(ax=ax, color="k", lw=0.5)
                if i_ == 0:
                    ax.scatter(obs_est.longitude, obs_est.latitude, c=obs_est[col_o], cmap="viridis", vmin=vmin,
                               vmax=vmax, s=28, edgecolors="w", linewidths=0.8, zorder=3)
                ax.set_xlim(-58.5, -38.5); ax.set_ylim(-34.5, -14); ax.set_aspect("equal")
                ax.set_title(f"{rot}" + ("" if i_ == 0 else " − ERA5"),
                             fontsize=10 * FS)
        fig.colorbar(im, ax=axs[0, :], shrink=0.7, label=f"{nome} da rajada diária (m/s)")
        fig.colorbar(im2, ax=axs[1, 1:], shrink=0.7, label="correção − ERA5 (m/s)")
        fig.suptitle(f"Extrapolação para a grade, {nome} da rajada diária 2000–2025: todas as opções de modelo "
                     "(círculos = INMET nos dias não vistos)", fontsize=13 * FS)
        plt.show()
""")

md("**Mapas por trimestre:** média da rajada do ERA5 (1ª coluna) e a correção de cada modelo em relação a ele, por trimestre.")

py("""
if all(p_.exists() for p_ in mapas.values()):
    mes_est = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
               9: "SON", 10: "SON", 11: "SON"}
    sz = {r_: {v: ds[v].groupby(xr.DataArray([mes_est[m] for m in ds.time.dt.month.values], dims="time",
                                             coords={"time": ds.time}, name="est")).mean("time").load()
               for v in ["era5", *[m_ for m_, _ in MOD]]} for r_, ds in dss.items()}
    lim = float(np.nanpercentile(np.abs(np.concatenate(
        [(d_[m_] - d_["era5"]).values.ravel() for d_ in sz.values() for m_, _ in MOD])), 99))
    vmin = min(float(d_["era5"].min()) for d_ in sz.values()); vmax = max(float(d_["era5"].max()) for d_ in sz.values())
    n = len(MOD) + 1
    fig, axs = plt.subplots(4, n, figsize=(4.4 * n, 21), constrained_layout=True)
    for i_, e in enumerate(EST4):
        for j_, (v, rot) in enumerate([("era5", "ERA5"), *MOD]):
            ax = axs[i_, j_]
            shp.plot(ax=ax, color="#f4f4f4", edgecolor="#cccccc", lw=0.3)
            for r_, d_ in sz.items():
                ds = dss[r_]
                if v == "era5":
                    im = ax.pcolormesh(ds.lon, ds.lat, d_[v].sel(est=e), cmap="viridis", vmin=vmin, vmax=vmax,
                                       shading="nearest")
                else:
                    im2 = ax.pcolormesh(ds.lon, ds.lat, (d_[v] - d_["era5"]).sel(est=e), cmap="RdBu_r", vmin=-lim,
                                        vmax=lim, shading="nearest")
            shp[shp.cluster.astype(int).isin(CLS)].boundary.plot(ax=ax, color="k", lw=0.4)
            ax.set_xlim(-58.5, -38.5); ax.set_ylim(-34.5, -14); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            nm_ = {"ml_local": "ml_local_x", "ml_global_trimestral": "ml_G_x", "lstm_local": "ls_local_x"}.get(v, "x_x")
            ax.set_title(f"ERA5, média {e}" if v == "era5" else rot_est(nm_, rot, e) + " − ERA5", fontsize=9 * FS)
    fig.colorbar(im, ax=axs[:, 0], shrink=0.4, label="média da rajada diária (m/s)")
    fig.colorbar(im2, ax=axs[:, 1:], shrink=0.4, label="correção − ERA5 (m/s)")
    fig.suptitle("Extrapolação para a grade por trimestre, 2000–2025", fontsize=13 * FS)
    plt.show()
    del sz
""")

md("Um **mês típico** (ganho perto da mediana do cluster, não escolhido a dedo), numa estação de cada região, em dias que nenhum modelo viu:")

py("""
def mes_exemplo(cl):
    # mês TÍPICO, não escolhido a dedo: na estação com mais dias, o mês (≥ 25 dias) cujo ganho de RMSE sobre o
    # ERA5 fica mais perto do ganho mediano dos meses do cluster
    s = nv[nv.cluster_id == cl].assign(mes=lambda x: x.time.dt.to_period("M"))
    e_ = s.groupby("estacao").size().idxmax()
    gm = (s.groupby(["estacao", "mes"]).filter(lambda x: len(x) >= 25)
           .groupby(["estacao", "mes"]).apply(lambda x: rmse(x.era5_gust_max, x[TARGET]) - rmse(x.ls_GA_eqm, x[TARGET])))
    alvo = gm.median()
    ge = gm.loc[e_]
    mes = (ge - alvo).abs().idxmin()
    return e_, s[(s.estacao == e_) & (s.mes == mes)].sort_values("time")


fig, axs = plt.subplots(4, 1, figsize=(15, 14.5), constrained_layout=True)
for ax, cl in zip(axs, (4, 3, 5, 12)):
    e_, s = mes_exemplo(cl)
    ax.plot(s.time, s[TARGET], color=C_INMET, marker="o", ms=4, lw=1.4, label="INMET (medido)")
    ax.plot(s.time, s.era5_gust_max, color=C_ERA5, lw=1.3, label="ERA5")
    for ce, _, nome, cor in MODS:
        ax.plot(s.time, s[ce], color=cor, lw=2.2 if ce == "ls_GA_eqm" else 1.1, label=rot_est(ce, nome, s.season.iloc[0]),
                ls="-" if ce.startswith("ls") else "--")
    ax.set_title(f"cluster {cl}, estação {e_}, {s.time.iloc[0]:%m/%Y} (RMSE no mês: ERA5 "
                 f"{rmse(s.era5_gust_max, s[TARGET]):.2f} → LSTM global anual {rmse(s.ls_GA_eqm, s[TARGET]):.2f})")
    ax.set_ylabel("rajada diária (m/s)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
for ax in axs:
    ax.legend(ncol=4, loc="upper left", fontsize=7 * FS)
plt.show()
""")

# ── 8 ──────────────────────────────────────────────────────────────────────
md("## 8. Resumo, recomendação e limites")

py("""
t = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")]
linhas = {}
for fam, ch, versao, rot in [("LSTM", "GA", "Corrigido ML", "LSTM global anual"),
                             ("LSTM", "G", "Corrigido ML", "LSTM global por trimestre"),
                             ("ML", "GA", "Corrigido ML", "alternativa barata: ML global anual"),
                             ("ML", "G", "Corrigido ML", "ML global por trimestre"),
                             ("ML", "A", "Corrigido ML", "modelo do cluster 3, sem treino local")]:
    linhas[rot] = t[(t.familia == fam) & (t.chave == ch) & (t.versao == versao)].set_index("cluster")[["RMSE", "POD_P99"]]
loc_ = t[(t.chave == "B" + t.cluster.astype(str)) & (t.versao == "Corrigido ML")]
for fam in ("ML", "LSTM"):
    linhas[f"{fam} local"] = loc_[loc_.familia == fam].set_index("cluster")[["RMSE", "POD_P99"]]
era_ = t[t.versao == "ERA5"].groupby("cluster")[["RMSE", "POD_P99"]].first()
linhas["ERA5"] = era_
res = pd.DataFrame({k: v.RMSE for k, v in linhas.items()}).T.reindex(columns=CLS)
pod = {k: v.POD_P99 for k, v in linhas.items()}
print("RMSE diário (m/s), meses de teste 2000–2024:")
print(res.round(2).to_string())
print()
print("POD no P99:")
print(pd.DataFrame(pod).T.reindex(columns=CLS).round(2).to_string())
""")

md("""
### O que aprendemos

1. **O ERA5 acerta a média, mas espalha o erro e corta os picos.** Também achata a diferença entre lugares
   calmos e ventosos.
2. **Só o vento local do ERA5 carrega informação forte.** Convecção e pressão ajudam pouco; uma seleção de 12
   variáveis basta. O relevo estático (`anor`) virava código de estação e foi retirado.
3. **O modelo melhora o dia a dia, mas comprime os extremos.** O **mapeamento de quantis** (capítulo 4) devolve a cauda,
   ao custo de um pouco de erro diário. **Só reescalar o ERA5, sem o modelo, não serve**: piora o erro do ERA5 e só
   multiplica os alarmes. Por isso o mapeamento é parte do modelo.
4. **A correção é regional.** Um modelo levado para uma região sem estações no treino não melhora o ERA5.
   Com estações da região no treino, melhora sempre.
5. **ML e LSTM empatam** quando têm dados da região; em região nova o ML generaliza um pouco melhor.
6. **Um único modelo para todos os clusters e o ano todo é o melhor de todos.** A **LSTM global
   anual** erra **0,17 a 0,75 m/s menos que o ERA5** em todos os clusters nos meses de teste e detecta tantos ou mais
   extremos (POD no P99 de 0,13 a 0,27, contra 0,08 a 0,22 do ERA5), exceto no cluster 5, onde detecta menos
   (0,09 contra 0,14).

### Recomendação

- **LSTM global anual**, treinada com estações de todas as regiões disponíveis. Um único modelo
  para todos os clusters e o ano todo.
- **Alternativa barata: ML global anual (CatBoost).** Fica a poucos centésimos, roda em menos de um minuto e
  não precisa de GPU.
- **Para levar a uma região nova, incluir estações dela no treino.**

### Limites

- **Rajadas convectivas de verão (DJF)** continuam difíceis: o ERA5 não as vê, e nenhum modelo as inventa.
- **O mapeamento de quantis carrega o clima de onde foi ajustado** e depende de poucos dias na cauda (P99,9 ainda subestimado).
- **Uma seed por ajuste**, e as arquiteturas não foram retriadas por região: diferenças de centésimos estão no ruído.
- **Cluster 5 sem 2025 e com efeito pequeno nos outros.** As estações do cluster 5 não têm 2025, então ele só é
  avaliado no teste de 2000–2024. Incluí-lo no treino global mudou o RMSE dos outros clusters em menos de 0,13 m/s
  (em geral piorou de 0,02 a 0,08 m/s no ML e ficou neutro na LSTM global anual).
- **Cluster 5 com vizinhança incompleta.** O ERA5 dele veio só com os pontos dentro dos polígonos, sem os vizinhos ao
  norte e a leste. As 24 colunas da própria célula são idênticas às da grade completa; as 25 de vizinhança e gradiente
  mudam (no teste com as 59 células em comum com a grade de RS/SC, 2010). Duas estações (A712 e A847) usam a célula
  disponível mais próxima.
- **Três colunas** (`mcpr_r75_fracpos`, `mcpr_l75_250`, `grad_mslp_*`) seguem a definição do spec sem conferência
  contra o pipeline original.

Detalhes de cada etapa em `documentation/` (09 estudo de features, 11 mapeamento de quantis, 12 variáveis, 13 RS/SC, 14 ML × LSTM).
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
