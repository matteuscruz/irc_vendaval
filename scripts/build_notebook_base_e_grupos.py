"""Gera `notebooks/feature_study_grupos/anteriores/raw_horario/base_horaria_e_grupos_1_e_4.ipynb`.

Notebook descritivo: mostra o que o modelo recebe — a base horária nova e os
grupos 1 e 4 — sem ajustar nenhum modelo. Serve para conhecer o dado antes de
ler qualquer resultado.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/anteriores/raw_horario/base_horaria_e_grupos_1_e_4.ipynb")

nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# A base horária nova e os grupos 1 e 4

O que o modelo recebe, sem ajustar nenhum modelo aqui. É o notebook para
conhecer o dado antes de ler qualquer resultado.

**Camadas do dataset** (585 colunas no arm `full`):

| camada | o que é | origem |
|---|---|---|
| **base** | 11 variáveis ERA5 × 24 horas, mais lat/lon | `test_cluster_3_hourly.nc` |
| **grupo 1** | vento a 100 m, rajada, camada limite, fricção... | `new_features/cluster3/sl/` |
| **grupo 4** | rugosidade da superfície e relevo | `sl/fsr` e `static/` |
| **cape** | energia convectiva disponível — fora dos dois grupos | `sl/cape` |

Cada linha é um **(estação, dia)**: as 24 horas de cada variável viram 24
colunas. O alvo é a rajada máxima daquele dia medida pelo INMET. Todas as
features são ERA5; o INMET entra só como alvo. Horário sempre em **UTC**
(o cluster 3 está em UTC−3, então o meio-dia local cai por volta das 15 UTC).
""")

py("""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
BASE = RAIZ / "artifacts/feature_study/cluster3_raw"
MASCARA = RAIZ / "artifacts/cluster_masks/cluster_3_mask.shp"
ALVO, PROXY = "daily_wind_gust_max", "wind_mag_max"
HORA_REF = 15          # ~ meio-dia local

meta = json.loads((BASE / "data/meta.json").read_text())
treino = pd.read_parquet(BASE / "data/sample_full.parquet")
teste = pd.read_parquet(BASE / "data/test.parquet")
tudo = pd.concat([treino, teste], ignore_index=True)

print(f"{len(tudo)} linhas (treino+validação {len(treino)}, teste {len(teste)}) | "
      f"{tudo.estacao.nunique()} estações | {tudo.time.min():%Y-%m-%d} a {tudo.time.max():%Y-%m-%d}")
""")

# ── 1. Composição ───────────────────────────────────────────────────────────
md("""
## 1. Composição: o que é base e o que é grupo

As cores abaixo se repetem em todo o notebook.
""")

py("""
from src.feature_study.group_importance import DESCRICAO, GEMEO_NA_BASE, ROTULO_GRUPO
from src.feature_study.hourly_flat import base_variable_of

UNIDADE = {
    "ws_h": "m/s", "sin_dir_h": "", "cos_dir_h": "", "msl_h": "Pa", "t2m_h": "K",
    "rh_h": "%", "td_dep_h": "K", "tp_h": "m", "tp_roll24h": "m", "tp_roll48h": "m",
    "tp_roll72h": "m", "fg10": "m/s", "ws10": "m/s", "ws100": "m/s",
    "shear_100_10": "m/s", "zust": "m/s", "blh": "m", "avg_ibld": "W/m²",
    "avg_ishf": "W/m²", "t2m": "K", "d2m": "K", "sp": "Pa", "cape": "J/kg", "fsr": "m",
}
DESCRICAO_BASE = {
    "ws_h": "vento a 10 m", "sin_dir_h": "direção do vento (seno)",
    "cos_dir_h": "direção do vento (cosseno)", "msl_h": "pressão ao nível do mar",
    "t2m_h": "temperatura a 2 m", "rh_h": "umidade relativa",
    "td_dep_h": "depressão do ponto de orvalho", "tp_h": "precipitação horária",
    "tp_roll24h": "precipitação acumulada 24 h", "tp_roll48h": "precipitação acumulada 48 h",
    "tp_roll72h": "precipitação acumulada 72 h",
}
DESC = {**DESCRICAO, **DESCRICAO_BASE}

CAMADA = {}
for col in meta["base_features"] + meta["new_features"]:
    if col in ("latitude", "longitude"):
        continue
    v = base_variable_of(col)
    if col.startswith("hf_"):
        CAMADA[v] = "base"
    else:
        CAMADA[v] = ROTULO_GRUPO.get(v, "?")

CORES = {"base": "#7a7a7a", "grupo 1": "#2a7ab0", "grupo 4": "#d98c3f", "sem grupo": "#8a6fb0"}

linhas = []
for col in meta["base_features"] + meta["new_features"]:
    if col in ("latitude", "longitude"):
        continue
    v = base_variable_of(col)
    linhas.append({"camada": CAMADA[v], "variável": v})
inv = (pd.DataFrame(linhas).groupby(["camada", "variável"]).size()
       .rename("colunas").reset_index())
inv["descrição"] = inv["variável"].map(DESC)
inv["unidade"] = inv["variável"].map(UNIDADE).fillna("")
inv["repete a base?"] = inv["variável"].map(lambda v: f"sim (~ {GEMEO_NA_BASE[v]})" if v in GEMEO_NA_BASE else "")
ordem = {"base": 0, "grupo 1": 1, "grupo 4": 2, "sem grupo": 3}
inv = inv.sort_values(["camada", "variável"], key=lambda s: s.map(ordem) if s.name == "camada" else s)
print(inv.to_string(index=False))
print()
resumo = inv.groupby("camada").agg(variáveis=("variável", "size"), colunas=("colunas", "sum")).reindex(list(ordem))
resumo.loc["lat/lon", ["variáveis", "colunas"]] = [2, 2]
print(resumo.astype(int).to_string())
print(f"\\ntotal do arm full: {int(resumo.colunas.sum())} colunas")
""")

py("""
fig, ax = plt.subplots(figsize=(9, 3.4))
r = resumo.drop("lat/lon")
ax.barh(r.index[::-1], r.colunas[::-1], color=[CORES[k] for k in r.index[::-1]], edgecolor="black")
for y, (k, v) in enumerate(zip(r.index[::-1], r.colunas[::-1])):
    ax.text(v + 4, y, f"{int(v)} colunas · {int(r.loc[k, 'variáveis'])} variáveis", va="center", fontsize=9.5)
ax.set_xlim(0, r.colunas.max() * 1.45)
ax.set_xlabel("colunas no arm full")
ax.set_title("Quanto de cada camada o modelo recebe")
plt.tight_layout(); plt.show()
""")

md("""
Os grupos do Paulo seguem o spec (`features_grupos_1_e_4.md`). O **grupo 1**
inclui três variáveis que chegam pela base horária por estação — `W10`, `T2m` e
a depressão do ponto de orvalho —, então na tabela elas aparecem como **base**.
Também estão no grupo 1 do spec lat/lon, que entram na base como coordenadas.
""")

# ── 2. Uma linha ────────────────────────────────────────────────────────────
md("""
## 2. Como é uma linha: o dia de uma estação em 24 horas

Cada linha do dataset é um dia numa estação. Abaixo, o dia de **maior rajada
observada** no teste, com as variáveis principais de cada camada. A linha
tracejada é a rajada máxima do INMET nesse dia.
""")

py("""
def _serie(linha, var):
    prefixo = "hf_" if CAMADA[var] == "base" else "hfn_"
    nome = var.removesuffix("_h") if prefixo == "hf_" else var
    return np.array([linha[f"{prefixo}{nome}_h{h:02d}"] for h in range(24)], dtype=float)


def anatomia(estacao=None, data=None, variaveis=("ws_h", "fg10", "ws100", "cape", "blh", "t2m")):
    if estacao is None:
        i = teste[ALVO].idxmax()
        linha = teste.loc[i]
    else:
        linha = teste[(teste.estacao == estacao) & (teste.time == pd.Timestamp(data))].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, constrained_layout=True)
    for ax, v in zip(axes.ravel(), variaveis):
        y = _serie(linha, v)
        ax.plot(range(24), y, "o-", color=CORES[CAMADA[v]], lw=1.8, ms=4)
        if UNIDADE.get(v) == "m/s":
            ax.axhline(linha[ALVO], color="#111", ls="--", lw=1.2, label=f"INMET (máx. do dia) {linha[ALVO]:.1f}")
            ax.legend(fontsize=8.5, loc="upper left")
        ax.set_title(f"{v} — {DESC.get(v, '')}  [{CAMADA[v]}]", fontsize=10.5)
        ax.set_ylabel(UNIDADE.get(v, ""))
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("hora do dia (UTC)")
    fig.suptitle(f"{linha.estacao} — {linha.time:%d/%m/%Y}  |  rajada observada {linha[ALVO]:.1f} m/s, "
                 f"ERA5 cru {linha[PROXY]:.1f} m/s", fontsize=13)
    plt.show()
    return linha


_ = anatomia()
""")

md("""
Para outro dia, chame `anatomia("A834", "2015-07-30")` com a estação e a data.
O eixo x são as 24 colunas de cada variável: é isso que o modelo lê.
""")

# ── 3. Ciclo diário médio ───────────────────────────────────────────────────
md("""
## 3. O perfil típico de cada variável

Média por hora do dia, com a faixa entre os percentis 10 e 90. Mostra o que é
comum e o que varia — e quais variáveis têm ciclo diário forte, que é justamente
a informação que a base diária destruía ao tirar média e máximo.
""")

py("""
def perfil(var):
    prefixo = "hf_" if CAMADA[var] == "base" else "hfn_"
    nome = var.removesuffix("_h") if prefixo == "hf_" else var
    cols = [f"{prefixo}{nome}_h{h:02d}" for h in range(24)]
    return tudo[cols].to_numpy(dtype=float)


horarias = [v for v in inv["variável"] if not v.startswith("nf_")]
fig, axes = plt.subplots(5, 5, figsize=(19, 14), sharex=True, constrained_layout=True)
for ax, v in zip(axes.ravel(), horarias):
    m = perfil(v)
    ax.fill_between(range(24), np.nanpercentile(m, 10, axis=0), np.nanpercentile(m, 90, axis=0),
                    color=CORES[CAMADA[v]], alpha=0.22)
    ax.plot(range(24), np.nanmean(m, axis=0), color=CORES[CAMADA[v]], lw=2)
    ax.set_title(f"{v} [{CAMADA[v]}]", fontsize=9.5)
    ax.set_ylabel(UNIDADE.get(v, ""), fontsize=8)
    ax.set_xticks(range(0, 24, 6))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
for ax in axes.ravel()[len(horarias):]:
    ax.axis("off")
fig.suptitle("Perfil médio (linha) e faixa p10–p90 por hora do dia — todas as variáveis horárias", fontsize=14)
plt.show()
""")

md("""
### Nem toda variável tem ciclo diário

Olhando a figura acima, `tp_roll24h/48h/72h` e `fsr` são quase retas. A tabela
abaixo mede isso: a variação **dentro** do dia (entre as 24 colunas de uma
linha) contra a variação **entre dias na mesma estação**. Tira-se a média de cada
estação antes de comparar — sem isso, variáveis com diferença fixa entre
estações (a pressão de superfície, por causa da altitude) pareceriam constantes
ao longo do dia sem ser.

A última coluna diz que fração da variação total é simplesmente diferença
**entre estações**.
""")

py("""
linhas = []
est_das_linhas = tudo.estacao.to_numpy()
for v in horarias:
    m = perfil(v)
    media_dia = np.nanmean(m, axis=1)
    media_estacao = pd.Series(media_dia).groupby(est_das_linhas).transform("mean").to_numpy()
    dentro = float(np.nanmean(np.nanstd(m, axis=1)))
    entre = float(np.nanstd(media_dia - media_estacao))
    frac_estacoes = 1 - (np.nanvar(media_dia - media_estacao) / np.nanvar(media_dia))
    linhas.append({"variável": v, "camada": CAMADA[v], "dentro do dia": dentro,
                   "entre dias (mesma estação)": entre,
                   "dentro / entre": dentro / entre if entre else np.nan,
                   "% da variação que é entre estações": 100 * frac_estacoes})
ciclo = pd.DataFrame(linhas).sort_values("dentro / entre").reset_index(drop=True)
# Duas lentes, porque a razão sozinha engana: com o `fsr` o denominador é
# minúsculo (numa mesma estação ele quase não muda) e a razão dá 0,43, o que
# soaria como "tem ciclo diário". A coluna de % entre estações desfaz o engano.
ciclo["leitura"] = np.select(
    [ciclo["% da variação que é entre estações"] > 90, ciclo["dentro / entre"] < 0.3],
    ["diferença entre estações domina", "varia devagar no dia"],
    default="tem ciclo diário")
print(ciclo.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), constrained_layout=True)
o = ciclo.sort_values("dentro / entre")
axes[0].barh(o["variável"], o["dentro / entre"], color=[CORES[k] for k in o.camada],
             edgecolor="black", linewidth=0.5)
axes[0].axvline(0.3, color="#c44", ls="--", lw=1.2, label="0,3 — abaixo disto varia devagar no dia")
axes[0].set_xlabel("variação dentro do dia ÷ variação entre dias (mesma estação)")
axes[0].set_title("Quanto muda ao longo das 24 horas")
axes[0].legend(loc="lower right")
o = ciclo.sort_values("% da variação que é entre estações")
axes[1].barh(o["variável"], o["% da variação que é entre estações"],
             color=[CORES[k] for k in o.camada], edgecolor="black", linewidth=0.5)
axes[1].set_xlabel("% da variação total que é diferença entre estações")
axes[1].set_title("Quanto é só 'que estação é essa'")
plt.show()
""")

md("""
**`fsr` é o caso extremo:** quase toda a sua variação é diferença **entre
estações**; numa mesma estação ele praticamente não muda, nem de hora em hora
nem de dia para dia. Na prática é uma estática — 24 colunas para um único número
por estação —, o que combina com o grupo 4 ter acrescentado pouco.

`tp_roll*`, `msl`, `sp` e `d2m` variam devagar ao longo do dia (razão abaixo de
0,3), mas não são constantes. Para elas, embaralhar só uma faixa de horas mede
pouco, porque as outras horas carregam quase a mesma informação. Por isso o
número de colunas **efetivas** é menor que as 585 do arm `full`.
""")

# ── 4. Grupo 4: estáticas ───────────────────────────────────────────────────
md("""
## 4. Grupo 4: as estáticas de relevo, estação por estação

As sete estáticas não variam no tempo: cada estação tem um valor fixo. Como a
base já traz lat/lon, elas funcionam sobretudo como **identificador de estação**
— o que ajuda a entender por que o grupo 4 acrescentou pouco no estudo.
""")

py("""
ESTATICAS = [c for c in meta["new_features"] if c.startswith("nf_")]
est = tudo.groupby("estacao")[ESTATICAS + ["latitude", "longitude"]].first()
est["n_dias"] = tudo.groupby("estacao").size()
nomes = {c: f"{c.removeprefix('nf_')} — {DESC.get(c.removeprefix('nf_'), DESC.get(c, ''))}" for c in ESTATICAS}

print(est.round(3).to_string())
print()
z = (est[ESTATICAS] - est[ESTATICAS].mean()) / est[ESTATICAS].std()
fig, ax = plt.subplots(figsize=(10, 4.6))
sns.heatmap(z.rename(columns={c: c.removeprefix("nf_") for c in ESTATICAS}), annot=est[ESTATICAS].round(2).rename(
    columns={c: c.removeprefix("nf_") for c in ESTATICAS}), fmt="", cmap="RdBu_r", center=0,
    cbar_kws={"label": "desvios-padrão da média entre estações"}, linewidths=0.5, ax=ax)
ax.set_title("Grupo 4 — valor de cada estática por estação (cor = posição relativa entre as 9 estações)")
ax.set_ylabel("")
plt.tight_layout(); plt.show()
""")

py("""
mascara = gpd.read_file(MASCARA)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
for ax, col, cmap in zip(axes, ["nf_orog_height", "nf_sdor", "nf_lsm"], ["terrain", "YlOrBr", "Blues"]):
    mascara.plot(ax=ax, facecolor="#eef2f5", edgecolor="#9fb3c8", linewidth=0.8)
    sc = ax.scatter(est.longitude, est.latitude, c=est[col], cmap=cmap, s=260,
                    edgecolor="black", linewidth=0.8, zorder=3)
    desloca = {"B807": (0, -24), "A801": (0, 13)}
    for e, r in est.iterrows():
        ax.annotate(f"{e}\\n{r[col]:.2f}", (r.longitude, r.latitude), textcoords="offset points",
                    xytext=desloca.get(e, (0, 13)), ha="center", fontsize=8)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(nomes[col], fontsize=10.5)
fig.suptitle("Grupo 4 — onde ficam as estações e como é o relevo em cada uma", fontsize=13)
plt.show()
""")

# ── 5. Repetição entre base e grupo 1 ───────────────────────────────────────
md("""
## 5. O que se repete entre a base e o grupo 1

Correlação, no horário de referência (15 UTC), entre as variáveis da base e as
da grade. Correlação perto de 1 significa **a mesma informação por dois
caminhos** — é o que faz `ws10`, `t2m` e `d2m` não serem novidade.
""")

py("""
def coluna(v, h=HORA_REF):
    prefixo = "hf_" if CAMADA[v] == "base" else "hfn_"
    nome = v.removesuffix("_h") if prefixo == "hf_" else v
    return f"{prefixo}{nome}_h{h:02d}"


vars_base = [v for v in inv["variável"] if CAMADA[v] == "base"]
vars_grade = [v for v in inv["variável"] if CAMADA[v] in ("grupo 1", "sem grupo") or v == "fsr"]

cm = pd.DataFrame({b: {g: tudo[coluna(b)].corr(tudo[coluna(g)]) for g in vars_grade} for b in vars_base})
fig, ax = plt.subplots(figsize=(11, 6.6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "correlação"}, linewidths=0.4, ax=ax, annot_kws={"size": 8.5})
ax.set_xlabel("variáveis da BASE"); ax.set_ylabel("variáveis da GRADE (grupo 1, grupo 4, cape)")
ax.set_title(f"Correlação entre base e grade às {HORA_REF} UTC — valores ≥ 0,95 são a mesma informação")
plt.tight_layout(); plt.show()

fortes = cm.stack()
fortes = fortes[fortes.abs() >= 0.95].sort_values(ascending=False)
print("pares com |correlação| ≥ 0,95:")
print(fortes.round(3).to_string())
""")

py("""
# `sp` × `msl`: entre estações a correlação é baixa (a altitude separa), dentro da estação é quase 1.
a, b = coluna("sp"), coluna("msl_h")
dentro = tudo[[a, b]] - tudo.groupby("estacao")[[a, b]].transform("mean")
print(f"corr(sp, msl) entre todas as linhas:   {tudo[a].corr(tudo[b]):.3f}   (a altitude da estação domina)")
print(f"corr(sp, msl) dentro de cada estação:  {dentro[a].corr(dentro[b]):.3f}   (só a variação no tempo)")
""")

# ── 6. Relação com o alvo ───────────────────────────────────────────────────
md("""
## 6. Quanto cada variável se relaciona com a rajada observada

Para cada variável, a **melhor hora do dia**: a maior correlação (em módulo) entre
alguma das suas 24 colunas e a rajada máxima do INMET.

Cuidado ao ler: correlação **não é importância**. Ela mede relação linear
isolada e não enxerga redundância — `ws10` aparecerá tão alta quanto `ws`
justamente por ser a mesma coisa. As estáticas (hachuradas) refletem só a
diferença entre estações. O ranking de importância está no notebook
`grupos_1_e_4_importancia_por_variavel`.
""")

py("""
linhas = []
for v in inv["variável"]:
    if v.startswith("nf_"):
        r = tudo[v].corr(tudo[ALVO])
        linhas.append({"variável": v, "camada": CAMADA[v], "corr": r, "hora": None})
    else:
        cors = {h: tudo[coluna(v, h)].corr(tudo[ALVO]) for h in range(24)}
        h = max(cors, key=lambda k: abs(cors[k]))
        linhas.append({"variável": v, "camada": CAMADA[v], "corr": cors[h], "hora": h})
rel = pd.DataFrame(linhas)
rel["abs"] = rel["corr"].abs()
rel = rel.sort_values("abs")

fig, ax = plt.subplots(figsize=(10.5, 9))
ax.barh(rel["variável"], rel["corr"], color=[CORES[k] for k in rel.camada], edgecolor="black",
        hatch=["//" if v.startswith("nf_") else "" for v in rel["variável"]], linewidth=0.5)
ax.axvline(0, color="black", lw=0.8)
for y, (_, r) in enumerate(rel.iterrows()):
    txt = f"{r['corr']:+.2f}" + (f" ({int(r.hora):02d}h)" if r.hora is not None and not pd.isna(r.hora) else " (estática)")
    ax.text(r["corr"] + (0.012 if r["corr"] >= 0 else -0.012), y, txt, va="center",
            ha="left" if r["corr"] >= 0 else "right", fontsize=8)
ax.set_xlim(-0.5, 0.95)
ax.set_xlabel("correlação com a rajada máxima observada (melhor hora)")
ax.set_title("Relação linear de cada variável com o alvo — por camada")
mãos = [plt.Rectangle((0, 0), 1, 1, fc=CORES[k], ec="black") for k in CORES]
ax.legend(mãos, CORES.keys(), loc="lower right")
plt.tight_layout(); plt.show()
""")

# ── 7. A tabela ─────────────────────────────────────────────────────────────
md("""
## 7. A tabela em si

Seis linhas, com o alvo, o ERA5 cru e uma coluna de cada camada. Cada coluna
horária tem o sufixo da hora (`_h15` = 15 UTC).
""")

py("""
mostra = ["estacao", "time", ALVO, PROXY, "latitude", "longitude",
          "hf_ws_h15", "hf_msl_h15", "hfn_fg10_h15", "hfn_ws100_h15", "hfn_cape_h15", "hfn_fsr_h15",
          "nf_orog_height", "nf_lsm"]
print(teste[mostra].sample(6, random_state=7).sort_values(["estacao", "time"])
      .rename(columns={ALVO: "INMET_rajada", PROXY: "ERA5_cru"}).round(3).to_string(index=False))
print()
print("cobertura — linhas por estação e trimestre (treino + validação + teste):")
cob = tudo.pivot_table(index="estacao", columns="season", values=ALVO, aggfunc="size")[["DJF", "MAM", "JJA", "SON"]]
cob["total"] = cob.sum(axis=1)
print(cob.to_string())
""")

md("""
### O que reter

- A **base** (266 colunas) tem as variáveis clássicas do ERA5 em 24 horas. Ela já
  contém vento, temperatura, umidade, pressão e chuva.
- O **grupo 1** acrescenta 7 variáveis de fato novas (`fg10`, `ws100`,
  `shear_100_10`, `zust`, `blh`, `avg_ibld`, `avg_ishf`) e 4 que repetem a base
  (`ws10`, `t2m`, `d2m`, `sp`) — a seção 5 mostra isso nos números.
- O **grupo 4** é, na maior parte, geografia fixa por estação; só `fsr` varia no
  tempo.
- O `cape` não pertence a nenhum dos dois grupos.
- Os dados cobrem apenas as **9 estações** com alvo INMET em 2000–2024.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
DESTINO.parent.mkdir(exist_ok=True)
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
