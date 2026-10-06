"""Gera `notebooks/feature_study_grupos/atual/grupos_1_e_4_importancia_por_variavel.ipynb` (agora com os 4 grupos).

Importância por permutação por VARIÁVEL (família de colunas), na estrutura por
grupo (`cluster3_groups`: uma linha por estação-dia na hora do pico).
O nome do arquivo é mantido; o conteúdo cobre os grupos 1, 2, 3 e 4.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/grupos_1_e_4_importancia_por_variavel.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Grupos 1, 2, 3 e 4 — que variável acrescenta informação?

Versão atualizada: cobre os **quatro grupos** (antes só 1 e 4), na estrutura por grupo
(uma linha por estação-dia, na hora do pico da rajada ERA5). A unidade de análise é a
**variável** (a família de colunas: valor no ponto, janelas r75/r250, defasagens etc.).

O estudo pago já respondeu ao nível de **grupo** (bootstrap pareado; ver
`resultados_finais_grupos.ipynb`). Aqui se desce à variável com um instrumento mais
fraco, a **importância por permutação**: quanto o RMSE de teste piora ao embaralhar as
colunas da variável. Mede **uso** pelo modelo, não ganho fora da amostra, e variáveis
colineares dividem o crédito.

**Dado:** cluster 3, 9 estações com alvo INMET, 2000–2024. Modelo: LightGBM por trimestre
(sempre o mesmo, para a comparação entre variáveis ser justa), treinado no treino do estudo
e avaliado no teste (jan/abr/jul/out). Os controles do estudo entram como régua.
""")

py("""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor

from src.feature_study.diagnostics.group_importance import permutacao_por_bloco

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
DADOS = RAIZ / "artifacts/feature_study/cluster3_groups/data"
MASCARA = RAIZ / "artifacts/cluster_masks/cluster_3_mask.shp"
SEASONS = ("DJF", "MAM", "JJA", "SON")
ALVO, REF = "daily_wind_gust_max", "era5_gust_max"

meta = json.loads((DADOS / "meta.json").read_text())
GRUPO_DE = meta["group_of"]
arms = {a["name"]: a for a in json.loads((DADOS / "arms.json").read_text())}
FEATURES = arms["full"]["features"]
treino = pd.read_parquet(DADOS / "sample_full.parquet")
treino = treino[treino._split == "train"]
teste = pd.read_parquet(DADOS / "test.parquet")
print(f"{len(FEATURES)} features | treino {len(treino)} | teste {len(teste)} linhas | "
      f"{teste.estacao.nunique()} estações")

# BASE antiga (12 variáveis do ERA5 horário por estação) na hora do pico, SEM nenhum grupo:
# o modelo "só BASE" treina nas mesmas linhas. Não entra em FEATURES.
from src.feature_study.data.base_only import base_at_peak, base_columns

BASE_COLS = base_columns()
picos = pd.read_parquet(RAIZ / "artifacts/feature_study/cluster3_groups/_cache/_daily_peak_cache.parquet",
                        columns=["estacao", "time", "hora_pico_utc"])


def com_base(df):
    chaves = df[["estacao", "time"]].merge(picos, on=["estacao", "time"], how="left")
    b = base_at_peak(RAIZ / "dataset/raw", chaves)
    out = df.merge(b.drop(columns="hora_pico_utc"), on=["estacao", "time"], how="left")
    assert len(out) == len(df), "a junção da BASE duplicou ou perdeu linhas"
    return out


# seleção congelada (escolhida na validação): as 12 variáveis do arm `sel__val12`
from src.feature_study.selection.selected import columns_of, load_selection

SEL = load_selection(RAIZ / "config/selected_features_val12.json")
SEL_COLS = columns_of(SEL["variables"], FEATURES)
print(f"seleção: {len(SEL['variables'])} variáveis, {len(SEL_COLS)} colunas (fp {SEL['columns_fp']})")

treino, teste = com_base(treino), com_base(teste)
print(f"BASE na hora do pico: {len(BASE_COLS)} colunas | NaN no teste: "
      f"{teste[BASE_COLS].isna().any(axis=1).mean():.2%} das linhas")
""")

md("""
## 1. As variáveis de cada grupo

Cada coluna é reduzida à sua **variável de origem** retirando sufixos de estatística
(`_ponto`, `_r75_*`, `_r250_*`, `_c75_250`, `_l75_250`) e de defasagem (`_lag1h`,
`_max_prev3h`, `_std_prev3h`, `_delta3h`). Variável = bloco que é embaralhado de uma vez.
""")

py("""
SUF = re.compile(r"(_ponto|_r75_\\w+|_r250_\\w+|_c75_250|_l75_250|_lag1h|_max_prev3h|_std_prev3h|_delta3h)$")
ROTULO = {"grupo1": "grupo 1", "grupo2": "grupo 2", "grupo3": "grupo 3", "grupo4": "grupo 4"}


def variavel(col):
    return SUF.sub("", col)


blocos, grupo_var = {}, {}
for col in FEATURES:
    g = GRUPO_DE.get(col)
    if g is None:           # latitude/longitude etc. já têm grupo; o resto não entra
        continue
    v = variavel(col)
    blocos.setdefault(v, []).append(col)
    grupo_var[v] = ROTULO[g]
inv = pd.DataFrame({"variável": list(blocos), "grupo": [grupo_var[v] for v in blocos],
                    "colunas": [len(x) for x in blocos.values()]}).sort_values(["grupo", "variável"])
print(inv.groupby("grupo").agg(variáveis=("variável", "size"), colunas=("colunas", "sum")).to_string())
print()
for g, d in inv.groupby("grupo"):
    print(g, "->", ", ".join(d["variável"]))
""")

md("""
### Todas as colunas, uma a uma

Lê direto dos 4 parquets (`dataset/raw/feature_study_cluster_3`) e marca o que o estudo
**usa** e o que fica de fora, com o motivo. Mais a BASE antiga, que não faz parte da
estrutura por grupo.
""")

py("""
import pyarrow.parquet as pq

RAW = RAIZ / "dataset/raw/feature_study_cluster_3"
CHAVES = {"estacao", "time", "data", "date", "hora_pico_utc", "codigo_estacao"}
BASE_ANTIGA = ["ws_h", "wd_h", "sin_dir_h", "cos_dir_h", "msl_h", "t2m_h", "rh_h", "td_dep_h",
               "tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h"]
linhas = []
for g in (1, 2, 3, 4):
    nomes = [c for c in pq.ParquetFile(RAW / f"features_grupo{g}_cluster3.parquet").schema_arrow.names
             if c not in CHAVES]
    for c in nomes:
        if c in FEATURES:
            uso, motivo = "usada", ""
        elif c == "cin":
            uso, motivo = "fora", "75 % nulo (nada é imputado)"
        elif c in ("grid_lat", "grid_lon", "dist_celula_km"):
            uso, motivo = "fora", "identificador da célula"
        elif c.endswith("_n"):
            uso, motivo = "fora", "contagem constante por estação"
        else:
            uso, motivo = "fora", "outro"
        linhas.append({"grupo": f"grupo {g}", "coluna": c, "variável": variavel(c), "uso": uso,
                       "grupo no estudo": ROTULO.get(GRUPO_DE.get(c), "—"), "motivo": motivo})
for c in BASE_ANTIGA:
    linhas.append({"grupo": "BASE", "coluna": c, "variável": c, "uso": "fora",
                   "grupo no estudo": "—", "motivo": "ERA5 horário antigo, fora da estrutura por grupo"})
todas = pd.DataFrame(linhas)
pd.set_option("display.max_rows", None)
print(todas.groupby(["grupo", "uso"]).size().unstack(fill_value=0).to_string())
print()
for g, d in todas.groupby("grupo"):
    print(f"== {g} ({len(d)} colunas) ==")
    print(d[["coluna", "uso", "grupo no estudo", "motivo"]].to_string(index=False))
    print()
""")

md("""
## 2. Ajuste e permutação

Um LightGBM por trimestre. Além das variáveis, medimos o **grupo inteiro** como um bloco
(se embaralhar o grupo dói muito mais que a soma das partes, as variáveis são redundantes)
e os dois **controles** do estudo, que dão o piso do que é "uso sem informação".
""")

py("""
modelos, dados_season = {}, {}
for s in SEASONS:
    tr, te = treino[treino.season == s], teste[teste.season == s]
    modelos[s] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[FEATURES], tr[ALVO])
    dados_season[s] = (te[FEATURES], te[ALVO].to_numpy(float), te.estacao.to_numpy(), te)

# controles do estudo: ruído e estáticas permutadas, como blocos extras
CTRL = {"CTRL ruído": ["ctrl_noise"]}
perm = [c for c in teste.columns if c.startswith("perm_")]
blocos_c = dict(blocos)
for g in ROTULO:
    blocos_c[f"GRUPO {g[-1]} (inteiro)"] = sorted(c for v in blocos if grupo_var[v] == ROTULO[g] for c in blocos[v])
imp = []
for s in SEASONS:
    x, y, est, _ = dados_season[s]
    r = permutacao_por_bloco(modelos[s], x, y, est, blocos_c, n_repeticoes=5, seed=42)
    r["season"] = s
    imp.append(r)
imp = pd.concat(imp, ignore_index=True)
imp["grupo"] = imp.variavel.map(grupo_var).fillna("inteiro")
print(f"{imp.variavel.nunique()} blocos × {len(SEASONS)} trimestres × 5 repetições")
""")

md("""
### Importância coluna a coluna

A análise por **variável** (família) agrupa as colunas de uma variável (ponto, r75, r250,
defasagens) num só bloco. Para mostrar **todas as colunas da lista**, repetimos a permutação
**uma coluna por vez** (167 colunas usadas). Colunas irmãs são colineares, então cada uma
isolada tende a parecer mais fraca que a família inteira; as duas leituras se complementam.
""")

py("""
imp_col = []
for s in SEASONS:
    x, y, est, _ = dados_season[s]
    r = permutacao_por_bloco(modelos[s], x, y, est, {c: [c] for c in FEATURES}, n_repeticoes=5, seed=42)
    r["season"] = s
    imp_col.append(r)
imp_col = pd.concat(imp_col, ignore_index=True)

# as 185 colunas da lista (4 parquets): usadas e fora, com o motivo
col_tab = todas[todas.grupo != "BASE"].set_index("coluna")[["grupo", "uso", "motivo"]].copy()
col_tab["família"] = [variavel(c) for c in col_tab.index]
m = imp_col.groupby("variavel").agg(d_rmse=("d_rmse", "mean"), sd=("d_rmse", "std"))
col_tab = col_tab.join(m)
col_tab = col_tab.join(imp_col.pivot_table(index="variavel", columns="season", values="d_rmse")[list(SEASONS)])
print(f"{len(col_tab)} colunas na lista | {int((col_tab.uso == 'usada').sum())} usadas | "
      f"{int((col_tab.uso == 'fora').sum())} fora do modelo")
# latitude/longitude ficam no grupo 4 da lista; o estudo as trata como grupo 1 (base) ao treinar
""")

md("""
## 3. Ranking por variável, com os quatro grupos lado a lado

Piora do RMSE ao embaralhar a variável (média dos 4 trimestres, barra = desvio entre
trimestres e repetições; **não** é IC de bootstrap).
""")

py("""
CORES = {"grupo 1": "#2a7ab0", "grupo 2": "#d98c3f", "grupo 3": "#5aa469", "grupo 4": "#8a6fb0"}
por_var = (imp[imp.grupo.isin(CORES)].groupby(["variavel", "grupo"], as_index=False)
           .agg(d_rmse=("d_rmse", "mean"), sd=("d_rmse", "std"), d_rmse_p90=("d_rmse_p90", "mean"))
           .sort_values("d_rmse", ascending=False))

GCOR = {"grupo 1": "#2a7ab0", "grupo 2": "#d98c3f", "grupo 3": "#5aa469", "grupo 4": "#8a6fb0"}
altura = max((col_tab.grupo == g).sum() for g in GCOR) * 0.21 + 2
fig, axes = plt.subplots(1, 4, figsize=(22, altura), constrained_layout=True)
for ax, (g, cor) in zip(axes, GCOR.items()):
    d = col_tab[col_tab.grupo == g]
    usadas = d[d.uso == "usada"].sort_values("d_rmse")
    fora = d[d.uso == "fora"]
    ordem_c = list(fora.index) + list(usadas.index)            # fora embaixo, maior no topo
    y = np.arange(len(ordem_c))
    ax.barh(y[len(fora):], usadas.d_rmse, xerr=usadas.sd.fillna(0), color=cor, edgecolor="black", linewidth=0.3, capsize=1.5)
    ax.barh(y[:len(fora)], np.zeros(len(fora)), color="#dddddd", hatch="///", edgecolor="#888")
    for yi, c in zip(y[:len(fora)], fora.index):
        ax.text(0, yi, f" fora: {fora.loc[c, 'motivo']}", va="center", fontsize=6.5, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(ordem_c, fontsize=7)
    ax.axvline(0, color="k", lw=0.7)
    ax.margins(y=0.004)
    ax.set_title(f"{g}: {len(d)} colunas ({len(usadas)} usadas, {len(fora)} fora)", fontsize=10.5)
    ax.set_xlabel("piora do RMSE (m/s), escala própria", fontsize=8.5)
fig.suptitle("Importância por COLUNA, todas as variáveis da lista (cinza hachurado = fora do modelo)", fontsize=13)
plt.show()
print(col_tab[["grupo", "uso", "d_rmse", "sd", *SEASONS, "motivo"]].round(4).sort_values(["grupo", "d_rmse"], ascending=[True, False]).to_string())
""")

py("""
# O grupo inteiro dói mais do que a soma das partes? (redundância)
inteiro = imp[imp.variavel.str.startswith("GRUPO")].groupby("variavel").d_rmse.mean()
soma = por_var.groupby("grupo").d_rmse.sum()
comp = pd.DataFrame({"grupo": list(CORES),
                     "embaralhando o grupo INTEIRO": [inteiro[f"GRUPO {g[-1]} (inteiro)"] for g in CORES],
                     "soma das variáveis isoladas": [soma[g] for g in CORES]})
comp["redundância (inteiro − soma)"] = comp.iloc[:, 1] - comp.iloc[:, 2]
print(comp.round(4).to_string(index=False))
""")

md("""
Leitura: inteiro > soma indica variáveis substituíveis entre si (cada uma parece fraca
porque as vizinhas compensam). Compare com o estudo pago: um grupo pode concentrar muito
**uso** e ainda assim não acrescentar nada fora da amostra.
""")

md("""
## 4. Espacial — onde cada variável importa

A mesma permutação com o erro recalculado por estação, para **cada uma das 12 variáveis
selecionadas** (ordenadas por grupo e importância), para comparar a geografia entre elas.
""")

py("""
coords = teste.groupby("estacao")[["latitude", "longitude"]].first()
cols_est = [c for c in imp.columns if c.startswith("d_rmse__")]
espacial = imp[imp.grupo.isin(CORES)].groupby("variavel")[cols_est].mean()
espacial.columns = [c.replace("d_rmse__", "") for c in espacial.columns]
TOP = [por_var[por_var.grupo == g].variavel.iloc[0] for g in CORES]
mascara = gpd.read_file(MASCARA)

SEL_VARS = sorted(SEL["variables"], key=lambda v: (grupo_var[v], -por_var.set_index("variavel").d_rmse[v]))
nlin = int(np.ceil(len(SEL_VARS) / 4))
fig, axes = plt.subplots(nlin, 4, figsize=(20, 5.0 * nlin), constrained_layout=True)
for ax, var in zip(axes.ravel(), SEL_VARS):
    val = espacial.loc[var]
    mascara.plot(ax=ax, facecolor="#eef2f5", edgecolor="#9fb3c8", linewidth=0.8)
    sc = ax.scatter(coords.longitude, coords.latitude, c=val[coords.index], cmap="YlOrRd", vmin=0,
                    vmax=max(val.max(), 1e-3), s=220, edgecolor="black", linewidth=0.7, zorder=3)
    for e in coords.index:
        ax.annotate(f"{val[e]:+.3f}", (coords.longitude[e], coords.latitude[e]),
                    textcoords="offset points", xytext=(0, -22 if e == "B807" else 12), ha="center", fontsize=8)
    ax.set_title(f"{var}  ({grupo_var[var]})")
    plt.colorbar(sc, ax=ax, shrink=0.8)
for ax in axes.ravel()[len(SEL_VARS):]:
    ax.set_visible(False)
fig.suptitle(f"Onde importa cada uma das {len(SEL_VARS)} variáveis selecionadas — piora do RMSE por estação "
             "(escala própria de cada mapa; ordenadas por grupo e importância)", fontsize=13)
plt.show()
conc = pd.DataFrame({"média": espacial.mean(axis=1), "máx": espacial.max(axis=1),
                     "onde mais": espacial.idxmax(axis=1), "estações > 0,01": (espacial > 0.01).sum(axis=1)}
                    ).join(por_var.set_index("variavel").grupo).sort_values("média", ascending=False)
conc["selecionada"] = conc.index.isin(SEL["variables"])
print(conc.round(4).to_string())
""")

md("## 5. Temporal — por trimestre\n\nMesma medição por estação do ano, **todas as 185 colunas** da lista (as que ficam fora do modelo aparecem sem valor).")

py("""
ordem_h = pd.concat([col_tab[col_tab.grupo == g].sort_values(["uso", "d_rmse"], ascending=[False, False]) for g in GCOR])
fig, ax = plt.subplots(figsize=(8.5, len(ordem_h) * 0.19 + 2))
sns.heatmap(ordem_h[list(SEASONS)], annot=True, fmt=".3f", annot_kws={"fontsize": 6}, cmap="YlOrRd", center=0,
            linewidths=0.3, ax=ax, cbar_kws={"label": "piora do RMSE (m/s)", "shrink": 0.4}, mask=ordem_h[list(SEASONS)].isna())
ax.set_yticklabels([f"{c}  [{g[-1]}]" + ("  (fora)" if u == "fora" else "") for c, g, u in
                    zip(ordem_h.index, ordem_h.grupo, ordem_h.uso)], rotation=0, fontsize=6.5)
ax.set_title("Importância por coluna × trimestre, as 185 colunas da lista  [n = grupo; (fora) = não entra no modelo]")
ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.show()
""")

md("""
## 6. Temporal — as séries onde a variável decide

Para a variável de topo de cada grupo, a previsão do modelo íntegro contra a do mesmo
modelo com a variável embaralhada, nos 2 episódios de teste em que ela mais mudou a previsão.
A linha vermelha é a **rajada** do ERA5 (`gust10fg` na hora do pico).
""")

py("""
rng = np.random.default_rng(42)
# modelo só com a BASE (sem nenhum grupo), um por trimestre, nas mesmas linhas
modelos_base = {}
for s in SEASONS:
    tr = treino[treino.season == s]
    modelos_base[s] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[BASE_COLS], tr[ALVO])
modelos_sel = {}
for s in SEASONS:
    tr = treino[treino.season == s]
    modelos_sel[s] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[SEL_COLS], tr[ALVO])
series = {}
for var in TOP:
    partes = []
    for s in SEASONS:
        x, y, est, te = dados_season[s]
        p_ok = np.asarray(modelos[s].predict(x), float)
        emb = x.copy()
        emb[blocos[var]] = x[blocos[var]].to_numpy()[rng.permutation(len(x))]
        p_sem = np.asarray(modelos[s].predict(emb), float)
        p_base = np.asarray(modelos_base[s].predict(te[BASE_COLS]), float)
        p_sel = np.asarray(modelos_sel[s].predict(te[SEL_COLS]), float)
        partes.append(te.assign(pred=p_ok, pred_sem=p_sem, pred_base=p_base, pred_sel=p_sel, contrib=np.abs(p_ok - p_sem))
                      [["estacao", "time", "season", ALVO, REF, "pred", "pred_sem", "pred_base", "pred_sel", "contrib"]])
    series[var] = pd.concat(partes, ignore_index=True)

fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
for lin, var in zip(axes, TOP):
    for ax, (_, ev) in zip(lin, series[var].nlargest(2, "contrib").iterrows()):
        g = series[var][(series[var].estacao == ev.estacao) & (series[var].time.between(
            ev.time - pd.Timedelta(days=10), ev.time + pd.Timedelta(days=10)))].sort_values("time")
        ax.plot(g.time, g[ALVO], "o-", color="#111", ms=4, lw=1.2, label="observado", zorder=4)
        ax.plot(g.time, g[REF], "s--", color="#c44", ms=3, lw=1, alpha=0.8, label="ERA5 (rajada)")
        ax.plot(g.time, g.pred, "^-", color="#2a7ab0", ms=4, lw=1.2, label="modelo")
        ax.plot(g.time, g.pred_sem, "v:", color="#8a6fb0", ms=4, lw=1.3, label=f"sem {var}")
        ax.plot(g.time, g.pred_base, "d-.", color="#e08a00", ms=4, lw=1.3, label="só BASE (sem grupos)")
        ax.plot(g.time, g.pred_sel, "*-", color="#1b9e77", ms=6, lw=1.3, label="seleção (12 var.)")
        ax.set_title(f"{var} [{grupo_var[var]}] · {ev.estacao} {ev.time:%d/%m/%Y}: contribuição {ev.contrib:.1f} m/s", fontsize=9.5, loc="left")
        ax.tick_params(axis="x", labelsize=8)
axes[0, 0].legend(fontsize=8, ncol=3)
fig.suptitle("Episódios em que a variável de topo de cada grupo decidiu a previsão\\n"
             "laranja = só a BASE antiga, sem nenhum grupo; verde = seleção de 12 variáveis", fontsize=13)
plt.show()

fx = pd.cut(series[TOP[0]][ALVO], [0, 8, 12, 16, 100], labels=["< 8", "8–12", "12–16", "> 16 m/s"])
tab = pd.DataFrame({f"{v} [{grupo_var[v][-1]}]": series[v].groupby(fx, observed=True).contrib.mean() for v in TOP})
print("contribuição média |Δpredição| (m/s) por faixa da rajada observada:"); print(tab.round(3).to_string())
sb = series[TOP[0]]
rm = lambda c: float(np.sqrt(np.mean((sb[c] - sb[ALVO]) ** 2)))  # noqa: E731
print(f"\\nRMSE no teste: ERA5 {rm(REF):.3f} | só BASE {rm('pred_base'):.3f} | seleção {rm('pred_sel'):.3f} | modelo com grupos {rm('pred'):.3f}")
""")

md("""
## 7. Cada grupo isolado — qual vale continuar?

Aqui o grupo entra **sozinho**: um modelo por trimestre treinado só com as colunas dele
(e, como referência, a **BASE antiga sozinha**: as 12 variáveis do ERA5 horário por estação,
lidas na hora do pico, sem nenhum grupo) e a **seleção de 12 variáveis** (escolhida na validação, cruzando os grupos)
(LightGBM, mesmos treino e teste). Dois pisos para ler o resultado: **só ruído** (o que o
modelo faz sem informação nenhuma, quase a média do trimestre) e a **rajada ERA5 bruta**
(o que já temos de graça). Um grupo só merece continuar se, sozinho, ganha do piso e se
aproxima ou supera o ERA5. IC de 95 % por bootstrap em blocos (ano, mês), 1000 sorteios.
""")

py("""
from sklearn.metrics import r2_score

def isolado(cols, s):
    tr, te = treino[treino.season == s], teste[teste.season == s]
    m = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[cols], tr[ALVO])
    return te.assign(pred=m.predict(te[cols]))

ISOL = {f"{g} sozinho": [c for c in FEATURES if GRUPO_DE.get(c) == f"grupo{g[-1]}"] for g in ROTULO.values()}
ISOL["só ruído (piso)"] = ["ctrl_noise"]
ISOL["BASE antiga sozinha"] = BASE_COLS
ISOL["seleção (12 var.)"] = SEL_COLS
saidas = {k: pd.concat([isolado(v, s) for s in SEASONS], ignore_index=True) for k, v in ISOL.items()}
base_te = saidas["só ruído (piso)"]
saidas["ERA5 (rajada bruta)"] = base_te.assign(pred=base_te[REF])

bloco = (base_te.year * 100 + base_te.month).factorize()[0]
nb_ = bloco.max() + 1
rng = np.random.default_rng(42)
pesos = rng.multinomial(len(base_te), np.ones(nb_) / nb_, size=1000)   # blocos sorteados
def rmse_boot(d):
    e2 = (d.pred.to_numpy() - d[ALVO].to_numpy()) ** 2
    soma = np.bincount(bloco, e2, nb_); n = np.bincount(bloco, minlength=nb_)
    return np.sqrt((pesos @ soma) / (pesos @ n))

y = base_te[ALVO].to_numpy(); p90 = y >= np.quantile(y, 0.9)
tab = []
for k, d in saidas.items():
    b = rmse_boot(d); lo, hi = np.quantile(b, [0.025, 0.975])
    e = d.pred.to_numpy() - y
    tab.append({"modelo": k, "n_colunas": len(ISOL.get(k, [])), "RMSE": np.sqrt((e**2).mean()), "IC_lo": lo, "IC_hi": hi,
                "RMSE ≥P90": np.sqrt((e[p90]**2).mean()), "R²": r2_score(y, d.pred)})
tab = pd.DataFrame(tab).set_index("modelo")
piso, era = tab.loc["só ruído (piso)", "RMSE"], tab.loc["ERA5 (rajada bruta)", "RMSE"]
tab["ganho sobre o piso"] = piso - tab.RMSE
tab["ganho sobre o ERA5"] = era - tab.RMSE

ordem = ["só ruído (piso)", "BASE antiga sozinha"] + [f"{g} sozinho" for g in ROTULO.values()] + ["seleção (12 var.)", "ERA5 (rajada bruta)"]
cor = {"só ruído (piso)": "#999", "BASE antiga sozinha": "#b0824a", "seleção (12 var.)": "#1b9e77", "ERA5 (rajada bruta)": "#c44", **{f"{g} sozinho": CORES[g] for g in ROTULO.values()}}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
for a, col, tit in ((ax[0], "RMSE", "RMSE (todos os dias)"), (ax[1], "RMSE ≥P90", "RMSE, dias ≥ P90")):
    v = tab.loc[ordem, col]
    erro = [[v[k] - tab.loc[k, "IC_lo"] for k in ordem], [tab.loc[k, "IC_hi"] - v[k] for k in ordem]] if col == "RMSE" else None
    a.barh(ordem[::-1], v[::-1], color=[cor[k] for k in ordem[::-1]], edgecolor="black", linewidth=0.5,
           xerr=None if erro is None else [x[::-1] for x in erro], capsize=3)
    for i, k in enumerate(ordem[::-1]):
        a.text(v[k] + 0.03, i, f"{v[k]:.2f}", va="center", fontsize=9)
    a.axvline(era if col == "RMSE" else tab.loc["ERA5 (rajada bruta)", col], color="#c44", ls="--", lw=1)
    a.set_title(tit); a.set_xlabel("m/s")
fig.suptitle("Cada grupo isolado contra o piso (só ruído) e o ERA5 — menor é melhor", fontsize=13)
plt.show()
print(tab.loc[ordem].round(3).to_string())
""")

md("""
**Como decidir.** Continua o grupo que, **sozinho**, tem RMSE bem abaixo do piso e perto
ou abaixo do ERA5. Um grupo que sozinho fica no piso não tem informação própria para o
alvo; só pode ajudar em combinação, e isso o estudo pago já testou (somar os grupos 2 e 3
ao grupo 1 não acrescenta). O grupo 4 sozinho pode parecer bom por identificar a estação
(as estáticas variam por estação), não por física.
""")

md("""
## 8. Todas as variáveis: quais manter?

Regra, aplicada a cada variável (os números vêm das seções anteriores):

1. **Piso do ruído:** uma coluna aleatória (`ctrl_noise`) é adicionada ao mesmo modelo e
   embaralhada como as demais. Seu efeito, o maior entre os 4 trimestres, é o que a
   permutação produz **sem informação nenhuma**.
2. **manter** = piora média do RMSE acima do piso **e** positiva em pelo menos 3 dos 4 trimestres.
3. **inconclusiva** = acima do piso, mas positiva em 2 trimestres ou menos (sinal de um só período).
4. **descartar** = não passa do piso.

A permutação mede **uso**, não ganho fora da amostra. Por isso a última coluna traz o
veredito do estudo pago para o grupo: só o grupo 1 acrescentou informação. Variável
"mantida" num grupo que não acrescenta é uso, não ganho (ex.: `anor`, que identifica a estação).
""")

py("""
# piso do ruído: mesmo modelo, com uma coluna aleatória a mais
pisos = []
for s in SEASONS:
    tr, te = treino[treino.season == s], teste[teste.season == s]
    cols = FEATURES + ["ctrl_noise"]
    m = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[cols], tr[ALVO])
    r = permutacao_por_bloco(m, te[cols], te[ALVO].to_numpy(float), te.estacao.to_numpy(),
                             {"ctrl_noise": ["ctrl_noise"]}, n_repeticoes=5, seed=42)
    pisos.append(r.d_rmse.mean())
PISO = max(pisos)
print(f"piso do ruído (maior entre trimestres): {PISO:.5f} m/s | por trimestre: {np.round(pisos, 5)}")

por_s = (imp[imp.grupo.isin(CORES)].groupby(["variavel", "season"]).d_rmse.mean().unstack()[list(SEASONS)])
dec = por_var.set_index("variavel")[["grupo", "d_rmse", "d_rmse_p90"]].join(por_s)
dec["trimestres > 0"] = (por_s > 0).sum(axis=1)
dec["decisão"] = np.select(
    [(dec.d_rmse > PISO) & (dec["trimestres > 0"] >= 3), (dec.d_rmse > PISO)],
    ["manter", "inconclusiva"], default="descartar")
VEREDITO_GRUPO = {"grupo 1": "acrescenta", "grupo 2": "não acrescenta", "grupo 3": "não acrescenta",
                  "grupo 4": "não acrescenta (identifica a estação)"}
dec["grupo no estudo pago"] = dec.grupo.map(VEREDITO_GRUPO)
dec["colunas"] = [len(blocos[v]) for v in dec.index]
dec = dec.sort_values(["grupo", "d_rmse"], ascending=[True, False])

# variáveis que já ficaram de fora antes da permutação
fora = (todas[(todas.uso == "fora") & (todas.grupo != "BASE")]
        .assign(v=lambda d: d.coluna.map(variavel)).groupby(["grupo", "v"]).size())
print("\\n=== DECISÃO POR VARIÁVEL ===")
print(dec[["grupo", "colunas", "d_rmse", "d_rmse_p90", *SEASONS, "trimestres > 0", "decisão",
           "grupo no estudo pago"]].round(4).to_string())
print("\\nResumo:")
print(dec.groupby(["grupo", "decisão"]).size().unstack(fill_value=0).to_string())
print("\\nMANTER (por permutação):")
for g, d in dec[dec.decisão == "manter"].groupby("grupo"):
    print(f"  {g}: {', '.join(d.index)}")
print("\\nDESCARTAR:")
for g, d in dec[dec.decisão == "descartar"].groupby("grupo"):
    print(f"  {g}: {', '.join(d.index)}")
print("\\nColunas fora do modelo (cin, identificadores, *_n): ver a tabela da seção 1 e a figura acima.")

dec_fam = dec.decisão.to_dict()
col_tab["decisão (da variável)"] = [dec_fam.get(f, "") if u == "usada" else "fora do modelo"
                                    for f, u in zip(col_tab["família"], col_tab.uso)]
COR_D = {"manter": "#2a7ab0", "inconclusiva": "#e0a040", "descartar": "#bbbbbb", "fora do modelo": "#ffffff"}
d = pd.concat([col_tab[col_tab.grupo == g].sort_values(["uso", "d_rmse"], ascending=[True, True]) for g in list(GCOR)[::-1]])
fig, ax = plt.subplots(figsize=(10.5, len(d) * 0.19 + 2))
ax.barh(np.arange(len(d)), d.d_rmse.fillna(0), color=[COR_D[x] for x in d["decisão (da variável)"]],
        edgecolor="black", linewidth=0.3, hatch=["///" if x == "fora do modelo" else "" for x in d["decisão (da variável)"]])
ax.set_yticks(np.arange(len(d)))
ax.set_yticklabels([f"{c}  [{g[-1]}]" for c, g in zip(d.index, d.grupo)], fontsize=6.5)
ax.axvline(PISO, color="#c44", ls="--", lw=1.2, label=f"piso do ruído ({PISO:.4f})")
ax.margins(y=0.004)
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_xlabel("piora do RMSE ao embaralhar a coluna (m/s, escala symlog)")
ax.set_title("Todas as 185 colunas: cor = decisão da VARIÁVEL a que pertencem (azul manter, laranja inconclusiva, cinza descartar, hachurado = fora do modelo)", fontsize=9)
ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
""")

md("""
## 9. Modelo só com as variáveis mantidas

**Cuidado com circularidade.** A seleção da seção 8 usou o conjunto de **teste** (a
permutação rodou nas linhas de teste). Testar nele um modelo com essas variáveis
favoreceria a escolha. Por isso a seleção é **refeita na validação** (linhas de validação
do estudo, que não entram no treino nem no teste, mesma regra e mesmo piso de ruído) e o
modelo é avaliado no teste. A versão escolhida no teste aparece só como referência,
marcada como vazada.

Comparamos, no mesmo teste e com o mesmo modelo (LightGBM, seed 42; com os parâmetros padrão ele é determinístico, então variar a seed não mudaria nada): o conjunto completo,
só o grupo 1 (a base do estudo), as variáveis mantidas e o ERA5. O delta contra o completo
é pareado, com bootstrap em blocos (ano, mês).
""")

py("""
val = treino_total = pd.read_parquet(DADOS / "sample_full.parquet")
val = val[val._split == "val"]
print(f"validação: {len(val)} linhas")

# seleção na VALIDAÇÃO: mesma regra da seção 8
imp_v, pisos_v = [], []
for s in SEASONS:
    tr, va = treino[treino.season == s], val[val.season == s]
    cols = FEATURES + ["ctrl_noise"]
    m = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[cols], tr[ALVO])
    r = permutacao_por_bloco(m, va[cols], va[ALVO].to_numpy(float), va.estacao.to_numpy(),
                             {**blocos, "ctrl_noise": ["ctrl_noise"]}, n_repeticoes=5, seed=42)
    r["season"] = s
    imp_v.append(r)
imp_v = pd.concat(imp_v, ignore_index=True)
piso_v = imp_v[imp_v.variavel == "ctrl_noise"].groupby("season").d_rmse.mean().max()
pv = imp_v[imp_v.variavel != "ctrl_noise"].groupby(["variavel", "season"]).d_rmse.mean().unstack()[list(SEASONS)]
dec_v = pd.DataFrame({"d_rmse": pv.mean(axis=1), "trimestres > 0": (pv > 0).sum(axis=1)})
mantidas_val = dec_v[(dec_v.d_rmse > piso_v) & (dec_v["trimestres > 0"] >= 3)].index.tolist()
mantidas_teste = dec[dec.decisão == "manter"].index.tolist()
print(f"piso do ruído na validação: {piso_v:.5f} m/s")
print("seleção refeita aqui == JSON congelado usado nas figuras anteriores:",
      set(mantidas_val) == set(SEL["variables"]))
print(f"mantidas pela VALIDAÇÃO: {len(mantidas_val)} variáveis | pelo TESTE (seção 8): {len(mantidas_teste)} | "
      f"em comum: {len(set(mantidas_val) & set(mantidas_teste))}")
print("só na validação:", sorted(set(mantidas_val) - set(mantidas_teste)))
print("só no teste:    ", sorted(set(mantidas_teste) - set(mantidas_val)))
print("\\nmantidas pela validação, por grupo:")
for g in CORES:
    print(f"  {g}: {', '.join(v for v in mantidas_val if grupo_var[v] == g)}")
""")

py("""
def colunas(vars_):
    return [c for v in vars_ for c in blocos[v]]

BRACOS = {
    "completo (167 colunas)": FEATURES,
    "só grupo 1 (base do estudo)": [c for c in FEATURES if GRUPO_DE.get(c) == "grupo1"],
    f"mantidas, escolhidas na VALIDAÇÃO ({len(mantidas_val)} var.)": colunas(mantidas_val),
    f"mantidas, escolhidas no TESTE — vazada ({len(mantidas_teste)} var.)": colunas(mantidas_teste),
}
SEEDS = (42,)
preds = {k: [] for k in BRACOS}
for k, cols in BRACOS.items():
    for seed in SEEDS:
        partes = []
        for s in SEASONS:
            tr, te = treino[treino.season == s], teste[teste.season == s]
            m = LGBMRegressor(random_state=seed, verbose=-1, n_jobs=4).fit(tr[cols], tr[ALVO])
            partes.append(m.predict(te[cols]))
        preds[k].append(np.concatenate(partes))

y_all = base_te[ALVO].to_numpy()
ordem_te = base_te   # mesma ordem (estação-dia por trimestre) das seções anteriores
res = {k: ordem_te.assign(pred=p[0]) for k, p in preds.items()}          # seed 42 para o bootstrap
res["ERA5 (rajada bruta)"] = ordem_te.assign(pred=ordem_te[REF])

full = "completo (167 colunas)"
b_full = rmse_boot(res[full])
linhas = []
for k, d in res.items():
    b = rmse_boot(d)
    e = d.pred.to_numpy() - y_all
    por_seed = [np.sqrt(np.mean((p - y_all) ** 2)) for p in preds[k]] if k in preds else [np.nan]
    delta = b - b_full
    linhas.append({"modelo": k, "RMSE": np.mean(por_seed),
                   "RMSE ≥P90": np.sqrt((e[p90] ** 2).mean()), "R²": r2_score(y_all, d.pred),
                   "Δ vs completo": delta.mean(), "Δ IC_lo": np.quantile(delta, 0.025),
                   "Δ IC_hi": np.quantile(delta, 0.975)})
cmp_ = pd.DataFrame(linhas).set_index("modelo")
print(cmp_.round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
nomes = list(cmp_.index)
cores = ["#555", CORES["grupo 1"], "#5aa469", "#e0a040", "#c44"]
ax.barh(nomes[::-1], cmp_.RMSE[::-1], color=cores[::-1], edgecolor="black", linewidth=0.5,
        )
for i, k in enumerate(nomes[::-1]):
    ax.text(cmp_.loc[k, "RMSE"] + 0.01, i, f"{cmp_.loc[k, 'RMSE']:.3f}", va="center", fontsize=9)
ax.set_xlim(2.0, cmp_.RMSE.max() * 1.08)
ax.set_xlabel("RMSE no teste (m/s)")
ax.set_title("Variáveis mantidas contra o conjunto completo, só o grupo 1 e o ERA5")
plt.show()
""")

md("""
**Como ler.** O que interessa é o **Δ contra o completo** (negativo = o conjunto menor é
melhor), com seu IC. Se o IC do Δ da linha "validação" contém zero, as variáveis
descartadas não faziam falta, e o conjunto menor serve; se for positivo e excluir zero,
perdemos informação ao cortar. Compare a linha "VALIDAÇÃO" com a "TESTE (vazada)": a
diferença entre as duas é o tamanho do viés de selecionar olhando para o teste. Um modelo
(LightGBM, seed 42): serve para decidir o corte, não como prova.
""")

md("""
## 10. O modelo treinado no Modal: seleção contra completo e grupo 1

Aqui não é mais o LightGBM do notebook: é o **arm `sel__val12` treinado no Modal**, no
mesmo estudo e com a mesma população dos demais arms. 5 seeds × 4 trimestres × os 5
melhores modelos de cada trimestre (10 modelos distintos no total), as 12 variáveis
escolhidas na validação (58 colunas). Os resíduos por linha de `sel__val12`, `full` e
`base` vêm do volume, e cada modelo é comparado nas **mesmas linhas** nos três arms.

Estimador: RMSE por (seed, modelo), média entre eles, e Δ pareado com bootstrap em blocos
(ano, mês), 2000 sorteios, o mesmo sorteio nos dois lados. É uma reimplementação enxuta do
`compute_effects` do estudo (sem correção de múltiplas hipóteses). Δ > 0 significa que o
**segundo** arm é melhor.
""")

py("""
UNIDADES = RAIZ / "artifacts/feature_study/cluster3_groups_modal/units"
TAGS = ["full", "full_s43", "full_s44", "full_s45", "full_s46"]
ARMS_M = ["base", "full", "sel__val12"]
tm = pd.read_parquet(RAIZ / "artifacts/feature_study/cluster3_groups_modal/data/test.parquet")
tm = tm.sort_values("row_id").reset_index(drop=True)
chave = (tm.year * 100 + tm.month).factorize()[0]
nb_m = chave.max() + 1
bloco_de = pd.Series(chave, index=tm.row_id.to_numpy())
era_m = pd.Series((tm[REF] - tm[ALVO]).to_numpy(), index=tm.row_id.to_numpy())
rng_m = np.random.default_rng(42)
W = rng_m.multinomial(len(tm), np.ones(nb_m) / nb_m, size=2000).astype(float)   # (2000, blocos)


def rmse_draws(e2, b):
    soma = np.bincount(b, e2, nb_m); n = np.bincount(b, minlength=nb_m)
    return np.sqrt((W @ soma) / (W @ n)), np.sqrt(soma.sum() / n.sum())


# um bloco (tag, modelo): linhas em que os 3 arms têm resíduo finito, em todos os trimestres
casos = {}
for tag in TAGS:
    for s in SEASONS:
        r = {a: pd.read_parquet(UNIDADES / tag / f"resid__{s}__{a}.parquet").set_index("row_id") for a in ARMS_M}
        for modelo in r["sel__val12"].columns:
            if all(modelo in r[a] for a in ARMS_M):
                d = pd.concat({a: r[a][modelo] for a in ARMS_M}, axis=1).dropna()
                casos.setdefault((tag, modelo), []).append(d)
casos = {k: pd.concat(v) for k, v in casos.items()}
print(f"{len(casos)} pares (seed, modelo) | modelos: {sorted({m for _, m in casos})}")

pontos = {a: [] for a in ARMS_M + ["ERA5"]}
draws = {a: [] for a in ARMS_M + ["ERA5"]}
for (tag, modelo), d in casos.items():
    b = bloco_de.loc[d.index].to_numpy()
    for a in ARMS_M:
        dr, pt = rmse_draws(d[a].to_numpy() ** 2, b)
        draws[a].append(dr); pontos[a].append(pt)
    dr, pt = rmse_draws(era_m.loc[d.index].to_numpy() ** 2, b)
    draws["ERA5"].append(dr); pontos["ERA5"].append(pt)
D = {a: np.mean(v, axis=0) for a, v in draws.items()}
P = {a: float(np.mean(v)) for a, v in pontos.items()}


def delta(pior, melhor):
    d_ = D[pior] - D[melhor]
    return P[pior] - P[melhor], *np.quantile(d_, [0.025, 0.975])


SESOI_M = 0.01 * P["base"]
linhas = [("ERA5 → grupo 1 (base)", "ERA5", "base"), ("grupo 1 (base) → seleção (12 var.)", "base", "sel__val12"),
          ("seleção (12 var.) → completo (167)", "sel__val12", "full"), ("grupo 1 (base) → completo (167)", "base", "full")]
tab_m = pd.DataFrame([{"comparação": n, "ganho de RMSE": delta(a, b)[0], "IC_lo": delta(a, b)[1], "IC_hi": delta(a, b)[2]}
                      for n, a, b in linhas]).set_index("comparação")
tab_m["veredito"] = np.where(tab_m.IC_lo > SESOI_M, "acrescenta",
                     np.where((tab_m.IC_hi < SESOI_M) & (tab_m.IC_lo > -SESOI_M), "sem informação",
                     np.where(tab_m.IC_hi < -SESOI_M, "prejudica", "inconclusivo")))
print("RMSE médio (10 modelos × 5 seeds, mesmas linhas):")
print(pd.Series({"ERA5 (rajada)": P["ERA5"], "grupo 1 (base)": P["base"],
                 "seleção (12 var., 58 col.)": P["sel__val12"], "completo (167 col.)": P["full"]}).round(4).to_string())
print(f"\\nSESOI = {SESOI_M:.4f} m/s (1 % do RMSE da base)")
print(tab_m.round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
nomes_m = ["ERA5 (rajada)", "grupo 1 (base)", "seleção (12 var.)", "completo (167)"]
vals = [P["ERA5"], P["base"], P["sel__val12"], P["full"]]
ax[0].barh(nomes_m[::-1], vals[::-1], color=["#5aa469", "#e0a040", CORES["grupo 1"], "#c44"], edgecolor="black", linewidth=0.5)
for i, v in enumerate(vals[::-1]):
    ax[0].text(v + 0.01, i, f"{v:.3f}", va="center")
ax[0].set_xlim(2.0, max(vals) * 1.07); ax[0].set_xlabel("RMSE (m/s)"); ax[0].set_title("Desempenho (menor é melhor)")
y_ = np.arange(len(tab_m))[::-1]
cv = {"acrescenta": "#2a7ab0", "sem informação": "#bbbbbb", "inconclusivo": "#e0a040", "prejudica": "#c44"}
for yi, (n, r) in zip(y_, tab_m.iterrows()):
    ax[1].errorbar(r["ganho de RMSE"], yi, xerr=[[r["ganho de RMSE"] - r.IC_lo], [r.IC_hi - r["ganho de RMSE"]]],
                   fmt="o", color=cv[r.veredito], capsize=4, lw=2, ms=8)
    ax[1].text(r.IC_hi + 0.01, yi, f"{r['ganho de RMSE']:+.3f}  ({r.veredito})", va="center", fontsize=9)
ax[1].axvspan(-SESOI_M, SESOI_M, color="#ddd", alpha=0.7); ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_yticks(y_); ax[1].set_yticklabels(tab_m.index); ax[1].set_xlim(right=0.62)
ax[1].set_xlabel("ganho de RMSE (m/s); faixa cinza = ±SESOI"); ax[1].set_title("Ganho entre arms, IC 95 %")
fig.suptitle("Modelo treinado no Modal: seleção de 12 variáveis contra completo e grupo 1", fontsize=12.5)
plt.show()
""")

md("## 11. Veredito por grupo")

py("""
teto = imp[imp.variavel == "CTRL ruído"].d_rmse.mean() if (imp.variavel == "CTRL ruído").any() else 0.0
res = por_var.groupby("grupo").agg(variáveis=("variavel", "size"), soma=("d_rmse", "sum"),
                                   mediana=("d_rmse", "median"), máx=("d_rmse", "max"),
                                   top=("variavel", "first")).round(4)
print(res.to_string())
print()
print("variáveis com piora < 0,002 m/s (uso praticamente nulo):")
print(por_var[por_var.d_rmse < 0.002].groupby("grupo").variavel.apply(lambda s: ", ".join(s)).to_string())
""")

md("""
### Limites

- **Permutação mede uso, não ganho fora da amostra.** O estudo pago diz que só o grupo 1
  acrescenta; um grupo pode ter uso alto e ganho nulo (as estáticas do grupo 4 identificam
  a estação).
- **Colinearidade:** variáveis irmãs dividem o crédito (ex.: `w10`, `gust10fg`, `w100`);
  o bloco **inteiro do grupo** é a leitura robusta.
- **Um modelo (LightGBM) por trimestre**, não o campeão de cada trimestre, para que as
  variáveis sejam comparáveis. A incerteza mostrada é entre repetições, não bootstrap em blocos.
- Uma linha por dia na hora do pico: a análise por faixa do dia da versão anterior não
  existe nesta estrutura e foi retirada.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
