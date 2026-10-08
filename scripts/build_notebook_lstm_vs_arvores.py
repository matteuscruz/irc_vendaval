"""Gera `notebooks/feature_study_grupos/atual/lstm_vs_arvores.ipynb`.

A LSTM roda nos MESMOS arms e linhas do estudo tabular. O notebook compara os dois: os efeitos dos grupos, o
erro por arm e a importância por grupo e por coluna — a pergunta é se os dois modelos concordam sobre quais
features importam. Lê o que `run_feature_study_local.py --stage lstm/aggregate-lstm` e o estudo do Modal gravaram.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/lstm_vs_arvores.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# LSTM contra árvores: as features importantes são as mesmas?

O estudo de features mede, com árvores, quais grupos de variáveis acrescentam informação. Aqui o **mesmo
estudo** (mesmas linhas, mesmos arms, mesma partição, mesmo bootstrap pareado em blocos) roda com uma **LSTM**
que vê as últimas 24 h até a hora do pico, para checar se a conclusão vale fora das árvores.

| | árvores | LSTM |
|---|---|---|
| modelos | os 5 melhores por trimestre (10 distintos), 5 seeds | 1 modelo, **1 seed (42)** |
| entrada | uma linha na hora do pico | janela de 24 h (grupos 1–3 como sequência) |
| fonte | `cluster3_groups_modal` | `cluster3_groups` (rodada local) |

**Leia junto das ressalvas do fim:** a LSTM tem uma seed e hiperparâmetros sem ajuste para este estudo.
""")

py("""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from src.feature_study.diagnostics.group_importance import permutacao_por_bloco

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
ARV = RAIZ / "artifacts/feature_study/cluster3_groups_modal"
LOC = RAIZ / "artifacts/feature_study/cluster3_groups"
SEASONS = ("DJF", "MAM", "JJA", "SON")
COR = {"LSTM": "#c44", "árvores": "#2a7ab0"}

ef = lambda d: d[(d.family == "all") & (d.metric == "rmse")].set_index("comparison")      # noqa: E731
E_L = ef(pd.read_csv(LOC / "summary/lstm/effects.csv"))
E_T = ef(pd.read_csv(ARV / "summary/main/effects.csv"))
print(len(E_L), "comparações da LSTM |", len(E_T), "das árvores")
""")

md("""
## 1. Os efeitos dos grupos, lado a lado

Ganho de RMSE (m/s): **positivo = o segundo arm é melhor**. A faixa cinza é o tamanho mínimo relevante (SESOI).
Intervalos: bootstrap pareado em blocos (ano, mês), que mede a incerteza do **teste**, não a variância de treino
da LSTM (uma seed).
""")

py("""
ordem = ["drop_grp__grupo1", "add_grp__grupo2", "drop_grp__grupo2", "add_grp__grupo3", "drop_grp__grupo3",
         "add_grp__grupo4", "drop_grp__grupo4", "full_vs_base", "ctrl__noise", "ctrl__perm_static", "real_vs_perm_static"]
rot = {"drop_grp__grupo1": "remover grupo 1", "add_grp__grupo2": "somar grupo 2", "drop_grp__grupo2": "remover grupo 2",
       "add_grp__grupo3": "somar grupo 3", "drop_grp__grupo3": "remover grupo 3", "add_grp__grupo4": "somar grupo 4",
       "drop_grp__grupo4": "remover grupo 4", "full_vs_base": "completo contra base",
       "ctrl__noise": "controle: ruído", "ctrl__perm_static": "controle: estáticas permutadas",
       "real_vs_perm_static": "estática real contra permutada"}
sesoi = float(E_L.sesoi.iloc[0])
y = np.arange(len(ordem))[::-1]
fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
for off, (nome, E) in zip((0.17, -0.17), (("LSTM", E_L), ("árvores", E_T))):
    for yi, k in zip(y, ordem):
        r = E.loc[k]
        ax.errorbar(r.effect, yi + off, xerr=[[r.effect - r.ci_lo], [r.ci_hi - r.effect]], fmt="o", ms=6,
                    color=COR[nome], capsize=3, lw=1.8, label=nome if k == ordem[0] else None)
ax.axvspan(-sesoi, sesoi, color="#ddd", alpha=0.7, zorder=0, label=f"±SESOI ({sesoi:.3f})")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([rot[k] for k in ordem])
ax.set_xlabel("ganho de RMSE (m/s)")
ax.set_title("Figura 1: efeito de cada grupo, LSTM contra árvores")
ax.legend(loc="lower left")
plt.show()

tab = pd.DataFrame({"LSTM": E_L.loc[ordem, "effect"], "veredito LSTM": E_L.loc[ordem, "verdict"],
                    "árvores": E_T.loc[ordem, "effect"], "veredito árvores": E_T.loc[ordem, "verdict"]})
tab.index = [rot[k] for k in tab.index]
print(tab.round(3).to_string())
""")

md("""
**Leitura.** Os dois concordam em três coisas: **só o grupo 1 acrescenta informação**, o **grupo 3 não traz nada**
e o **controle de ruído é zero**. Divergem no resto: somar o grupo 2 ou o 4 (e as estáticas permutadas) **piora** a
LSTM, e as árvores ficam indiferentes. É o padrão de sobreajuste: árvore ignora coluna inútil, a rede a usa e erra
mais no teste.
""")

py("""
def rmse_por_arm(pasta, tags):
    partes = []
    for t in tags:
        for p in glob.glob(str(pasta / "units" / t / "metrics__*__*.parquet")):
            m = pd.read_parquet(p)
            partes.append(m[m.split == "test"][["arm", "season", "model", "RMSE"]])
    d = pd.concat(partes)
    return d.groupby(["arm", "season"]).RMSE.mean().groupby("arm").mean()          # média de modelos/seeds e depois dos trimestres

r_l = rmse_por_arm(LOC, ["full_lstm"])
r_t = rmse_por_arm(ARV, ["full", "full_s43", "full_s44", "full_s45", "full_s46"])
era5 = float(pd.read_csv(ARV / "robustez/desempenho_cauda.csv").set_index("arm").loc["ERA5 (rajada bruta)", "rmse"])
arms = [a for a in r_l.sort_values().index if a in r_t.index]
x = np.arange(len(arms))
fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
ax.bar(x - 0.2, r_t[arms], 0.4, color=COR["árvores"], label="árvores (média de 10 modelos × 5 seeds)")
ax.bar(x + 0.2, r_l[arms], 0.4, color=COR["LSTM"], label="LSTM (seed 42)")
ax.axhline(era5, color="#555", ls="--", lw=1.2, label=f"ERA5 (rajada): {era5:.2f}")
ax.set_xticks(x); ax.set_xticklabels(arms, rotation=35, ha="right")
ax.set_ylim(2.0, max(r_l.max(), r_t.max()) * 1.05); ax.set_ylabel("RMSE no teste (m/s)")
ax.set_title("Figura 2: erro por arm. A LSTM empata com as árvores no `base` e piora ao somar grupos")
ax.legend()
plt.show()
print(pd.DataFrame({"árvores": r_t[arms], "LSTM": r_l[arms]}).round(3).to_string())
""")

md("""
## 2. Importância por grupo

Permutação: quanto o RMSE do **teste** piora ao embaralhar o grupo inteiro, entre linhas. Na LSTM é o arm `full`
(a mesma permutação nos 24 passos); nas árvores, um LightGBM por trimestre treinado aqui, também no `full`. Mede
**uso**, não ganho fora da amostra: um modelo pode usar muito um grupo que não ajuda a generalizar.
""")

py("""
DADOS = LOC / "data"
meta = json.loads((DADOS / "meta.json").read_text())
GRUPO_DE = meta["group_of"]
arms_json = {a["name"]: a for a in json.loads((DADOS / "arms.json").read_text())}
FEATURES = arms_json["full"]["features"]
treino = pd.read_parquet(DADOS / "sample_full.parquet")
treino = treino[treino._split == "train"]
teste = pd.read_parquet(DADOS / "test.parquet")

# LSTM: importância gravada pelo worker (arm full)
imp_l = pd.concat([pd.read_parquet(p) for p in glob.glob(str(LOC / "units/full_lstm/importance__*__full.parquet"))])

# árvores: LightGBM por trimestre no arm full, mesmos blocos (grupo e coluna)
blocos_g = {}
for col in FEATURES:
    blocos_g.setdefault(GRUPO_DE[col], []).append(col)
blocos = {**{f"G::{g}": cols for g, cols in blocos_g.items()}, **{c: [c] for c in FEATURES}}
imp_t = []
for s in SEASONS:
    tr, te = treino[treino.season == s], teste[teste.season == s]
    m = LGBMRegressor(random_state=42, verbose=-1, n_jobs=4).fit(tr[FEATURES], tr["daily_wind_gust_max"])
    r = permutacao_por_bloco(m, te[FEATURES], te["daily_wind_gust_max"].to_numpy(float), te.estacao.to_numpy(), blocos,
                             n_repeticoes=3, seed=42)
    r["season"] = s
    imp_t.append(r)
imp_t = pd.concat(imp_t)
print("LightGBM:", imp_t.variavel.nunique(), "blocos | LSTM:", imp_l.nome.nunique(), "nomes")
""")

py("""
g_l = imp_l[imp_l.nivel == "grupo"].groupby("nome").d_rmse.mean()
g_t = imp_t[imp_t.variavel.str.startswith("G::")].assign(g=lambda d: d.variavel.str[3:]).groupby("g").d_rmse.mean()
grupos = ["grupo1", "grupo2", "grupo3", "grupo4"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
x = np.arange(4)
ax[0].bar(x - 0.2, g_t[grupos], 0.4, color=COR["árvores"], label="LightGBM")
ax[0].bar(x + 0.2, g_l[grupos], 0.4, color=COR["LSTM"], label="LSTM")
ax[0].set_xticks(x); ax[0].set_xticklabels(grupos); ax[0].set_ylabel("piora do RMSE (m/s)"); ax[0].legend()
ax[0].set_title("absoluto")
fa_t, fa_l = 100 * g_t[grupos] / g_t[grupos].sum(), 100 * g_l[grupos] / g_l[grupos].sum()
ax[1].bar(x - 0.2, fa_t, 0.4, color=COR["árvores"]); ax[1].bar(x + 0.2, fa_l, 0.4, color=COR["LSTM"])
for xi, (a, b) in enumerate(zip(fa_t, fa_l)):
    ax[1].text(xi - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=9); ax[1].text(xi + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels(grupos); ax[1].set_ylabel("% do total entre grupos"); ax[1].set_title("fatia relativa")
fig.suptitle("Figura 3: importância por grupo (permutação, arm full)", fontsize=12.5)
plt.show()
print(pd.DataFrame({"LightGBM": g_t[grupos], "LSTM": g_l[grupos]}).round(3).to_string())
""")

md("""
## 3. Importância por coluna: as mesmas colunas?

Cada coluna isolada, nos dois modelos. A correlação de postos (Spearman) mede se ordenam as 167 colunas do mesmo
jeito; a sobreposição do top 20, se escolhem as mesmas. Colunas irmãs (vizinhança `r75`/`r250`, defasagens) são
colineares e dividem o crédito, então a ordem exata entre vizinhas é frágil.
""")

py("""
col_l = imp_l[imp_l.nivel == "coluna"].groupby("nome").d_rmse.mean()
col_t = imp_t[~imp_t.variavel.str.startswith("G::")].groupby("variavel").d_rmse.mean()
comum = col_l.index.intersection(col_t.index)
rho, pval = spearmanr(col_l[comum], col_t[comum])
top_l, top_t = set(col_l.nlargest(20).index), set(col_t.nlargest(20).index)
print(f"Spearman entre LSTM e LightGBM nas {len(comum)} colunas: {rho:.2f} (p = {pval:.1e})")
print(f"top 20 em comum: {len(top_l & top_t)} de 20  ->  {sorted(top_l & top_t)}")

top = col_l.nlargest(20).index[::-1]
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
yy = np.arange(len(top))
ax.barh(yy + 0.2, col_l[top], 0.4, color=COR["LSTM"], label="LSTM")
ax.barh(yy - 0.2, col_t[top], 0.4, color=COR["árvores"], label="LightGBM")
ax.set_yticks(yy); ax.set_yticklabels([f"{c}  [{GRUPO_DE[c][-1]}]" for c in top], fontsize=8.5)
ax.set_xlabel("piora do RMSE ao embaralhar a coluna (m/s)"); ax.legend(loc="lower right")
ax.set_title("Figura 4: as 20 colunas mais importantes para a LSTM  [n = grupo]")
plt.show()

top_t20 = col_t.nlargest(20)
print("\\nTop 10 do LightGBM:"); print(pd.DataFrame({"LightGBM": top_t20.head(10), "LSTM": col_l.reindex(top_t20.head(10).index)}).round(3).to_string())
""")

md("""
## 4. Todos os modelos de ML juntos (execução do Modal: arms `base` e `full`)

Os 10 modelos tabulares do estudo de referência (os 5 melhores de cada trimestre, 5 seeds) e a LSTM treinada no
Modal (1 seed). Cada modelo só existe nos trimestres em que foi eleito, então **a comparação é por trimestre**:
em cada um há 5 modelos tabulares e a LSTM. O erro é o RMSE no teste; o efeito é `full` contra `base` por modelo,
com o mesmo bootstrap pareado em blocos. Gerado por `scripts/analise_todos_modelos.py`.
""")

py("""
TODOS = ARV / "todos_os_modelos"
ps = pd.read_csv(TODOS / "erro_por_modelo_arm_trimestre.csv")
ef_m = pd.read_csv(TODOS / "efeito_full_vs_base_por_modelo.csv")
era5_s = ps.drop_duplicates("season").set_index("season").era5_rmse.reindex(SEASONS)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, constrained_layout=True)
ordem_m = ps[ps.arm == "base"].groupby("model").RMSE.mean().sort_values().index
for ax, arm in zip(axes, ("base", "full")):
    pv = ps[ps.arm == arm].pivot_table(index="model", columns="season", values="RMSE").reindex(index=ordem_m, columns=list(SEASONS))
    pv.loc["ERA5 (rajada bruta)"] = era5_s
    im = ax.imshow(pv.to_numpy(float), cmap="YlOrRd", aspect="auto", vmin=2.05, vmax=2.95)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.iloc[i, j]
            ax.text(j, i, "·" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if pv.index[i] == "LSTM" else "normal",
                    color="white" if (not np.isnan(v) and v > 2.75) else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(SEASONS)
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    ax.grid(False); ax.set_title(f"arm {arm}")
    ax.axhline(pv.shape[0] - 1.5, color="k", lw=1)
fig.suptitle("Figura 5: RMSE no teste por modelo e trimestre ( · = o modelo não está no top-5 do trimestre)", fontsize=12.5)
plt.show()

rank = ps.assign(posicao=ps.groupby(["arm", "season"]).RMSE.rank()).query("model == 'LSTM'").pivot_table(
    index="arm", columns="season", values="posicao")[list(SEASONS)]
print("posição da LSTM entre os 6 modelos de cada trimestre (1 = melhor):"); print(rank.astype(int).to_string())
""")

py("""
cor_f = {"LSTM": "#c44", "boosting": "#2a7ab0", "bagging": "#5aa469", "outros": "#999999"}
d = ef_m.sort_values("efeito_full_vs_base")
y = np.arange(len(d))
fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
for yi, (_, r) in zip(y, d.iterrows()):
    ax.errorbar(r.efeito_full_vs_base, yi, xerr=[[r.efeito_full_vs_base - r.ci_lo], [r.ci_hi - r.efeito_full_vs_base]],
                fmt="o", ms=7, color=cor_f[r.familia], capsize=3, lw=2)
ax.axvspan(-d.sesoi.iloc[0], d.sesoi.iloc[0], color="#ddd", alpha=0.7, zorder=0, label=f"±SESOI ({d.sesoi.iloc[0]:.3f})")
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(d.model)
ax.set_xlabel("ganho de RMSE ao passar do base para o full (m/s); positivo = o full é melhor")
ax.set_title("Figura 6: o que somar os grupos 2–4 faz com cada modelo")
for f_, c_ in cor_f.items():
    ax.plot([], [], "o", color=c_, label=f_)
ax.legend(loc="lower right")
plt.show()
print(ef_m[["model", "familia", "efeito_full_vs_base", "ci_lo", "ci_hi", "veredito"]].round(3).to_string(index=False))
""")

md("""
**Leitura.** Com o `base` (grupo 1), a **LSTM compete com as melhores árvores**: 1º em MAM (2,09), 2º em SON, 3º em
JJA, 4º em DJF, sempre a menos de ~0,05 m/s do melhor. Com o `full` ela cai para o **último lugar em 3 de 4
trimestres**: as duas redes (LSTM −0,285 e MLP −0,243) **perdem** com os grupos extras, as **árvores ficam
indiferentes** (efeitos entre −0,003 e +0,046) e os **modelos lineares ganham** (BayesianRidge +0,068, regressão
linear +0,062, as únicas vitórias "acrescenta"). O padrão é de capacidade: modelos que ajustam muitas colunas
(redes) sobreajustam as colunas inúteis, os que as ignoram (árvores) não se importam, e os lineares, com forte
regularização implícita, aproveitam um pouco. A LSTM no Modal (T4) reproduz a rodada local (efeito −0,285 contra
−0,290; RMSE do `base` 2,281 nos dois): o resultado não depende da máquina.
""")

md("""
## Limitações e leitura final

- **Uma seed e hiperparâmetros da pipeline** (96 unidades, janela de 24 h, parada antecipada), sem ajuste para
  este estudo. O "prejudica" da LSTM é sobreajuste com ~5,6 mil linhas de treino por trimestre e até 167 colunas
  repetidas em 24 passos, e não prova que os grupos 2 e 4 sejam inúteis para uma rede regularizada.
- **Importância por permutação mede uso.** As estáticas do grupo 4 aparecem como as mais usadas por *ambos* os
  modelos e, ao mesmo tempo, não acrescentam nada fora da amostra: identificam a estação.
- **Árvores e LSTM não veem a mesma coisa.** A LSTM tem as 24 h anteriores; ainda assim com o `base` ela só
  empata com as árvores, então o histórico horário não traz ganho mensurável aqui.
- A seção 4 usa a LSTM do Modal com **uma seed** e só os arms `base` e `full`. Para decidir de forma robusta, repetir a LSTM nas 5 seeds e em todos os arms (Modal: `--stage fit-lstm`).
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
