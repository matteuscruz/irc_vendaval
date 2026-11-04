"""Gera `notebooks/feature_study_grupos/atual/ml_vs_lstm_global.ipynb`: ML e LSTM nos mesmos desenhos (local,
regional RS/SC, global e LOCO global), clusters 1–4 e 12, mesmas linhas e mesmo EQM.

Lê `artifacts/feature_study/global/` (`scripts/treinar_global.py` + `src/modal/global_lstm.py`). Nada é treinado.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/ml_vs_lstm_global.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# ML × LSTM, de igual para igual: local, regional, global e LOCO

As duas famílias com **as mesmas linhas** (estação, dia), **o mesmo split** (calendário do estudo: teste =
jan/abr/jul/out de 2000–2024; validação = blocos (ano, mês); 2025 fora), **as mesmas 49 colunas** (selecionadas, sem
`anor`) e **o mesmo EQM** ajustado na validação do próprio treino. Só o modelo muda:

- **ML**: o campeão do cluster 3 em cada trimestre (MLP DJF, ExtraTrees MAM, CatBoost JJA, HistGB SON), que vê a
  hora do pico;
- **LSTM**: a do estudo (LSTM(96) → Dropout(0,3) → Dense(1), Huber, early stopping), que vê as **24 horas até o
  pico** das mesmas colunas.

| desenho | treina com | avaliado em |
|---|---|---|
| **A** (só ML) | o modelo do cluster 3 do estudo | todos |
| **B** local | as estações do próprio cluster | todos (matriz treino × teste) |
| **C** regional | RS/SC (clusters 1–4) | todos (no 12 é região não vista) |
| **G** global | clusters 1–4 + 12 | todos |
| **L** LOCO global | todos menos o avaliado | só o que ficou de fora (região nunca vista) |

Uma seed (42) por ajuste nas duas famílias. Só entram linhas com janela de 24 h completa.
""")

py("""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
G = RAIZ / "artifacts/feature_study/global"
TARGET = "daily_wind_gust_max"
mt = pd.read_csv(G / "metricas_global.csv") if (G / "metricas_global.csv").exists() else pd.read_csv(G / "metricas_ml.csv")
TEM_LSTM = "LSTM" in set(mt.familia)
base = pd.read_parquet(G / "base_global.parquet", columns=["row_id", "estacao", "cluster_id", "time", "season", "_split",
                                                          "janela_ok", TARGET, "era5_gust_max"])
print("famílias:", sorted(mt.familia.unique()), "| linhas com janela completa:", int(base.janela_ok.sum()), "de", len(base))
print(base[base.janela_ok].groupby(["cluster_id", "_split"]).size().unstack().to_string())
CLS = [1, 2, 3, 4, 12]
DES = {"A": "A cl.3", "B": "B local", "C": "C regional", "G": "G global", "GA": "GA anual", "L": "L LOCO"}


def desenho_de(chave: pd.Series) -> pd.Series:
    # B1…B12 → B, L1…L12 → L; as outras chaves (A, C, G, GA) são o próprio desenho
    return chave.where(~chave.str.match(r"^[BL]\d"), chave.str[0]).map(DES)


def tabela(recorte="teste 2000–2024", met="RMSE"):
    t = mt[(mt.trimestre == "Todos") & (mt.recorte == recorte)].copy()
    t = t[~(t.chave.str.startswith("B") & (t.chave.str[1:] != t.cluster.astype(str)))]       # B: só o local
    t["d"] = desenho_de(t.chave)
    era = t[t.versao == "ERA5"].groupby("cluster")[met].first()
    linhas = {"ERA5": era}
    for fam in ("ML", "LSTM"):
        for versao, suf in (("modelo", ""), ("Corrigido ML", " + EQM")):
            for d in DES.values():
                v = t[(t.familia == fam) & (t.versao == versao) & (t.d == d)].groupby("cluster")[met].first()
                if len(v):
                    linhas[f"{fam} {d}{suf}"] = v
    return pd.DataFrame(linhas).T.reindex(columns=CLS)
""")

md("## 1. Tabela principal: RMSE diário por cluster (meses de teste 2000–2024)")

py("""
for rec in ("teste 2000–2024", "2025"):
    print(f"=== RMSE, {rec}")
    print(tabela(rec).round(3).to_string())
    print()
print("=== POD no P99, teste 2000–2024")
print(tabela("teste 2000–2024", "POD_P99").round(3).to_string())
""")

md("""
## 2. Ganho sobre o ERA5: ML × LSTM, por desenho e cluster

Verde = melhor que o ERA5. Linhas = desenho; colunas = cluster avaliado. Sem EQM (erro diário) e com EQM.
""")

py("""
def ganho_mat(fam, versao):
    t = tabela()
    suf = "" if versao == "modelo" else " + EQM"
    linhas = [f"{fam} {d}{suf}" for d in DES.values() if f"{fam} {d}{suf}" in t.index]
    return (t.loc["ERA5"] - t.loc[linhas]).rename(index=lambda s: s.replace(f"{fam} ", "").replace(suf, ""))


fams = ["ML", "LSTM"] if TEM_LSTM else ["ML"]
fig, axs = plt.subplots(len(fams), 2, figsize=(15, 3.8 * len(fams)), constrained_layout=True, squeeze=False)
mats = {(f, v): ganho_mat(f, v) for f in fams for v in ("modelo", "Corrigido ML")}
lim = max(float(np.nanmax(np.abs(m.values))) for m in mats.values())
for i, f in enumerate(fams):
    for j, v in enumerate(("modelo", "Corrigido ML")):
        m, ax = mats[(f, v)], axs[i, j]
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if np.isfinite(m.values[a, b]):
                    ax.text(b, a, f"{m.values[a, b]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(CLS)), [f"cluster {x}" for x in CLS]); ax.set_yticks(range(len(m)), m.index)
        ax.set_title(f"{f} — {'sem EQM' if v == 'modelo' else 'com EQM'}"); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Ganho de RMSE sobre o ERA5 (m/s), meses de teste 2000–2024")
plt.show()
""")

md("""
## 3. LSTM menos ML, desenho a desenho

Diferença de RMSE (LSTM − ML) no mesmo desenho e cluster: **negativo (azul) = a LSTM erra menos**.
""")

py("""
if TEM_LSTM:
    t = tabela()
    fig, axs = plt.subplots(1, 2, figsize=(15, 3.8), constrained_layout=True)
    for ax, suf, tit in zip(axs, ("", " + EQM"), ("sem EQM", "com EQM")):
        ds = [d for d in DES.values() if f"LSTM {d}{suf}" in t.index and f"ML {d}{suf}" in t.index]
        m = pd.DataFrame({d: t.loc[f"LSTM {d}{suf}"] - t.loc[f"ML {d}{suf}"] for d in ds}).T
        lim = float(np.nanmax(np.abs(m.values)))
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                if np.isfinite(m.values[a, b]):
                    ax.text(b, a, f"{m.values[a, b]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(CLS)), [f"cluster {x}" for x in CLS]); ax.set_yticks(range(len(m)), m.index)
        ax.set_title(f"RMSE LSTM − ML ({tit})"); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
    plt.show()
else:
    print("LSTM ainda não avaliada: rode o Modal (fit + download) e `treinar_global.py --stage avaliar`.")
""")

md("""
## 4. Transferência: matriz treino × teste do desenho B (local)

Cada linha é o cluster de treino; cada coluna, o avaliado. Ganho de RMSE sobre o ERA5, sem EQM.
""")

py("""
fams = ["ML", "LSTM"] if TEM_LSTM else ["ML"]
fig, axs = plt.subplots(1, len(fams), figsize=(7.5 * len(fams), 4.6), constrained_layout=True, squeeze=False)
B = mt[mt.chave.str.startswith("B") & (mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")]
for ax, f in zip(axs[0], fams):
    v = B[(B.familia == f) & (B.versao == "modelo")].pivot_table(index="treino", columns="cluster", values="RMSE")
    e = B[(B.familia == f) & (B.versao == "ERA5")].pivot_table(index="treino", columns="cluster", values="RMSE")
    m = (e - v).reindex(index=[str(x) for x in CLS], columns=CLS)
    lim = float(np.nanmax(np.abs(m.values)))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=-lim, vmax=lim)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            ax.text(b, a, f"{m.values[a, b]:+.2f}", ha="center", va="center", fontsize=9,
                    weight="bold" if m.index[a] == str(m.columns[b]) else None)
    ax.set_xticks(range(len(CLS)), [f"avalia {x}" for x in CLS]); ax.set_yticks(range(len(CLS)), [f"treina {x}" for x in CLS])
    ax.set_title(f"{f}: local, ganho sobre o ERA5 (diagonal = próprio cluster)"); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()
""")

md("## 5. A troca acerto diário × extremos, por cluster")

py("""
t = mt[(mt.trimestre == "Todos") & (mt.recorte == "teste 2000–2024")].copy()
t = t[~(t.chave.str.startswith("B") & (t.chave.str[1:] != t.cluster.astype(str)))]
t["d"] = desenho_de(t.chave)
fig, axs = plt.subplots(1, len(CLS), figsize=(4.2 * len(CLS), 4.4), constrained_layout=True, sharey=False)
mk = {"modelo": "o", "Corrigido ML": "^"}
cor = {"ML": "#2a7ab0", "LSTM": "#17808a"}
for ax, cl in zip(axs, CLS):
    s = t[t.cluster == cl]
    e = s[s.versao == "ERA5"].iloc[0]
    ax.scatter(e.RMSE, e.POD_P99, s=120, color="#c44", marker="*", zorder=4, label="ERA5")
    for _, r in s[s.versao != "ERA5"].iterrows():
        ax.scatter(r.RMSE, r.POD_P99, s=45, color=cor[r.familia], marker=mk[r.versao], alpha=0.85, zorder=3)
        ax.annotate(r.d.split()[0], (r.RMSE, r.POD_P99), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_title(f"cluster {cl}"); ax.set_xlabel("RMSE (m/s)")
axs[0].set_ylabel("POD no P99")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="*", color="#c44", ls="", ms=12, label="ERA5"),
                    Line2D([], [], marker="o", color="#2a7ab0", ls="", label="ML sem EQM"),
                    Line2D([], [], marker="^", color="#2a7ab0", ls="", label="ML com EQM"),
                    Line2D([], [], marker="o", color="#17808a", ls="", label="LSTM sem EQM"),
                    Line2D([], [], marker="^", color="#17808a", ls="", label="LSTM com EQM")],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.suptitle("Cada ponto = um desenho (letra); canto superior esquerdo = melhor nos dois"); plt.show()
""")

md("## 6. Quantis: global por trimestre (G) × global anual (GA)")

py("""
pml = pd.read_parquet(G / "previsoes_ml.parquet")
pls = pd.read_parquet(G / "previsoes_lstm.parquet") if TEM_LSTM else None
tem_ga = "GA_bruto" in pml
b = base[base.janela_ok].merge(pml[["row_id", "G_bruto", *(["GA_bruto", "GA_eqm"] if tem_ga else [])]], on="row_id")
if pls is not None:
    cl_ = ["row_id", "G_bruto", *(["GA_bruto", "GA_eqm"] if "GA_bruto" in pls else [])]
    b = b.merge(pls[cl_].rename(columns={c_: "L" + c_ for c_ in cl_ if c_ != "row_id"}), on="row_id")
b = b[b._split.isin(["test", "novo"])]
VERS = {"ERA5": ("era5_gust_max", "#c44", "-"), "ML global por trimestre": ("G_bruto", "#9ecae1", "-")}
if tem_ga:
    VERS.update({"ML anual": ("GA_bruto", "#2a7ab0", "-"), "ML anual + EQM": ("GA_eqm", "#2a7ab0", "--")})
if pls is not None:
    VERS["LSTM global por trimestre"] = ("LG_bruto", "#99d8c9", "-")
    if "LGA_bruto" in b:
        VERS.update({"LSTM anual": ("LGA_bruto", "#17808a", "-"), "LSTM anual + EQM": ("LGA_eqm", "#17808a", "--")})
fig, axs = plt.subplots(1, len(CLS), figsize=(4.2 * len(CLS), 4.4), constrained_layout=True)
qs = np.linspace(0.5, 0.999, 200)
for ax, cl in zip(axs, CLS):
    d = b[b.cluster_id == cl].dropna(subset=[v for v, _, _ in VERS.values()])
    qi = np.quantile(d[TARGET], qs)
    for k, (v, cor_, ls) in VERS.items():
        ax.plot(qi, np.quantile(d[v], qs), color=cor_, ls=ls, lw=1.8, label=k)
    ax.plot([qi[0], qi[-1]], [qi[0], qi[-1]], "k:", lw=1)
    ax.set_title(f"cluster {cl} (n={len(d)})"); ax.set_xlabel("quantil do INMET (m/s)")
axs[0].set_ylabel("mesmo quantil (m/s)"); axs[0].legend(fontsize=7)
fig.suptitle("Global por trimestre × global anual: quantis da mediana ao P99,9 (dias não vistos)"); plt.show()
""")

md("""
## 7. Um modelo por trimestre (G) × um modelo para o ano todo (GA)

O **G** tem 4 modelos (um por trimestre) e 4 EQM; o **GA** tem UM modelo para todos os trimestres, com o dia do ano
como coluna, e UM EQM. Mesmos clusters (1–4 + 12), mesmas linhas. No ML, o GA é um CatBoost; no G, os campeões de cada
trimestre. RMSE por trimestre (todos os clusters juntos) e por cluster.
""")

py("""
t = mt[(mt.recorte == "teste 2000–2024") & mt.chave.isin(["G", "GA"])]
por_trim = t[t.versao == "modelo"].pivot_table(index=["familia", "chave"], columns="trimestre", values="RMSE")
por_trim = por_trim.reindex(columns=["Todos", "DJF", "MAM", "JJA", "SON"])
print("RMSE sem EQM, todos os clusters, por trimestre (cada linha de 'Todos' é a média das métricas por cluster):")
print(t[t.versao == "modelo"].pivot_table(index=["familia", "chave"], columns="cluster", values="RMSE",
                                          aggfunc="first").round(3).to_string())
fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), constrained_layout=True)
for ax, versao, tit in zip(axs, ("modelo", "Corrigido ML"), ("sem EQM", "com EQM")):
    s_ = t[(t.versao == versao) & (t.trimestre == "Todos")]
    e_ = mt[(mt.recorte == "teste 2000–2024") & (mt.trimestre == "Todos") & (mt.versao == "ERA5")].groupby("cluster").RMSE.first()
    x = np.arange(len(CLS)); w = 0.16
    ax.bar(x - 2 * w, e_.reindex(CLS), w, color="#c44", label="ERA5")
    for k, (fam, ch, cor) in enumerate([("ML", "G", "#9ecae1"), ("ML", "GA", "#2a7ab0"),
                                        ("LSTM", "G", "#99d8c9"), ("LSTM", "GA", "#17808a")]):
        v = s_[(s_.familia == fam) & (s_.chave == ch)].set_index("cluster").RMSE.reindex(CLS)
        ax.bar(x + (k - 1) * w, v, w, color=cor, label=f"{fam} {'anual (GA)' if ch == 'GA' else 'por trimestre (G)'}")
    ax.set_xticks(x, [f"cluster {c_}" for c_ in CLS]); ax.set_ylabel("RMSE (m/s)"); ax.set_title(tit)
    ax.set_ylim(1.5, None); ax.legend(fontsize=7)
fig.suptitle("Global por trimestre × global anual (meses de teste 2000–2024)"); plt.show()
""")

py("""
for versao in ("modelo", "Corrigido ML"):
    s_ = mt[(mt.recorte == "teste 2000–2024") & mt.chave.isin(["G", "GA"]) & (mt.versao == versao) & (mt.trimestre != "Todos")]
    print(f"--- RMSE {'sem' if versao == 'modelo' else 'com'} EQM por trimestre e cluster")
    print(s_.pivot_table(index=["familia", "chave", "trimestre"], columns="cluster", values="RMSE").round(3).to_string())
""")

md("""
## Leitura (execução de 24/10/2026; confira contra as tabelas se reexecutar)

**Um modelo para o ano todo (GA) ganha de um modelo por trimestre (G), nas duas famílias, em todos os clusters,
em 2000–2024 e em 2025.** RMSE sem EQM nos meses de teste:

| | 1 | 2 | 3 | 4 | 12 |
|---|---|---|---|---|---|
| ERA5 | 2,62 | 2,54 | 2,66 | 3,44 | 2,27 |
| ML global por trimestre | 2,41 | 2,29 | 2,46 | 2,79 | 2,05 |
| ML global anual | 2,32 | 2,26 | 2,39 | 2,68 | 2,01 |
| LSTM global por trimestre | 2,41 | 2,28 | 2,43 | 2,74 | 1,97 |
| **LSTM global anual** | **2,28** | **2,24** | **2,33** | **2,57** | **1,85** |
| melhor modelo local | 2,28 | 2,33 | 2,40 | 2,72 | 1,92 |

- **A LSTM anual é a melhor abordagem testada até aqui.** Ela empata com o melhor modelo local no cluster 1 e ganha
  dele nos outros quatro, com um único modelo para tudo. A correlação com o INMET sobe para 0,74–0,78 (contra
  0,62–0,71 do ERA5).
- **Por quê.** Com um modelo só, cada trimestre aproveita os dados dos outros: são 222 mil linhas de treino em vez
  de ~55 mil. O dia do ano (seno e cosseno) dá a época sem cortar os dados. O ganho aparece em todos os trimestres,
  e é maior em JJA na LSTM (2,23 → 1,99).
- **Com EQM, pela primeira vez a correção ganha do ERA5 no erro diário em todos os clusters e mantém ou supera a
  detecção de extremos.** A LSTM anual + EQM tem RMSE de 1,95 a 2,70 (o ERA5 vai de 2,27 a 3,44) e POD no P99 de 0,13
  a 0,28 (o ERA5, de 0,08 a 0,22). O ML anual + EQM detecta um pouco mais (0,15–0,31), com RMSE um pouco maior.
- **Ressalva do ML.** No ML, o GA usa CatBoost e o G usa os campeões de cada trimestre: parte do ganho pode vir do
  CatBoost. Na LSTM a arquitetura é a mesma nos dois, então lá o ganho é só do modelo único.

**Os demais desenhos:** com estações da região no treino, ML e LSTM empatam (local, regional, global por trimestre).
Em região não vista (LOCO), a LSTM generaliza um pouco pior que o ML (+0,06 a +0,15 m/s).

**Recomendação:** **LSTM global anual**, sem EQM para o erro do dia a dia e com EQM para percentis e extremos.
O **ML global anual (CatBoost)** fica a poucos centésimos dela e não precisa de janelas nem de GPU: é a
alternativa mais barata.
""")


md("""
## Limites

1. **Uma seed** por ajuste nas duas famílias: diferenças de poucos centésimos estão no ruído.
2. **A LSTM não teve busca de arquitetura**, a mesma do estudo; o ML usa os campeões escolhidos no cluster 3. Nenhum
   dos dois foi retriado por região.
3. **Janelas completas**: linhas sem as 24 h anteriores (início de 2000) ficam fora das duas famílias.
4. O **cluster 12** tem grade e vizinhança próprias (download separado); os clusters 1–4 compartilham a grade de RS/SC.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
