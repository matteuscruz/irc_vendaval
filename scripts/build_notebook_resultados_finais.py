"""Gera `notebooks/feature_study_grupos/atual/resultados_finais_grupos.ipynb` — seis figuras essenciais.

Resultados finais do estudo por grupos, com os grupos lado a lado. Lê só o que a
execução de referência (Modal, 5 seeds) e a análise de robustez já gravaram; a
única conta nova é o SHAP por grupo, que reajusta dois modelos por trimestre em
segundos.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/atual/resultados_finais_grupos.ipynb")
nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Resultados finais — grupos de features lado a lado

Cluster 3, 9 estações, 2000–2024. **Base = grupo 1**; os grupos 2, 3 e 4 entram por
cima como grupos inteiros. Seis figuras, cada uma responde uma pergunta:

| # | pergunta | figura |
|---|---|---|
| 1 | Qual grupo acrescenta informação? | efeito por grupo com IC, SESOI e controles |
| 2 | E nas caudas, não só no erro médio? | efeito por grupo × métrica de cauda |
| 3 | O que o modelo **usa**? | SHAP por grupo e trimestre, com os controles ao lado |
| 4 | Como ficam os arms entre si e contra o ERA5? | desempenho de todos os arms |
| 5 | O modelo é calibrado e detecta os extremos? | calibração e excedência |
| 6 | Dá para cortar as variáveis? | modelo treinado no Modal só com as 12 selecionadas |

**Fonte:** execução de referência no Modal (`cluster3_groups_modal`: 5 *seeds*, 10 modelos,
97 blocos de bootstrap) e `robustez/` (análises sem custo de nuvem). Todas as comparações
com o ERA5 avaliam a **rajada** do ERA5 (`gust10fg` na hora do pico), nas mesmas linhas
de cada modelo. **Leia sempre junto das limitações do fim.**
""")

py("""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
MODAL = RAIZ / "artifacts/feature_study/cluster3_groups_modal"
LOCAL = RAIZ / "artifacts/feature_study/cluster3_groups"
RESUMO, ROB = MODAL / "summary/main", MODAL / "robustez"

efeitos = pd.read_csv(RESUMO / "effects.csv")
esq = pd.read_csv(ROB / "efeitos_esquemas.csv")
esq = esq[esq.scheme == "mes (ano-mês)"]
desemp = pd.read_csv(ROB / "desempenho_cauda.csv").set_index("arm")
cal = pd.read_csv(ROB / "calibracao_condicional.csv")
exc = pd.read_csv(ROB / "excedencia.csv")

COR_G = {"grupo1": "#2a7ab0", "grupo2": "#d98c3f", "grupo3": "#5aa469", "grupo4": "#8a6fb0",
         "ruído (controle)": "#999999", "estáticas permutadas (controle)": "#c44"}
COR_V = {"acrescenta": "#2a7ab0", "sem informação": "#bbbbbb", "inconclusivo": "#e0a040", "prejudica": "#c44"}
COR_P = {"ERA5 (rajada bruta)": "#c44", "modelo, arm base": "#2a7ab0", "modelo, arm full": "#5aa469"}
print("modelos:", int(efeitos[(efeitos.family == "all")].n_models.max()),
      "| réplicas:", int(efeitos.n_replicates.max()), "| blocos:", int(efeitos.n_blocos.max()))
""")

# ── Fig 1 ───────────────────────────────────────────────────────────────────
md("""
## 1. Qual grupo acrescenta informação?

Efeito no RMSE (m/s; **positivo = o arm melhor é melhor**), com IC de 95 % do bootstrap
pareado em blocos. A faixa cinza é o **SESOI** (1 % do RMSE da base): um efeito só
"acrescenta" se o IC inteiro estiver à direita dela. À esquerda, **somar** cada grupo ao
grupo 1; à direita, **remover** cada grupo do conjunto completo. Os dois controles
negativos ficam no mesmo gráfico, como régua do que é "ganho sem informação".
""")

py("""
e = efeitos[(efeitos.family == "all") & (efeitos.metric == "rmse")].set_index("comparison")
sesoi = float(e.sesoi.iloc[0])

painel_a = [("add_grp__grupo2", "somar grupo 2"), ("add_grp__grupo3", "somar grupo 3"),
            ("add_grp__grupo4", "somar grupo 4"), ("ctrl__noise", "controle: ruído"),
            ("ctrl__perm_static", "controle: estáticas\\npermutadas")]
painel_b = [("drop_grp__grupo1", "remover grupo 1"), ("drop_grp__grupo2", "remover grupo 2"),
            ("drop_grp__grupo3", "remover grupo 3"), ("drop_grp__grupo4", "remover grupo 4")]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1.25]})
for ax, itens, titulo in ((axes[0], painel_a, "A · somar à base (grupo 1)"),
                          (axes[1], painel_b, "B · remover do conjunto completo")):
    y = np.arange(len(itens))[::-1]
    for yi, (k, rot) in zip(y, itens):
        r = e.loc[k]
        ax.errorbar(r.effect, yi, xerr=[[r.effect - r.ci_lo], [r.ci_hi - r.effect]], fmt="o", ms=8,
                    color=COR_V.get(r.verdict, "k"), ecolor=COR_V.get(r.verdict, "k"), capsize=4, lw=2, zorder=3)
        ax.text(r.ci_hi + 0.006, yi, f"{r.effect:+.3f}  ({r.verdict})", va="center", fontsize=8.5)
    ax.axvspan(-sesoi, sesoi, color="#dddddd", alpha=0.7, zorder=0, label=f"±SESOI ({sesoi:.3f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r for _, r in itens])
    ax.set_xlabel("ganho de RMSE (m/s)")
    ax.set_title(titulo)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5, frameon=False)
fig.suptitle("Figura 1 — Contribuição de cada grupo (RMSE, 5 seeds × 10 modelos)", fontsize=12.5)
plt.show()
""")

md("""
**Leitura.** Só o **grupo 1** acrescenta: removê-lo custa +0,216 m/s, cerca de nove vezes o SESOI.
Somar os grupos 2 e 3 não traz nada (intervalos dentro da faixa cinza). O **grupo 4**
ganha +0,021, e o controle com estáticas **permutadas entre estações** ganha +0,019: as
duas barras são indistinguíveis (`estática real − permutada` = +0,002), então a
informação está em *saber qual é a estação*, não no relevo. O controle de ruído é zero.
""")

# ── Fig 2 ───────────────────────────────────────────────────────────────────
md("""
## 2. E nas caudas?

O mesmo efeito, agora em RMSE restrito aos dias acima do P90, P95 e P99 da observação e
na perda quantílica (*pinball*) da previsão pontual (τ = 0,95 e 0,99). `*` marca IC que
exclui zero. A cor é padronizada por coluna, porque as caudas têm escalas diferentes.
""")

py("""
linhas = [("drop_grp__grupo1", "remover grupo 1"), ("add_grp__grupo2", "somar grupo 2"),
          ("add_grp__grupo3", "somar grupo 3"), ("add_grp__grupo4", "somar grupo 4"),
          ("full_vs_base", "full contra base"), ("ctrl__noise", "controle: ruído"),
          ("ctrl__perm_static", "controle: estáticas perm.")]
metricas = [("rmse", "RMSE"), ("rmse_p90", "RMSE\\ndias ≥ P90"), ("rmse_p95", "RMSE\\ndias ≥ P95"),
            ("rmse_p99", "RMSE\\ndias ≥ P99"), ("pinball_95", "pinball\\nτ=0,95"), ("pinball_99", "pinball\\nτ=0,99")]
X = esq.set_index(["comparison", "metric"])
val = np.array([[X.loc[(k, m)].effect for m, _ in metricas] for k, _ in linhas])
lo = np.array([[X.loc[(k, m)].ci_lo for m, _ in metricas] for k, _ in linhas])
hi = np.array([[X.loc[(k, m)].ci_hi for m, _ in metricas] for k, _ in linhas])
norm = val / np.abs(val).max(axis=0, keepdims=True)

fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
im = ax.imshow(norm, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
for i in range(val.shape[0]):
    for j in range(val.shape[1]):
        sig = (lo[i, j] > 0) or (hi[i, j] < 0)
        ax.text(j, i, f"{val[i, j]:+.3f}" + ("*" if sig else ""), ha="center", va="center",
                fontsize=9.5, fontweight="bold" if sig else "normal",
                color="white" if abs(norm[i, j]) > 0.55 else "black")
ax.set_xticks(range(len(metricas)))
ax.set_xticklabels([r for _, r in metricas])
ax.set_yticks(range(len(linhas)))
ax.set_yticklabels([r for _, r in linhas])
ax.grid(False)
ax.set_title("Figura 2 — Efeito por grupo e métrica (m/s; * = IC 95 % exclui zero; azul = o arm melhor é melhor)")
plt.show()
""")

md("""
**Leitura.** O padrão se mantém em todas as caudas: o grupo 1 domina e os grupos 2 e 3 não
se distinguem de zero, nem em P99, nem na perda quantílica. O grupo 4 e o controle de
estáticas permutadas andam juntos em toda coluna. Em P99 quase nada tem `*` porque há só
69 casos por modelo: o intervalo é largo, e "sem efeito" ali significa "sem poder".
""")

# ── Fig 3 ───────────────────────────────────────────────────────────────────
md("""
## 3. O que o modelo **usa**? (SHAP por grupo)

SHAP mede quanto o **modelo** usa cada grupo, não quanto ele melhora a previsão fora da
amostra. Por isso cada trimestre traz três barras: o `full`, e os dois arms de controle
(base + ruído; base + estáticas permutadas), que dão o que é "uso sem informação". O SHAP
de um grupo é a soma das suas colunas (exato). Dois modelos de árvore, CatBoost e
LightGBM, reajustados aqui com seed 42 no treino de cada trimestre, avaliados no teste.
""")

py("""
import time

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.feature_study.diagnostics import shap_groups as sg

DADOS = LOCAL / "data"
meta = json.loads((DADOS / "meta.json").read_text())
arms = {a["name"]: a for a in json.loads((DADOS / "arms.json").read_text())}
GRUPO_DE = meta["group_of"]
ALVO = "daily_wind_gust_max"

treino = pd.read_parquet(DADOS / "sample_full.parquet")
treino = treino[treino._split == "train"]
teste = pd.read_parquet(DADOS / "test.parquet")

# o SHAP usa os dados do estudo LOCAL; confere que é a MESMA população do Modal
modal_teste = pd.read_parquet(MODAL / "data/test.parquet", columns=["estacao", "time"])
assert len(modal_teste) == len(teste) and (
    modal_teste.sort_values(["estacao", "time"]).reset_index(drop=True)[["estacao", "time"]]
    .equals(teste[["estacao", "time"]].sort_values(["estacao", "time"]).reset_index(drop=True))), \\
    "população local diferente da do Modal"

MODELOS = {
    "CatBoost": lambda: CatBoostRegressor(random_seed=42, verbose=0, thread_count=4),
    "LightGBM": lambda: LGBMRegressor(random_state=42, verbose=-1, n_jobs=4),
}
BRACOS = [("full", "full"), ("ctrl__noise", "+ ruído"), ("ctrl__perm_static", "+ estáticas\\npermutadas")]
SEASONS = ("DJF", "MAM", "JJA", "SON")

fatias, maxerro = [], 0.0
t0 = time.time()
for nome, fab in MODELOS.items():
    for s in SEASONS:
        a, b = treino[treino.season == s], teste[teste.season == s]
        for arm, _ in BRACOS:
            F = arms[arm]["features"]
            m = fab().fit(a[F], a[ALVO])
            sv, base = sg.shap_values(m, b[F])
            maxerro = max(maxerro, sg.check_additivity(m, b[F], sv, base))
            f = sg.group_share(sg.group_shap(sv, F, GRUPO_DE))
            fatias.append(pd.DataFrame({"modelo": nome, "season": s, "arm": arm, "grupo": f.index, "fatia": f.values}))
fatias = pd.concat(fatias, ignore_index=True)
print(f"{len(fatias.groupby(['modelo', 'season', 'arm']))} ajustes+SHAP em {time.time() - t0:.0f}s | "
      f"maior erro de aditividade: {maxerro:.1e}")
""")

py("""
ordem = ["grupo1", "grupo2", "grupo3", "grupo4", sg.CONTROL_NOISE, sg.CONTROL_PERM]
fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True, constrained_layout=True)
for i, nome in enumerate(MODELOS):
    for j, s in enumerate(SEASONS):
        ax = axes[i, j]
        d = fatias[(fatias.modelo == nome) & (fatias.season == s)]
        base = np.zeros(len(BRACOS))
        for g in ordem:
            v = np.array([d[(d.arm == arm) & (d.grupo == g)].fatia.sum() for arm, _ in BRACOS])
            ax.bar(range(len(BRACOS)), v, bottom=base, color=COR_G[g], edgecolor="white", linewidth=0.6,
                   label=g if (i == 0 and j == 0) else None)
            for k, (vv, bb) in enumerate(zip(v, base)):
                if vv >= 7:
                    ax.text(k, bb + vv / 2, f"{vv:.0f}", ha="center", va="center", fontsize=8.5, color="white")
            base += v
        ax.set_xticks(range(len(BRACOS)))
        ax.set_xticklabels([r for _, r in BRACOS], fontsize=8.5)
        ax.set_title(f"{nome} · {s}", fontsize=10)
        ax.grid(False)
        if j == 0:
            ax.set_ylabel("% do |SHAP| médio")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Figura 3 — SHAP por grupo: o full, e os controles (ruído e estáticas trocadas entre estações)", fontsize=12.5)
plt.show()

tab = (fatias[fatias.arm == "full"].pivot_table(index=["modelo", "season"], columns="grupo", values="fatia")
       [["grupo1", "grupo2", "grupo3", "grupo4"]].round(1))
print("fatia do SHAP por grupo no arm full (%):"); print(tab.to_string())
ctrl = fatias[fatias.grupo.isin([sg.CONTROL_NOISE, sg.CONTROL_PERM])].groupby(["arm", "modelo"]).fatia.agg(["min", "max"]).round(1)
print("\\ncontroles (% do SHAP; mín–máx entre trimestres):"); print(ctrl.to_string())
""")

md("""
**Leitura.** O grupo 1 concentra a maior fatia em todos os trimestres, mas o essencial
está nos controles: a coluna de **ruído** fica abaixo de ~3,5 %, e as **estáticas
permutadas** — que não têm física nenhuma — recebem uma fatia da mesma ordem que o grupo 4
verdadeiro. O SHAP **descreve o uso** e, aqui, confirma que o grupo 4 é usado como
identificador de estação. Não substitui o teste pareado da Figura 1: o grupo 2 também recebe
fatia relevante em alguns trimestres e, ainda assim, não acrescenta nada à previsão.
""")

# ── Fig 4 ───────────────────────────────────────────────────────────────────
md("""
## 4. Todos os arms lado a lado, contra o ERA5

Desempenho médio (10 modelos × 5 seeds) no teste, **todos nas mesmas linhas**, e a
rajada do ERA5 como referência vertical. São estimativas pontuais; a incerteza dos
*contrastes* está nas Figuras 1 e 2.
""")

py("""
ordem_arms = ["modelo, arm base", "add_grp__grupo2", "add_grp__grupo3", "add_grp__grupo4", "modelo, arm full",
              "drop_grp__grupo1", "drop_grp__grupo2", "drop_grp__grupo3", "drop_grp__grupo4",
              "ctrl__noise", "ctrl__perm_static"]
rotulos = {"modelo, arm base": "base (grupo 1)", "modelo, arm full": "full (1+2+3+4)",
           "add_grp__grupo2": "base + grupo 2", "add_grp__grupo3": "base + grupo 3", "add_grp__grupo4": "base + grupo 4",
           "drop_grp__grupo1": "full − grupo 1", "drop_grp__grupo2": "full − grupo 2",
           "drop_grp__grupo3": "full − grupo 3", "drop_grp__grupo4": "full − grupo 4",
           "ctrl__noise": "base + ruído", "ctrl__perm_static": "base + estát. perm."}
ref = desemp.loc["ERA5 (rajada bruta)"]
y = np.arange(len(ordem_arms))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True, constrained_layout=True)
for ax, (col, tit) in zip(axes, [("rmse", "RMSE (todos os dias)"), ("rmse_p90", "RMSE, dias ≥ P90"),
                                  ("rmse_p95", "RMSE, dias ≥ P95")]):
    v = np.array([desemp.loc[a, col] for a in ordem_arms])
    cor = ["#c44" if "drop_grp__grupo1" in a else "#2a7ab0" if "base" in a else "#bbbbbb" if "ctrl" in a else "#5aa469"
           for a in ordem_arms]
    ax.hlines(y, ref[col], v, color="#cccccc", lw=1.5, zorder=1)
    ax.scatter(v, y, c=cor, s=70, zorder=3, edgecolor="black", linewidth=0.5)
    ax.axvline(ref[col], color="#c44", ls="--", lw=1.3, label=f"ERA5 (rajada): {ref[col]:.2f}")
    for yi, vi in zip(y, v):
        ax.text(vi + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), yi, f"{vi:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([rotulos[a] for a in ordem_arms])
    ax.set_title(tit)
    ax.set_xlabel("m/s")
    ax.legend(loc="lower right", fontsize=8.5)
fig.suptitle("Figura 4 — Desempenho por arm (mesmas linhas para todos; ERA5 = rajada, mesma cobertura)", fontsize=12.5)
plt.show()
""")

md("""
**Leitura.** Todos os arms que **mantêm o grupo 1** ficam agrupados e bem abaixo do ERA5
(RMSE ≈ 2,28–2,31 contra 2,63): acrescentar grupos mexe na terceira casa decimal. Só
`full − grupo 1` volta para perto do ERA5 (2,51). Nas caudas o ganho sobre o ERA5 cai de 0,33 (todos os dias) para ≈0,29 (P90) e ≈0,28 (P95); em P99 o
contraste é inconclusivo (poucos casos). A Figura 5 mostra a calibração e a detecção.
""")

# ── Fig 5 ───────────────────────────────────────────────────────────────────
md("""
## 5. Calibração e detecção de extremos: ERA5, melhor ML e LSTM

Cinco previsores nas **mesmas linhas** do teste, por trimestre e no total:

| previsor | o que é |
|---|---|
| ERA5 | a rajada bruta do ERA5 |
| ML, base / selecionadas | o melhor modelo tabular de cada trimestre (o eleito pelo estudo) com o grupo 1 e com as 12 variáveis selecionadas |
| LSTM, base / selecionadas | a LSTM (janela de 24 h, seed 42) com o grupo 1 e com as mesmas 12 variáveis |

**(a) Calibração condicionada na previsão:** o observado médio dentro de faixas dos quantis da própria previsão; um previsor
calibrado fica sobre a diagonal. (Condicionar no **observado** alto, como no "viés em P90", dá viés negativo até para um
previsor perfeito, por regressão à média.) **(b, c) Detecção:** POD = fração dos extremos observados que foram previstos;
FAR = fração dos alarmes que eram falsos, em P90, P95 e P99 do observado **do próprio escopo**.
""")

py("""
from src.feature_study.diagnostics import predictor_curves as pc

campeao = pc.champions(MODAL / "summary")
print("melhor ML por trimestre:", campeao)
prev = pc.load_predictors(MODAL, LOCAL, campeao)
cal5, exc5 = pc.curves(prev)
ESC = list(cal5.escopo.unique())
print(f"{len(prev)} linhas, as mesmas para os 5 previsores | linhas por escopo:", prev.season.value_counts().to_dict())

ESTILO5 = {pc.ERA5: ("#c44", "s--"), "ML, base": ("#6fa8cf", "o-"), "ML, selecionadas": ("#1f5f99", "o--"),
           "LSTM, base": ("#7fcf9f", "^-"), "LSTM, selecionadas": ("#1b8a5a", "^--")}
fig, axes = plt.subplots(len(ESC), 3, figsize=(16, 4.3 * len(ESC)), constrained_layout=True)
for lin, esc in zip(axes, ESC):
    a = lin[0]
    for nome, (cor, est) in ESTILO5.items():
        g = cal5[(cal5.escopo == esc) & (cal5.previsor == nome)]
        a.plot(g.prev_media, g.obs_media, est, color=cor, ms=3.5, lw=1.5, label=nome)
    c = cal5[cal5.escopo == esc]
    lim = [c.prev_media.min() - 0.5, c.prev_media.max() + 0.5]
    a.plot(lim, lim, "k:", lw=1, label="calibração perfeita")
    a.set_xlabel("previsão média na faixa (m/s)"); a.set_ylabel(f"{esc}\\nobservado médio (m/s)")
    a.set_title("calibração" if esc == ESC[0] else "", fontsize=10)
    for a, col, tit in ((lin[1], "pod", "POD: extremos observados previstos"), (lin[2], "far", "FAR: alarmes falsos")):
        e = exc5[exc5.escopo == esc]
        larg = 0.16
        for i, (nome, (cor, _)) in enumerate(ESTILO5.items()):
            g = e[e.previsor == nome]
            xs = np.arange(len(g)) + (i - 2) * larg
            a.bar(xs, g[col], larg, color=cor, edgecolor="black", linewidth=0.4)
            for x, v in zip(xs, g[col]):
                a.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6, rotation=90)
        ref = e[e.previsor == pc.ERA5]
        a.set_xticks(np.arange(len(ref)))
        a.set_xticklabels([f"{r.limiar}\\n({r['valor_m/s']:.1f} m/s)" for _, r in ref.iterrows()], fontsize=8.5)
        a.set_ylim(0, max(0.75, float(e[col].max()) * 1.15))
        a.set_title(tit if esc == ESC[0] else "", fontsize=10)
        a.grid(axis="y", alpha=0.3)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Figura 5: calibração e detecção de extremos: ERA5, melhor ML e LSTM (base e variáveis selecionadas)", fontsize=12.5)
plt.show()

rm5 = {esc: {p: float(np.sqrt(np.mean((d[p] - d["y"]) ** 2))) for p in pc.PREVISORES} for esc, d in pc.scopes(prev).items()}
print("RMSE no teste (m/s):"); print(pd.DataFrame(rm5).T.round(3).to_string())
print("\\nPOD e FAR, todos os trimestres:")
print(exc5[exc5.escopo == pc.TODOS].pivot_table(index="limiar", columns="previsor", values=["pod", "far"])
      .round(2).reindex(columns=list(pc.PREVISORES), level=1).to_string())
""")

md("""
**Leitura.**
- **Calibração.** O ERA5 **superestima** a cauda em todos os trimestres: na faixa acima de P99 o observado médio fica 2,0 a
  3,2 m/s abaixo do previsto. O melhor ML e a LSTM ficam bem mais perto da diagonal (desvios dentro de ±1,7 m/s no topo);
  o ML tende a superestimar um pouco (−0,8 no total) e a LSTM a subestimar (+0,6 com o `base`).
- **Detecção.** **Nenhum dos quatro modelos detecta mais extremos raros que o ERA5**: o POD de P99 é 0,14 no ERA5, 0,05 a 0,07
  no ML e **0,01 na LSTM**, que quase nunca cruza um limiar alto. Em P90 e P95 o ML iguala o ERA5 (0,47 e 0,31). O ganho de
  todos está em **menos alarmes falsos**: FAR de P90 de 0,41 no ERA5 para 0,30 no ML e 0,24 na LSTM com o `base`.
- **Variáveis selecionadas.** Para o ML não mudam nada (RMSE 2,316 contra 2,293; POD e FAR praticamente iguais). Para a LSTM
  pioram (RMSE 2,405 contra 2,287, e menos detecção), em linha com a Figura 6 e com o notebook de importância.
- **Por trimestre** o quadro varia. Em **DJF** o ML detecta mais que o ERA5 em P90 e P95 (0,35 a 0,39 contra 0,26 em P90). Em
  **JJA** o ERA5 detecta mais em todos os limiares (0,61 contra 0,51 do ML em P90). Em **MAM** os quatro modelos ficam perto do
  ERA5 e a LSTM chega a 0,24 em P99. Na calibração do topo, a LSTM **subestima** em SON e JJA (o observado médio fica 1,0 a
  1,5 m/s acima do previsto) e superestima em MAM; o ML erra menos nos três (abaixo de 0,8 m/s), exceto em DJF, onde
  superestima 1,2 a 1,7 m/s. Em P99 há só ~34 casos por trimestre: POD e FAR dali oscilam muito (o FAR de 1,00 do ML em JJA
  vem de pouquíssimos alarmes), e uma seed na LSTM com um modelo por trimestre no ML não permite concluir diferenças pequenas.
""")

# ── Fig 5b: exemplos de acerto, extremo perdido e falso alarme ─────────────
md("""
### Como POD e FAR contam: exemplos reais

Para um limiar (aqui o **P95 do observado**, calculado em todos os trimestres), cada dia cai em uma de quatro casas:

| casa | observado ≥ limiar | previsto ≥ limiar | entra em |
|---|---|---|---|
| **acerto** (extremo capturado) | sim | sim | numerador do POD e do FAR (como acerto) |
| **extremo perdido** | sim | não | só no denominador do POD |
| **falso alarme** | não | sim | numerador do FAR |
| correto sem extremo | não | não | não entra em nenhum |

**POD = acertos ÷ (acertos + perdidos)**: dos extremos que aconteceram, quantos o previsor avisou. **FAR = falsos alarmes ÷
(acertos + falsos alarmes)**: dos avisos que ele deu, quantos eram falsos. O gráfico mostra os dias de teste de cada previsor
nessas casas; abaixo, séries de dias reais de cada tipo.
""")

py("""
tm = pd.read_parquet(MODAL / "data/test.parquet", columns=["row_id", "estacao", "time"])
pv = prev.merge(tm, on="row_id").sort_values(["estacao", "time"]).reset_index(drop=True)
LIM = float(np.quantile(pv["y"], 0.95))
EXEMPLO = ["ERA5", "ML, selecionadas", "LSTM, selecionadas"]


def casas(y, pred, lim):
    obs, prv = y >= lim, pred >= lim
    return np.select([obs & prv, obs & ~prv, ~obs & prv], ["acerto", "perdido", "falso alarme"], "correto sem extremo")


COR_CASA = {"acerto": "#1b9e77", "perdido": "#e08a00", "falso alarme": "#c44", "correto sem extremo": "#cfcfcf"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True, sharey=True, constrained_layout=True)
for ax, nome in zip(axes, EXEMPLO):
    cs = casas(pv["y"].to_numpy(), pv[nome].to_numpy(), LIM)
    for casa in ("correto sem extremo", "perdido", "falso alarme", "acerto"):
        m = cs == casa
        ax.scatter(pv.loc[m, "y"], pv.loc[m, nome], s=7 if casa == "correto sem extremo" else 14,
                   color=COR_CASA[casa], alpha=0.35 if casa == "correto sem extremo" else 0.85, label=f"{casa} ({m.sum()})", zorder=3)
    ax.axvline(LIM, color="k", ls=":", lw=1); ax.axhline(LIM, color="k", ls=":", lw=1)
    ax.plot([0, 40], [0, 40], color="#999", lw=0.8)
    a_, p_, f_ = (int((cs == k).sum()) for k in ("acerto", "perdido", "falso alarme"))
    ax.set_title(f"{nome}\\nPOD = {a_}/{a_ + p_} = {a_ / (a_ + p_):.0%}   ·   FAR = {f_}/{a_ + f_} = {f_ / (a_ + f_):.0%}", fontsize=10)
    ax.set_xlabel("observado, INMET (m/s)"); ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("previsto (m/s)")
axes[0].set_xlim(0, 36); axes[0].set_ylim(0, 36)
fig.suptitle(f"Figura 5b: cada dia de teste contado como acerto, extremo perdido ou falso alarme (limiar P95 = {LIM:.1f} m/s)", fontsize=12.5)
plt.show()
""")

py("""
import matplotlib.dates as mdates

PREV_EX = "ML, selecionadas"
pv["casa"] = casas(pv["y"].to_numpy(), pv[PREV_EX].to_numpy(), LIM)
acertos = pv[pv.casa == "acerto"].nlargest(2, "y")
perdidos = pv[pv.casa == "perdido"].nlargest(2, "y")
falsos = pv[pv.casa == "falso alarme"].assign(excesso=lambda d: d[PREV_EX] - d["y"]).nlargest(2, "excesso")
exemplos = pd.concat([acertos.assign(tipo="acerto"), perdidos.assign(tipo="extremo perdido"),
                      falsos.assign(tipo="falso alarme")])

fig, axes = plt.subplots(2, 3, figsize=(17, 7.2), constrained_layout=True)
for col, tipo in enumerate(("acerto", "extremo perdido", "falso alarme")):
    for lin, (_, ev) in enumerate(exemplos[exemplos.tipo == tipo].iterrows()):
        ax = axes[lin, col]
        g = pv[(pv.estacao == ev.estacao) & ((pv.time - ev.time).abs() <= pd.Timedelta(days=5))]
        ax.plot(g.time, g["y"], "o-", color="#111", ms=4, lw=1.3, label="INMET (observado)", zorder=5)
        ax.plot(g.time, g["ERA5"], "s--", color="#c44", ms=3, lw=1, alpha=0.85, label="ERA5")
        ax.plot(g.time, g[PREV_EX], "^-", color="#1f5f99", ms=4, lw=1.3, label=PREV_EX)
        ax.axhline(LIM, color="k", ls=":", lw=1, label=f"limiar P95 ({LIM:.1f})")
        ax.scatter([ev.time], [ev["y"]], s=170, facecolors="none", edgecolors=COR_CASA[ev.casa], linewidths=2.2, zorder=6)
        ax.scatter([ev.time], [ev[PREV_EX]], s=170, facecolors="none", edgecolors=COR_CASA[ev.casa], linewidths=2.2, zorder=6)
        ax.set_title(f"{tipo}: {ev.estacao}, {ev.time:%d/%m/%Y}\\nobservado {ev['y']:.1f} · previsto {ev[PREV_EX]:.1f} m/s", fontsize=9.5, loc="left")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.tick_params(axis="x", labelsize=8.5)
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Exemplos de dias contados pelo previsor «{PREV_EX}» (círculo = o dia contado)", fontsize=12.5)
plt.show()

tab_ex = exemplos[["tipo", "estacao", "time", "y", "ERA5", "ML, selecionadas", "LSTM, selecionadas"]].rename(
    columns={"y": "observado", "time": "dia"})
print(f"limiar P95 = {LIM:.2f} m/s | previsor dos exemplos: {PREV_EX}")
print(tab_ex.round(1).to_string(index=False))
""")

# ── Fig 6 ───────────────────────────────────────────────────────────────────
md("""
## 6. O modelo com as variáveis selecionadas (treinado no Modal)

Depois dos grupos, a pergunta prática: **dá para cortar as variáveis?** O arm `sel__val12`
foi treinado no mesmo estudo (5 seeds × 4 trimestres × 10 modelos) só com as **12 variáveis
(58 colunas)** escolhidas na **validação** — regra: piora do RMSE ao embaralhar acima do
piso de uma coluna de ruído e positiva em pelo menos 3 de 4 trimestres. As variáveis são
`gust10fg`, `gust`, `w10`, `w100`, `v10`, `v100`, `hora_solar_sin` (grupo 1), `cape`, `mcpr`
(grupo 2), `grad_mslp`, `grad_mslp_hpa_100km` (grupo 3) e `anor` (grupo 4). Cada modelo é
comparado nas **mesmas linhas** dos três arms; o Δ é pareado, com bootstrap em blocos
(ano, mês), 2000 sorteios, e o veredito usa o SESOI (1 % do RMSE da base).
""")

py("""
UNIDADES = MODAL / "units"
TAGS = ["full", "full_s43", "full_s44", "full_s45", "full_s46"]
ARMS_M = ["base", "full", "sel__val12"]
ALVO, REF = "daily_wind_gust_max", "era5_gust_max"
tm = pd.read_parquet(MODAL / "data/test.parquet").sort_values("row_id").reset_index(drop=True)
chave = (tm.year * 100 + tm.month).factorize()[0]
nb_m = chave.max() + 1
bloco_de = pd.Series(chave, index=tm.row_id.to_numpy())
era_res = pd.Series((tm[REF] - tm[ALVO]).to_numpy(), index=tm.row_id.to_numpy())
obs = pd.Series(tm[ALVO].to_numpy(), index=tm.row_id.to_numpy())
W = np.random.default_rng(42).multinomial(len(tm), np.ones(nb_m) / nb_m, size=2000).astype(float)


def rmse_draws(e2, b):
    soma = np.bincount(b, e2, nb_m); n = np.bincount(b, minlength=nb_m)
    return np.sqrt((W @ soma) / (W @ n)), np.sqrt(soma.sum() / n.sum())


casos = {}
for tag in TAGS:
    for s in ("DJF", "MAM", "JJA", "SON"):
        r = {a: pd.read_parquet(UNIDADES / tag / f"resid__{s}__{a}.parquet").set_index("row_id") for a in ARMS_M}
        for modelo in r["sel__val12"].columns:
            if all(modelo in r[a] for a in ARMS_M):
                casos.setdefault((tag, modelo), []).append(pd.concat({a: r[a][modelo] for a in ARMS_M}, axis=1).dropna())
casos = {k: pd.concat(v) for k, v in casos.items()}
print(f"{len(casos)} pares (seed, modelo); modelos: {sorted({m for _, m in casos})}")

NOMES = ARMS_M + ["ERA5"]
draws = {a: [] for a in NOMES}; pts = {a: [] for a in NOMES}
p90 = {a: [] for a in NOMES}
for (tag, modelo), d in casos.items():
    b = bloco_de.loc[d.index].to_numpy()
    alto = (obs.loc[d.index] >= obs.loc[d.index].quantile(0.90)).to_numpy()
    for a in NOMES:
        res = era_res.loc[d.index].to_numpy() if a == "ERA5" else d[a].to_numpy()
        dr, pt = rmse_draws(res ** 2, b)
        draws[a].append(dr); pts[a].append(pt); p90[a].append(np.sqrt(np.mean(res[alto] ** 2)))
D = {a: np.mean(v, axis=0) for a, v in draws.items()}
P = {a: float(np.mean(v)) for a, v in pts.items()}
P90 = {a: float(np.mean(v)) for a, v in p90.items()}
SES = 0.01 * P["base"]


def delta(pior, melhor):
    x = D[pior] - D[melhor]
    return P[pior] - P[melhor], *np.quantile(x, [0.025, 0.975])


pares = [("ERA5 → grupo 1 (base)", "ERA5", "base"), ("grupo 1 → seleção (12 var.)", "base", "sel__val12"),
         ("seleção (12 var.) → completo (167)", "sel__val12", "full"), ("grupo 1 → completo (167)", "base", "full")]
tab6 = pd.DataFrame([{"comparação": n, "ganho": delta(a, b)[0], "IC_lo": delta(a, b)[1], "IC_hi": delta(a, b)[2]}
                     for n, a, b in pares]).set_index("comparação")
tab6["veredito"] = np.where(tab6.IC_lo > SES, "acrescenta", np.where(
    (tab6.IC_hi < SES) & (tab6.IC_lo > -SES), "sem informação", np.where(tab6.IC_hi < -SES, "prejudica", "inconclusivo")))

rotulo = {"ERA5": "ERA5 (rajada)", "base": "grupo 1 (base, 45 col.)", "sel__val12": "seleção (12 var., 58 col.)",
          "full": "completo (167 col.)"}
ordem6 = ["ERA5", "base", "sel__val12", "full"]
cor6 = {"ERA5": "#c44", "base": "#e0a040", "sel__val12": COR_G["grupo1"], "full": "#5aa469"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
for a_, (dic, tit) in zip(ax[:2], ((P, "RMSE (todos os dias)"), (P90, "RMSE, dias ≥ P90"))):
    v = [dic[k] for k in ordem6]
    a_.barh([rotulo[k] for k in ordem6][::-1], v[::-1], color=[cor6[k] for k in ordem6][::-1], edgecolor="black", linewidth=0.5)
    for i, x in enumerate(v[::-1]):
        a_.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=9)
    a_.set_xlim(min(v) * 0.9, max(v) * 1.07); a_.set_xlabel("m/s"); a_.set_title(tit)
cv = {"acrescenta": "#2a7ab0", "sem informação": "#bbbbbb", "inconclusivo": "#e0a040", "prejudica": "#c44"}
yy = np.arange(len(tab6))[::-1]
for yi, (n, r) in zip(yy, tab6.iterrows()):
    ax[2].errorbar(r.ganho, yi, xerr=[[r.ganho - r.IC_lo], [r.IC_hi - r.ganho]], fmt="o", color=cv[r.veredito], capsize=4, lw=2, ms=8)
    ax[2].text(r.IC_hi + 0.01, yi, f"{r.ganho:+.3f} ({r.veredito})", va="center", fontsize=9)
ax[2].axvspan(-SES, SES, color="#ddd", alpha=0.7); ax[2].axvline(0, color="k", lw=0.8)
ax[2].set_yticks(yy); ax[2].set_yticklabels(tab6.index); ax[2].set_xlim(right=0.65)
ax[2].set_xlabel("ganho de RMSE (m/s); faixa cinza = ±SESOI"); ax[2].set_title("Ganho entre arms, IC 95 %")
fig.suptitle("Figura 6 — Modelo treinado no Modal só com as 12 variáveis selecionadas", fontsize=12.5)
plt.show()
print(pd.DataFrame({"RMSE": P, "RMSE ≥P90": P90}).loc[ordem6].round(4).to_string())
print(f"\\nSESOI = {SES:.4f} m/s"); print(tab6.round(4).to_string())
""")

md("""
**Leitura.** Cortar de 167 para 12 variáveis custa **0,002 m/s**, abaixo do SESOI; e o
conjunto reduzido também **não supera o grupo 1 sozinho** de forma relevante (+0,003). O
ganho real continua sendo o do grupo 1 sobre o ERA5. A vantagem da seleção é parcimônia, não
precisão. A seleção foi feita na validação, não no teste, para a avaliação não ser circular.
O estimador é uma reimplementação enxuta do `compute_effects` (sem correção de múltiplas
hipóteses); os números de conferência batem com o estudo (completo − grupo 1 = 0,005;
ERA5 → grupo 1 = 0,335). A escolha das variáveis está em
`config/selected_features_val12.json`.
""")

md("""
### Colunas exatas do modelo com as variáveis selecionadas (`sel__val12`)

A seleção foi por família (12 nomes), mas o modelo treina com **58 colunas** — cada família traz as suas
variantes (raio, estatística, defasagem). Abaixo, cada coluna com o nome completo, colorida pelo grupo,
com a importância média da LSTM por permutação (Δ RMSE em m/s; perto de zero = a coluna não pesa).
""")

py("""
sel = json.load(open(RAIZ / "config/selected_features_val12.json"))
arm_sel = next(a for a in json.load(open(LOCAL / "data/arms.json")) if a["name"] == "sel__val12")
colunas = arm_sel.get("features") or arm_sel.get("columns")
grupo_de = {v: g for g, vs in sel["by_group"].items() for v in vs}
fam = lambda c: max((v for v in grupo_de if c == v or c.startswith(v + "_")), key=len)
imp6 = pd.concat([pd.read_parquet(LOCAL / f"units/full_lstm/importance__{s_}__full.parquet") for s_ in ["DJF", "MAM", "JJA", "SON"]])
imp6 = imp6[imp6.nivel == "coluna"].groupby("nome").d_rmse.mean().reindex(colunas)
tab = pd.DataFrame({"coluna": colunas, "familia": [fam(c) for c in colunas], "grupo": [grupo_de[fam(c)] for c in colunas],
                    "importancia_lstm": imp6.values}).sort_values("importancia_lstm")
fig, ax = plt.subplots(figsize=(9, 0.23 * len(tab) + 1.5))
ax.barh(tab.coluna, tab.importancia_lstm, color=[COR_G[g] for g in tab.grupo])
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Δ RMSE ao permutar a coluna (m/s), média dos 4 trimestres")
ax.set_title(f"As {len(tab)} colunas do modelo com variáveis selecionadas")
ax.tick_params(axis="y", labelsize=7)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COR_G[g]) for g in ("grupo1", "grupo2", "grupo3", "grupo4")],
          labels=["grupo 1", "grupo 2", "grupo 3", "grupo 4"], loc="lower right")
plt.tight_layout(); plt.show()
for f_, d in tab.groupby("familia", sort=False):
    print(f"{f_} ({len(d)}): " + ", ".join(d.sort_values("importancia_lstm", ascending=False).coluna))
""")

md("""
## 7. Distribuição da rajada por trimestre e por método

Rajada máxima **diária** (média entre as estações de cada dia) no teste, por trimestre climático. Só
resíduos já gravados: nada foi treinado para esta figura. **Grupos II, III e IV** aparecem como
*grupo I + grupo* (arms `add_grp__*`): o "grupo isolado" (só as colunas de um grupo) nunca foi
gravado. Linha inferior: as 5 colunas mais importantes da LSTM (permutação) por trimestre; a
importância dos modelos de ML não foi persistida.
""")

py("""
from src.feature_study.diagnostics import seasonal_distribution as sd

MLA = {"ML, grupo I (base)": "base", "ML, +grupo II": "add_grp__grupo2", "ML, +grupo III": "add_grp__grupo3",
       "ML, +grupo IV": "add_grp__grupo4", "ML, selecionadas": "sel__val12"}
LSA = {"LSTM, base": "base", "LSTM, selecionadas": "sel__val12"}
campeao7 = pc.champions(MODAL / "summary")
dist = sd.load_series(MODAL, LOCAL, campeao7, MLA, LSA)
COLS = [sd.INMET, sd.ERA5, *MLA, *LSA]
diaria = sd.daily_station_mean(dist, COLS)
COR7 = {sd.INMET: "#111111", sd.ERA5: "#c44", "ML, grupo I (base)": COR_G["grupo1"], "ML, +grupo II": COR_G["grupo2"],
        "ML, +grupo III": COR_G["grupo3"], "ML, +grupo IV": COR_G["grupo4"], "ML, selecionadas": "#1f3f66",
        "LSTM, base": "#4cb8c4", "LSTM, selecionadas": "#17808a"}
ROT7 = {sd.INMET: "INMET", sd.ERA5: "ERA5", "ML, grupo I (base)": "ML\\nI", "ML, +grupo II": "ML\\n+II",
        "ML, +grupo III": "ML\\n+III", "ML, +grupo IV": "ML\\n+IV", "ML, selecionadas": "ML\\nsel.",
        "LSTM, base": "LSTM\\nbase", "LSTM, selecionadas": "LSTM\\nsel."}
imp = pd.concat([pd.read_parquet(LOCAL / f"units/full_lstm/importance__{s}__full.parquet") for s in sd.SEASONS])
imp = imp[imp.nivel == "coluna"]

fig = plt.figure(figsize=(20, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.45, wspace=0.95)
for j, s in enumerate(sd.SEASONS):
    ax = fig.add_subplot(gs[0, j])
    d = diaria[diaria.season == s]
    bp = ax.boxplot([d[c].dropna() for c in COLS], patch_artist=True, widths=0.65, showfliers=False,
                    medianprops=dict(color="white", lw=1.6))
    for b, c in zip(bp["boxes"], COLS):
        b.set(facecolor=COR7[c], alpha=0.85, edgecolor="none")
    ax.set_xticks(range(1, len(COLS) + 1)); ax.set_xticklabels([ROT7[c] for c in COLS], fontsize=7)
    ax.set_title(f"{s}  (n = {len(d)} dias)", fontweight="bold"); ax.set_ylabel("rajada diária média (m/s)" if j == 0 else "")
    ax.axhline(d[sd.INMET].median(), color="#111", lw=0.8, ls=":")
    t = (imp[imp.season == s].groupby(["nome", "grupo"]).d_rmse.mean().reset_index()
         .nlargest(5, "d_rmse").iloc[::-1])
    ax2 = fig.add_subplot(gs[1, j])
    ax2.barh(t.nome, t.d_rmse, color=[COR_G.get(g, "#999") for g in t.grupo])
    ax2.set_title(f"top-5 colunas da LSTM — {s}", fontsize=9); ax2.set_xlabel("Δ RMSE ao permutar (m/s)")
fig.suptitle("Distribuição da rajada diária por trimestre climático e por método (teste, média entre estações)", y=0.96, fontsize=13)
plt.show()

est7 = sd.summary(diaria, COLS)
print(est7.pivot_table(index="metodo", columns="escopo", values="mediana").reindex(COLS)[["Todos", *sd.SEASONS]].round(2).to_string())
print("\\nP90:")
print(est7.pivot_table(index="metodo", columns="escopo", values="p90").reindex(COLS)[["Todos", *sd.SEASONS]].round(2).to_string())
""")

md("""
## Limitações — leia junto das figuras

1. **Uma linha por dia, na hora do pico da rajada ERA5.** O perfil intradiário se perde. O
   estudo anterior, com as 24 horas em colunas, indicava relevância da convecção; aqui o
   grupo 2 não acrescenta, mas só enxerga a hora do pico. "Grupo 2 não ajuda" vale
   **para este desenho**.
2. **O portão das estáticas permutadas não deu zero** no bootstrap (+0,019, IC acima de
   zero), embora o teste de Diebold–Mariano não o distinga de zero (p = 0,26). As estáticas
   funcionam como identificador de estação, o que também aparece no SHAP (Figura 3).
3. **SHAP é atribuição, não contribuição** (Figura 3), e só cobre modelos de árvore.
4. **Caudas com poucos casos** (69 por modelo em P99), e nove estações, uma delas curta.
5. **Truncamento das previsões** em [0, 80] m/s não foi reportado; exige reexecução.

Material completo, métodos e testes: `extreme_winds_doc_paper/supplementar_pipeline_testes.tex`.
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
