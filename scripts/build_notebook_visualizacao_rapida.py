"""Gera `notebooks/feature_study_grupos/anteriores/raw_horario/visualizacao_rapida_series.ipynb` — ferramenta de olhada rápida.

Notebook curto e reutilizável: escolhe estação, trimestre e feature, e desenha
a série. A parte cara (ajustar os 4 modelos campeões) fica em cache, então só a
primeira execução demora.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/anteriores/raw_horario/visualizacao_rapida_series.ipynb")

nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Visualização rápida — série temporal por estação e trimestre

Ferramenta de olhada, não de medição. Escolha **estação**, **trimestre** e
**feature**, e veja quatro linhas:

- **INMET** — a rajada máxima observada (o alvo)
- **ERA5 cru** — o que o ERA5 entrega sem correção
- **modelo completo** — a previsão com todas as 585 features
- **modelo com a feature embaralhada** — a mesma previsão, com aquela variável
  destruída

A distância entre as duas últimas linhas é a contribuição da feature naquele dia.

> "Sem a feature" aqui é **embaralhada**, não removida — o modelo não é
> reajustado. Para medir de verdade, use `grupos_1_e_4_importancia_por_variavel.ipynb`
> (permutação com repetições) ou o estudo no Modal (bootstrap pareado).

**Primeira execução:** alguns minutos, ajustando os 4 modelos. Depois fica em
cache e a resposta é imediata.
""")

py("""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
BASE = RAIZ / "artifacts/feature_study/cluster3_raw"
CACHE = BASE / "models_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SEASONS = ("DJF", "MAM", "JJA", "SON")
ALVO, PROXY = "daily_wind_gust_max", "wind_mag_max"

meta = json.loads((BASE / "data/meta.json").read_text())
FEATURES = meta["base_features"] + meta["new_features"]
CAMPEAO = {s: v[0] for s, v in
           json.loads((BASE / "summary/top_models.json").read_text())["by_season"].items()}

treino = pd.read_parquet(BASE / "data/sample_full.parquet")
treino = treino[treino._split == "train"]
teste = pd.read_parquet(BASE / "data/test.parquet")

from src.feature_study.diagnostics.group_importance import DESCRICAO, blocos_das_novas
from src.feature_study.diagnostics.group_importance import base_variable_of

BLOCOS = {}
for col in FEATURES:
    if col not in ("latitude", "longitude"):
        BLOCOS.setdefault(base_variable_of(col), []).append(col)

print(f"{teste.estacao.nunique()} estações | {len(FEATURES)} features | "
      f"teste {teste.time.min():%Y} a {teste.time.max():%Y}")
""")

md("""
## Modelos (cache)

Ajusta o campeão que o estudo elegeu para cada trimestre. Rode uma vez; depois
o cache responde na hora. Para refazer, apague `artifacts/feature_study/cluster3_raw/models_cache/`.
""")

py("""
def _novo(nome, seed=42):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    return {
        "LGBMRegressor": lambda: LGBMRegressor(random_state=seed, verbose=-1),
        "CatBoostRegressor": lambda: CatBoostRegressor(random_state=seed, verbose=0),
        "RandomForestRegressor": lambda: RandomForestRegressor(random_state=seed, n_jobs=-1),
    }[nome]()


MODELOS = {}
for s in SEASONS:
    caminho = CACHE / f"{s}_{CAMPEAO[s]}.joblib"
    if caminho.exists():
        MODELOS[s] = joblib.load(caminho)
        print(f"  {s}: {CAMPEAO[s]} (cache)")
    else:
        tr = treino[treino.season == s]
        print(f"  {s}: ajustando {CAMPEAO[s]} em {len(tr)} linhas...", flush=True)
        MODELOS[s] = _novo(CAMPEAO[s]).fit(tr[FEATURES], tr[ALVO])
        joblib.dump(MODELOS[s], caminho, compress=3)
print("pronto")
""")

md("""
## O que dá para escolher
""")

py("""
print("ESTAÇÕES:", ", ".join(sorted(teste.estacao.unique())))
print()
print("TRIMESTRES:", ", ".join(SEASONS), " (cada um testa num único mês: jan, abr, jul, out)")
print()
avaliaveis = sorted(blocos_das_novas(FEATURES))
print("FEATURES dos grupos 1 e 4:")
for v in avaliaveis:
    print(f"   {v:16s} {DESCRICAO.get(v, '')}")
print()
print("(qualquer outra variável da base também serve — ex.: 'msl_h', 'tp_h', 'rh_h')")
""")

md("""
## A função

`ver(estacao, trimestre, feature)` — e pronto. Os argumentos opcionais só
existem quando a janela padrão não serve.
""")

py("""
SEM_COR, COR = "#8a6fb0", "#2a7ab0"


def ver(estacao: str, trimestre: str, feature: str, ano: int | None = None,
        max_dias: int = 45, seed: int = 42, ax=None):
    \"\"\"Desenha a série de uma estação num trimestre, com e sem a feature.

    Sem `ano`, escolhe o trecho contínuo mais longo — a validação e o teste são
    blocos de mês, então uma janela arbitrária cairia em cima de buracos e o
    gráfico ligaria os dois lados com uma reta que não é dado.
    \"\"\"
    if feature not in BLOCOS:
        raise KeyError(f"{feature!r} não existe. Veja a lista acima.")

    g = teste[(teste.estacao == estacao) & (teste.season == trimestre)].sort_values("time")
    if g.empty:
        raise ValueError(f"{estacao} não tem dias de teste em {trimestre}")
    if ano is not None:
        g = g[g.time.dt.year == ano]
        if g.empty:
            raise ValueError(f"{estacao}/{trimestre} não tem {ano}. "
                             f"Anos: {sorted(teste[(teste.estacao == estacao) & (teste.season == trimestre)].time.dt.year.unique())}")
    else:
        bloco = (g.time.diff() > pd.Timedelta(days=1)).cumsum()
        g = g[bloco == bloco.value_counts().idxmax()]
    g = g.tail(max_dias)

    modelo, colunas = MODELOS[trimestre], BLOCOS[feature]
    p_ok = np.clip(modelo.predict(g[FEATURES]), 0, 80)
    emb = g[FEATURES].copy()
    emb[colunas] = g[colunas].to_numpy()[np.random.default_rng(seed).permutation(len(g))]
    p_sem = np.clip(modelo.predict(emb), 0, 80)

    proprio = ax is None
    if proprio:
        _, ax = plt.subplots(figsize=(14, 4.6))
    ax.plot(g.time, g[ALVO], "o-", color="#111", ms=4.5, lw=1.4, label="INMET (observado)", zorder=5)
    ax.plot(g.time, g[PROXY], "s--", color="#c44", ms=3.5, lw=1.1, alpha=0.85, label="ERA5 cru", zorder=2)
    ax.plot(g.time, p_ok, "^-", color=COR, ms=4.5, lw=1.4, label="modelo completo", zorder=4)
    ax.plot(g.time, p_sem, "v:", color=SEM_COR, ms=4.5, lw=1.4,
            label=f"modelo com `{feature}` embaralhada", zorder=3)
    ax.fill_between(g.time, p_ok, p_sem, color=SEM_COR, alpha=0.15, zorder=1)

    erro = lambda p: float(np.sqrt(np.mean((p - g[ALVO].to_numpy()) ** 2)))   # noqa: E731
    ax.set_title(f"{estacao} · {trimestre} · `{feature}` — {DESCRICAO.get(feature, '')}\\n"
                 f"{len(g)} dias ({g.time.min():%d/%m/%Y} a {g.time.max():%d/%m/%Y})  |  "
                 f"RMSE: ERA5 {erro(g[PROXY].to_numpy()):.2f} · completo {erro(p_ok):.2f} · "
                 f"embaralhada {erro(p_sem):.2f} m/s  |  "
                 f"contribuição média {np.abs(p_ok - p_sem).mean():.2f} m/s",
                 fontsize=10.5, loc="left")
    ax.set_ylabel("rajada máx. diária (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, ncol=4, loc="upper left")
    if proprio:
        plt.tight_layout(); plt.show()
    return g.assign(modelo=p_ok, modelo_sem=p_sem)
""")

md("""
## Use aqui

Troque os três valores e rode. A célula seguinte tem exemplos prontos.
""")

py("""
ESTACAO = "A834"
TRIMESTRE = "DJF"
FEATURE = "fg10"

_ = ver(ESTACAO, TRIMESTRE, FEATURE)
""")

md("""
### A mesma estação, as quatro estações do ano

Para ver se a feature pesa igual o ano todo.
""")

py("""
fig, axes = plt.subplots(4, 1, figsize=(14, 15), constrained_layout=True)
for ax, s in zip(axes, SEASONS):
    ver(ESTACAO, s, FEATURE, ax=ax)
fig.suptitle(f"{ESTACAO} — `{FEATURE}` nos quatro trimestres", fontsize=13)
plt.show()
""")

md("""
### A mesma fatia, features diferentes

Para comparar quem realmente segura a previsão. `fg10` costuma dominar; as
variáveis com gêmeo na base (`ws10`, `t2m`, `d2m`) aparecem fracas de propósito
— o modelo lê a mesma informação na cópia da base.
""")

py("""
COMPARAR = ["fg10", "ws100", "cape", "ws10"]

fig, axes = plt.subplots(len(COMPARAR), 1, figsize=(14, 4.2 * len(COMPARAR)),
                         constrained_layout=True)
for ax, f in zip(np.atleast_1d(axes), COMPARAR):
    ver(ESTACAO, TRIMESTRE, f, ax=ax)
fig.suptitle(f"{ESTACAO} · {TRIMESTRE} — mesma janela, features diferentes", fontsize=13)
plt.show()
""")

md("""
### Um ano específico

Quando quiser um episódio que você já conhece.
""")

py("""
anos = sorted(teste[(teste.estacao == ESTACAO) & (teste.season == TRIMESTRE)].time.dt.year.unique())
print(f"{ESTACAO}/{TRIMESTRE} tem: {anos}")

_ = ver(ESTACAO, TRIMESTRE, FEATURE, ano=anos[-1])
""")

md("""
### Tabela da fatia

`ver(...)` devolve o recorte com as duas previsões, caso queira olhar os números
ou exportar.
""")

py("""
d = ver(ESTACAO, TRIMESTRE, FEATURE)
plt.close()

print(d[["estacao", "time", ALVO, PROXY, "modelo", "modelo_sem"]]
      .rename(columns={ALVO: "INMET", PROXY: "ERA5"})
      .assign(contribuição=lambda x: (x.modelo - x.modelo_sem).abs())
      .round(2).head(15).to_string(index=False))
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
DESTINO.parent.mkdir(exist_ok=True)
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
