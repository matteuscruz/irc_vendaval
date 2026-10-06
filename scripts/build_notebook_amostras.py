"""Gera `notebooks/feature_study_grupos/anteriores/raw_horario/amostras_melhor_mediano_pior.ipynb`.

Dez amostras de série temporal em cada faixa de desempenho. Reaproveita o cache
de modelos criado por `visualizacao_rapida_series.ipynb`, então roda em segundos
se aquele já tiver sido executado uma vez.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DESTINO = Path("notebooks/feature_study_grupos/anteriores/raw_horario/amostras_melhor_mediano_pior.ipynb")

nb = nbf.v4.new_notebook()
c: list = []
md = lambda t: c.append(nbf.v4.new_markdown_cell(t.strip()))      # noqa: E731
py = lambda t: c.append(nbf.v4.new_code_cell(t.strip()))          # noqa: E731

md("""
# Amostras de série temporal — melhores, medianos e piores casos

Dez amostras em cada faixa, para ver o comportamento do modelo em vez de só a
média dele.

**O que é um "caso":** uma **janela de teste** — uma combinação
`(estação, trimestre, ano)`. Como cada trimestre testa num único mês
(jan/abr/jul/out), uma janela é cerca de um mês de uma estação num ano. Essa é a
unidade que dá amostras suficientes: com apenas 9 estações não haveria dez
"melhores estações", mas há centenas de janelas.

**Ranking por RMSE** da previsão contra o observado, dentro da janela. O R² e o
viés saem no título de cada painel, porque o RMSE sozinho engana: uma janela
calma tem RMSE baixo sem mérito do modelo.

Três linhas por painel: **INMET** (observado), **ERA5 cru** e **modelo completo**
(585 features, campeão do trimestre).
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
SEASONS = ("DJF", "MAM", "JJA", "SON")
ALVO, PROXY = "daily_wind_gust_max", "wind_mag_max"
MIN_DIAS = 15          # janela menor que isto não sustenta uma métrica

meta = json.loads((BASE / "data/meta.json").read_text())
FEATURES = meta["base_features"] + meta["new_features"]
CAMPEAO = {s: v[0] for s, v in
           json.loads((BASE / "summary/top_models.json").read_text())["by_season"].items()}
teste = pd.read_parquet(BASE / "data/test.parquet")

if not CACHE.exists() or not list(CACHE.glob("*.joblib")):
    raise FileNotFoundError(
        "Cache de modelos ausente. Execute `visualizacao_rapida_series.ipynb` uma vez "
        "— ele ajusta os 4 campeões e os guarda em models_cache/."
    )

partes = []
for s in SEASONS:
    modelo = joblib.load(CACHE / f"{s}_{CAMPEAO[s]}.joblib")
    g = teste[teste.season == s]
    partes.append(g.assign(modelo=np.clip(modelo.predict(g[FEATURES]), 0, 80)))
pred = pd.concat(partes, ignore_index=True).sort_values(["estacao", "time"])
pred["ano"] = pred.time.dt.year

print(f"{len(pred)} predições de teste | campeões: {CAMPEAO}")
""")

md("""
## As janelas e seu desempenho
""")

py("""
def metricas(g: pd.DataFrame) -> pd.Series:
    y, p, e = g[ALVO].to_numpy(), g.modelo.to_numpy(), g[PROXY].to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    return pd.Series({
        "n": len(g),
        "rmse": float(np.sqrt(((p - y) ** 2).mean())),
        "rmse_era5": float(np.sqrt(((e - y) ** 2).mean())),
        "r2": float(1 - ((p - y) ** 2).sum() / sst) if sst > 0 else np.nan,
        "vies": float((p - y).mean()),
        "obs_max": float(y.max()),
    })


janelas = (pred.groupby(["estacao", "season", "ano"])
           .apply(metricas, include_groups=False)
           .reset_index())
janelas = janelas[janelas.n >= MIN_DIAS].sort_values("rmse").reset_index(drop=True)
janelas["ganho_sobre_era5"] = janelas.rmse_era5 - janelas.rmse

print(f"{len(janelas)} janelas com pelo menos {MIN_DIAS} dias")
print(janelas[["n", "rmse", "rmse_era5", "r2", "vies", "obs_max"]].describe().round(2).to_string())
""")

py("""
N = 10
meio = len(janelas) // 2
FAIXAS = {
    "MELHORES": janelas.head(N),
    "MEDIANOS": janelas.iloc[meio - N // 2: meio - N // 2 + N],
    "PIORES": janelas.tail(N).iloc[::-1],
}
for nome, d in FAIXAS.items():
    print(f"{nome}: RMSE de {d.rmse.min():.2f} a {d.rmse.max():.2f} m/s | "
          f"R² mediano {d.r2.median():.2f} | rajada máx. observada mediana {d.obs_max.median():.1f} m/s")
""")

md("""
## Os painéis

Cada figura traz dez janelas. O título de cada painel tem `n` dias, RMSE do
modelo **contra** o do ERA5 cru, R² e viés.
""")

py("""
def painel(ax, linha):
    g = pred[(pred.estacao == linha.estacao) & (pred.season == linha.season)
             & (pred.ano == linha.ano)].sort_values("time")
    ax.plot(g.time, g[ALVO], "o-", color="#111", ms=3.5, lw=1.2, label="INMET", zorder=4)
    ax.plot(g.time, g[PROXY], "s--", color="#c44", ms=3, lw=1, alpha=0.85, label="ERA5 cru", zorder=2)
    ax.plot(g.time, g.modelo, "^-", color="#2a7ab0", ms=3.5, lw=1.2, label="modelo", zorder=3)
    ax.fill_between(g.time, g[PROXY], g[ALVO], color="#c44", alpha=0.08, zorder=1)
    ax.set_title(f"{linha.estacao} · {linha.season} {int(linha.ano)} · {int(linha.n)} dias\\n"
                 f"RMSE {linha.rmse:.2f} (ERA5 {linha.rmse_era5:.2f}) · "
                 f"R² {linha.r2:.2f} · viés {linha.vies:+.2f} · máx obs {linha.obs_max:.1f} m/s",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25)


def grade(d, titulo, cor):
    fig, axes = plt.subplots(5, 2, figsize=(16, 17), constrained_layout=True)
    for ax, (_, linha) in zip(axes.ravel(), d.iterrows()):
        painel(ax, linha)
    for ax in axes.ravel()[len(d):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.supylabel("rajada máx. diária (m/s)", fontsize=10)
    fig.suptitle(titulo, fontsize=14, color=cor)
    plt.show()
""")

md("""
### Melhores casos — onde o modelo acerta
""")

py("""
grade(FAIXAS["MELHORES"], "10 MELHORES janelas (menor RMSE)", "#1a7a3a")
""")

md("""
### Casos medianos — o comportamento típico
""")

py("""
grade(FAIXAS["MEDIANOS"], "10 janelas MEDIANAS (RMSE em torno da mediana)", "#8a6a1a")
""")

md("""
### Piores casos — onde o modelo falha
""")

py("""
grade(FAIXAS["PIORES"], "10 PIORES janelas (maior RMSE)", "#a32020")
""")

md("""
## O que separa os melhores dos piores

A comparação abaixo evita a leitura ingênua de que "o modelo é bom nuns lugares e
ruim noutros". Antes de concluir isso, vale ver se a diferença não é só a
intensidade do que aconteceu na janela.
""")

py("""
comp = pd.DataFrame({nome: d[["n", "rmse", "rmse_era5", "ganho_sobre_era5", "r2", "vies", "obs_max"]].median()
                     for nome, d in FAIXAS.items()}).T
print(comp.round(2).to_string())
print()
print("distribuição por trimestre e estação:")
for nome, d in FAIXAS.items():
    print(f"  {nome:9s} trimestres {dict(d.season.value_counts())} | "
          f"estações {dict(d.estacao.value_counts())}")
""")

py("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
cores = {"MELHORES": "#1a7a3a", "MEDIANOS": "#8a6a1a", "PIORES": "#a32020"}

for ax, (col, rot) in zip(axes, [("obs_max", "rajada máxima observada na janela (m/s)"),
                                 ("rmse", "RMSE do modelo (m/s)"),
                                 ("ganho_sobre_era5", "quanto o modelo ganha do ERA5 (m/s)")]):
    ax.boxplot([FAIXAS[k][col] for k in FAIXAS], tick_labels=list(FAIXAS),
               patch_artist=True,
               boxprops=dict(facecolor="#dfe7ee"), medianprops=dict(color="black"))
    for i, k in enumerate(FAIXAS, start=1):
        ax.scatter(np.full(len(FAIXAS[k]), i) + np.random.default_rng(1).normal(0, 0.04, len(FAIXAS[k])),
                   FAIXAS[k][col], color=cores[k], s=22, zorder=3, edgecolor="black", linewidth=0.4)
    ax.set_title(rot, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
fig.suptitle("As três faixas comparadas — a dificuldade da janela explica o RMSE?", fontsize=12)
plt.show()
""")

md("""
### Leitura

Se a caixa da esquerda mostrar que as janelas "piores" são justamente as de
rajada mais forte, então o RMSE alto é **dificuldade da janela**, não falha
localizada do modelo — e a métrica honesta passa a ser a da direita: quanto o
modelo ganha do ERA5 naquela janela.

Uma janela com RMSE baixo e ganho ~0 sobre o ERA5 não é um acerto do modelo: é um
mês calmo em que o ERA5 já bastava.
""")

py("""
melhor_ganho = janelas.nlargest(N, "ganho_sobre_era5")
print(f"As {N} janelas em que o modelo mais ganha do ERA5 "
      "(critério diferente do RMSE: aqui o mérito é do modelo, não do mês calmo)")
print(melhor_ganho[["estacao", "season", "ano", "n", "rmse", "rmse_era5",
                    "ganho_sobre_era5", "obs_max"]].round(2).to_string(index=False))
""")

py("""
grade(melhor_ganho, f"{N} janelas de MAIOR GANHO sobre o ERA5", "#1a5a7a")
""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
DESTINO.parent.mkdir(exist_ok=True)
nbf.write(nb, DESTINO)
print(f"{DESTINO} — {len(c)} células")
