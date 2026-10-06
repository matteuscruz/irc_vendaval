"""Mapa espacial e séries temporais do ENSAIO local do desenho RAW.

Cada figura carrega no título o que ela é e o que ela não é: validação (não
teste), um modelo (não os 39), sem IC. Sem isso, um PNG solto vira "resultado
do estudo" na primeira vez que alguém o abrir fora de contexto.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error as rmse

D = Path("/tmp/claude-1000/-media-matteuscruz-Projects-Entrep-irc-vendaval/"
         "ce9859b2-86d4-435b-a5fd-cfdb791fbbdf/scratchpad")
OUT = Path("artifacts/feature_study/cluster3_raw/plots")
OUT.mkdir(parents=True, exist_ok=True)
RODAPE = ("Estudo RAW horário, cluster 3 — conjunto de TESTE (jan/abr/jul/out, 2000-2024), "
          "campeão do trimestre no arm `full`, seed 42. Controles negativos ≈ 0.")

# Predições REAIS do estudo (units/full/preds__*.parquet, seed 42), e não mais
# as da sondagem local. Cada (estação, dia) tem uma linha por modelo do top-5
# do trimestre; o campeão de cada trimestre é quem vai para o gráfico.
bruto = pd.concat([pd.read_parquet(f) for f in glob.glob(str(D / "preds" / "*.parquet"))],
                  ignore_index=True)
campeao = json.loads((D / "top_models.json").read_text())["by_season"]
bruto["campeao"] = bruto.season.map({k: v[0] for k, v in campeao.items()})
p = bruto[(bruto.model == bruto.campeao) & (bruto.arm == "full")].copy()
p = p.rename(columns={"y_true": "daily_wind_gust_max", "era5_proxy": "wind_mag_max",
                      "y_pred": "raw_base"})
print("campeão por trimestre:", {k: v[0] for k, v in campeao.items()})
print(f"{len(p)} predições de TESTE, {p.estacao.nunique()} estações")


def metricas(g: pd.DataFrame, col: str) -> pd.Series:
    y, x = g.daily_wind_gust_max.to_numpy(), g[col].to_numpy()
    t = y >= np.percentile(y, 90)
    return pd.Series({
        "n": len(g), "R2": r2_score(y, x), "RMSE": rmse(y, x),
        "RMSE_P90": rmse(y[t], x[t]), "Bias_P90": (x[t] - y[t]).mean(),
        "corr": np.corrcoef(x, y)[0, 1],
    })


# ── 1. Mapa espacial ────────────────────────────────────────────────────────

por_est = {
    nome: p.groupby("estacao").apply(lambda g: metricas(g, c), include_groups=False)
    for nome, c in (("era5", "wind_mag_max"), ("raw", "raw_base"))
}
coord = p.groupby("estacao")[["latitude", "longitude"]].first()
mapa = coord.join(por_est["era5"].add_prefix("era5_")).join(por_est["raw"].add_prefix("raw_"))
mapa["d_corr"] = mapa.raw_corr - mapa.era5_corr
mapa["d_bias"] = mapa.era5_Bias_P90.abs() - mapa.raw_Bias_P90.abs()   # >0 = viés reduzido

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
mask = gpd.read_file("artifacts/cluster_masks/cluster_3_mask.shp")

paineis = [
    ("corr(ERA5 cru, observado)", mapa.era5_corr, "RdYlGn", 0.5, 0.9, "{:.2f}"),
    ("corr(modelo RAW, observado)", mapa.raw_corr, "RdYlGn", 0.5, 0.9, "{:.2f}"),
    ("viés |P90| removido (m/s)", mapa.d_bias, "Blues", 0, None, "{:+.1f}"),
]
for ax, (titulo, valores, cmap, vmin, vmax, fmt) in zip(axes, paineis):
    mask.plot(ax=ax, facecolor="#eef2f5", edgecolor="#9fb3c8", linewidth=0.8)
    sc = ax.scatter(mapa.longitude, mapa.latitude, c=valores, cmap=cmap,
                    vmin=vmin, vmax=vmax if vmax is not None else valores.max(),
                    s=260, edgecolor="black", linewidth=0.8, zorder=3)
    # A801 e B807 ficam a ~0,2° uma da outra: sem deslocar, os rótulos se
    # sobrepõem e nenhum dos dois é legível.
    desloca = {"B807": (0, -26), "A801": (0, 14)}
    for est, row in mapa.iterrows():
        ax.annotate(f"{est}\n{fmt.format(valores[est])}",
                    (row.longitude, row.latitude), textcoords="offset points",
                    xytext=desloca.get(est, (0, 14)), ha="center", fontsize=8.5, zorder=4)
    plt.colorbar(sc, ax=ax, shrink=0.78)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

fig.suptitle("Cluster 3 — 9 estações com alvo INMET em 2000–2024\n"
             "Correlação ERA5 × modelo e redução de viés na cauda (P90)", fontsize=13)
fig.text(0.5, -0.02, RODAPE, ha="center", fontsize=9, style="italic", color="#555")
fig.savefig(OUT / "mapa_espacial.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"→ {OUT / 'mapa_espacial.png'}")


# ── 2. Séries temporais: melhor, mediano e pior caso ────────────────────────
# "Caso" = estação. O critério é o R² do modelo RAW, e o `n` sai no título:
# B807 tem 1/12 das observações de A801 e lideraria qualquer ranking de pior
# caso por escassez de amostra, não por erro do modelo.

ranking = mapa.sort_values("raw_R2", ascending=False)
escolhidas = [
    ("melhor", ranking.index[0]),
    ("mediano", ranking.index[len(ranking) // 2]),
    ("pior", ranking.index[-1]),
]


def bloco_mais_longo(g: pd.DataFrame) -> pd.DataFrame:
    """Maior trecho de dias CONSECUTIVOS da estação.

    O teste são os meses jan/abr/jul/out de cada ano, então a série tem buracos de
    meses. Ligar os dois lados de um buraco com uma reta desenha dado que não
    existe — e numa série de rajada essa reta atravessa justamente a faixa de
    valores que interessa.
    """
    g = g.sort_values("time")
    quebra = g.time.diff() > pd.Timedelta(days=1)
    g = g.assign(_bloco=quebra.cumsum())
    maior = g._bloco.value_counts().idxmax()
    return g[g._bloco == maior].drop(columns="_bloco")


fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
for ax, (rotulo, est) in zip(axes, escolhidas):
    janela = bloco_mais_longo(p[p.estacao == est])
    ax.plot(janela.time, janela.daily_wind_gust_max, "o-", color="#111", ms=4,
            lw=1.2, label="observado (INMET)", zorder=3)
    ax.plot(janela.time, janela.wind_mag_max, "s--", color="#c44", ms=3.5, lw=1,
            alpha=0.8, label="ERA5 cru", zorder=2)
    ax.plot(janela.time, janela.raw_base, "^-", color="#2a7ab0", ms=4, lw=1.2,
            label="modelo (arm full, 585 col)", zorder=4)
    ax.fill_between(janela.time, janela.wind_mag_max, janela.daily_wind_gust_max,
                    color="#c44", alpha=0.10, zorder=1)
    r = mapa.loc[est]
    periodo = f"{janela.time.min():%d/%m/%Y} a {janela.time.max():%d/%m/%Y}"
    ax.set_title(f"{rotulo.upper()} caso — {est}  |  R²={r.raw_R2:.2f}  "
                 f"RMSE_P90={r.raw_RMSE_P90:.2f} m/s  |  {len(janela)} dias consecutivos "
                 f"({periodo})  |  n total de teste = {int(r.era5_n)}",
                 fontsize=10.5, loc="left")
    ax.set_ylabel("rajada máx. diária (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=3)

fig.suptitle("Série temporal por estação — melhor, mediano e pior caso (R² do modelo)\n"
             "trecho contínuo mais longo do teste; a área vermelha é o que o ERA5 deixa de ver",
             fontsize=12.5)
fig.text(0.5, -0.015, RODAPE, ha="center", fontsize=9, style="italic", color="#555")
fig.savefig(OUT / "series_temporais.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"→ {OUT / 'series_temporais.png'}")

print()
print(mapa[["n_x" if "n_x" in mapa else "era5_n", "era5_corr", "raw_corr", "d_corr",
            "era5_Bias_P90", "raw_Bias_P90", "d_bias", "raw_R2"]].round(3).to_string())
