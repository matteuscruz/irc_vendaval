"""Cobertura temporal das estações do cluster 3 — por que só 9 entram no estudo.

O arquivo horário do ERA5 cobre 21 estações do cluster 3, mas o ERA5 é modelo:
existe em qualquer ponto e em qualquer data. Quem limita é a OBSERVAÇÃO. Este
gráfico mostra, mês a mês, onde há rajada medida pelo INMET — e deixa visível
que 12 das 21 estações só começam a reportar depois da janela do estudo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "artifacts/feature_study/cluster3_raw/plots/cobertura_estacoes.png"
INICIO_ESTUDO, FIM_ESTUDO = pd.Timestamp("2000-01-01"), pd.Timestamp("2024-12-31")
ALVO = "daily_wind_gust_max"

USADA, DESCARTADA = "#2a7ab0", "#c9542e"


def main() -> None:
    with xr.open_dataset(RAIZ / "dataset/raw/test_cluster_3_hourly.nc") as h:
        do_cluster = [str(e) for e in h.estacao.values]

    with xr.open_dataset(RAIZ / "dataset/raw/INMET_Stratified.nc") as inmet:
        presentes = [e for e in do_cluster if e in set(map(str, inmet.estacao.values))]
        sub = inmet.sel(estacao=presentes)[ALVO]
        tempo = pd.DatetimeIndex(sub.time.values)
        valores = np.asarray(sub.values, dtype=float)          # (time, estacao)

    # Presença por (estação, mês): quantos dias com medição.
    mes = tempo.to_period("M")
    por_mes = (pd.DataFrame(np.isfinite(valores), index=mes, columns=presentes)
               .groupby(level=0).sum())

    # Um resumo por estação, para ordenar e rotular.
    resumo = pd.DataFrame({
        "n_dias": np.isfinite(valores).sum(axis=0),
        "primeiro": [tempo[np.flatnonzero(np.isfinite(v))[0]] if np.isfinite(v).any() else pd.NaT
                     for v in valores.T],
    }, index=presentes)
    # "Usada" = tem observação dentro da janela do estudo. É o critério real:
    # não foi escolha, foi o que sobrou depois de exigir alvo no período.
    dentro = (mes >= INICIO_ESTUDO.to_period("M")) & (mes <= FIM_ESTUDO.to_period("M"))
    resumo["dias_no_estudo"] = por_mes[dentro.tolist() if hasattr(dentro, "tolist") else dentro].sum()
    resumo["usada"] = resumo.dias_no_estudo > 0
    resumo = resumo.sort_values(["usada", "primeiro"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    ax.axvspan(INICIO_ESTUDO, FIM_ESTUDO, color="#f3f6f9", zorder=0)
    ax.axvline(INICIO_ESTUDO, color="#8fa3b5", lw=1, ls="--", zorder=1)
    ax.axvline(FIM_ESTUDO, color="#8fa3b5", lw=1, ls="--", zorder=1)

    for y, est in enumerate(resumo.index):
        cor = USADA if resumo.usada[est] else DESCARTADA
        meses = por_mes.index[por_mes[est] > 0]
        if len(meses) == 0:
            continue
        # Um traço por mês com medição: os buracos ficam à vista, e é por isso
        # que o gráfico não desenha apenas a primeira e a última data.
        ax.barh([y] * len(meses), width=31, left=[m.to_timestamp() for m in meses],
                height=0.62, color=cor, edgecolor="none", zorder=3)
        ax.text(FIM_ESTUDO + pd.Timedelta(days=200), y,
                f"{int(resumo.dias_no_estudo[est]):>5d} dias",
                va="center", fontsize=8.5, color=cor, fontweight="bold" if resumo.usada[est] else "normal")

    ax.set_yticks(range(len(resumo)))
    ax.set_yticklabels(resumo.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("1999-06-01"), pd.Timestamp("2027-06-01"))
    ax.set_xlabel("ano")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    n_ok = int(resumo.usada.sum())
    ax.set_title(
        f"Cobertura do INMET nas {len(resumo)} estações do cluster 3\\n"
        f"cada traço é um mês com rajada medida — a faixa clara é a janela do estudo "
        f"(2000–2024)", fontsize=12.5)
    ax.legend(handles=[
        Patch(facecolor=USADA, label=f"{n_ok} usadas — têm observação na janela"),
        Patch(facecolor=DESCARTADA, label=f"{len(resumo) - n_ok} fora — só reportam a partir de 2025"),
    ], loc="lower left", fontsize=9.5)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINO, dpi=150, bbox_inches="tight")
    print(f"→ {DESTINO}")

    print(f"\n{n_ok} de {len(resumo)} estações têm observação em 2000–2024.")
    print(resumo.assign(primeiro=resumo.primeiro.dt.date)
          [["primeiro", "n_dias", "dias_no_estudo", "usada"]].to_string())


if __name__ == "__main__":
    main()
