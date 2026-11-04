"""Análises pedidas na revisão do documento 15 (sem treinar nada; usa as previsões já gravadas).

    uv run python scripts/revisao_metricas.py

1. Limiares P95/P99 de extremo FIXADOS NO TREINO de cada estação (INMET), aplicados ao teste: POD, FAR e CSI.
2. Ablação do ajuste de quantis: cada modelo com e sem o ajuste (RMSE, viés, viés do P99, POD/FAR/CSI).
3. Significância: bootstrap por blocos (ano, mês) do teste para a diferença de RMSE e Wilcoxon pareado por estação.

Saída: `artifacts/feature_study/global/revisao/*.csv` e as tabelas no terminal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parents[1]
G = RAIZ / "artifacts/feature_study/global"
OUT = G / "revisao"
TARGET = "daily_wind_gust_max"
CLS = [1, 2, 3, 4, 5, 12]
MIN_TREINO, MIN_AVAL = 365, 300      # dias mínimos para a estação entrar nas métricas de extremo
MODS = [("ml", "B", "ML local"), ("ml", "G", "ML global por trimestre"), ("ml", "GA", "ML global anual"),
        ("ls", "B", "LSTM local"), ("ls", "G", "LSTM global por trimestre"), ("ls", "GA", "LSTM global anual")]


def carregar() -> pd.DataFrame:
    base = pd.read_parquet(G / "base_global.parquet", columns=["row_id", "estacao", "cluster_id", "time", "_split",
                                                              "janela_ok", TARGET, "era5_gust_max"])
    d = base[base.janela_ok].drop(columns="janela_ok").rename(columns={TARGET: "inmet", "era5_gust_max": "ERA5"})
    prev = {"ml": pd.read_parquet(G / "previsoes_ml.parquet").set_index("row_id"),
            "ls": pd.read_parquet(G / "previsoes_lstm.parquet").set_index("row_id")}
    for fam, ch, nome in MODS:
        for suf, rot in (("eqm", nome), ("bruto", f"{nome} | sem ajuste")):
            v = np.full(len(d), np.nan, "float32")
            for cl in CLS:
                k = (d.cluster_id == cl).to_numpy()
                col = f"B{cl}_{suf}" if ch == "B" else f"{ch}_{suf}"
                v[k] = prev[fam][col].reindex(d.row_id[k]).to_numpy()
            d[rot] = v
    d["bloco"] = d.time.dt.to_period("M")
    return d.reset_index(drop=True)


def limiares(d: pd.DataFrame) -> pd.DataFrame:
    """P95 e P99 do INMET de cada estação no TREINO (climatologia de referência, fora da avaliação)."""
    tr = d[d._split == "train"].groupby("estacao").inmet
    return pd.DataFrame({"n_treino": tr.size(), "P95": tr.quantile(0.95), "P99": tr.quantile(0.99)})


def contagem(s: pd.DataFrame, col: str, lim: pd.Series) -> dict:
    ev, al = s.inmet >= lim, s[col] >= lim
    a, b, c = int((ev & al).sum()), int((~ev & al).sum()), int((ev & ~al).sum())   # acerto, falso alarme, perda
    return {"eventos": a + c, "POD": a / max(a + c, 1), "FAR": b / max(a + b, 1), "CSI": a / max(a + b + c, 1)}


def tabela_extremos(av: pd.DataFrame, lim: pd.DataFrame, versoes: list[str]) -> pd.DataFrame:
    ok_est = lim.index[(lim.n_treino >= MIN_TREINO)]
    n_av = av.groupby("estacao").size()
    ok_est = ok_est.intersection(n_av.index[n_av >= MIN_AVAL])
    a = av[av.estacao.isin(ok_est)]
    lin = []
    for q in ("P95", "P99"):
        lq = a.estacao.map(lim[q])
        for cl in CLS:
            k = a.cluster_id == cl
            for v in versoes:
                lin.append({"limiar": q, "cluster": cl, "versao": v, "estacoes": a[k].estacao.nunique(),
                            **contagem(a[k], v, lq[k])})
    return pd.DataFrame(lin)


def rmse(p, y):
    return float(np.sqrt(np.nanmean((np.asarray(p) - np.asarray(y)) ** 2)))


def ablacao(av: pd.DataFrame) -> pd.DataFrame:
    lin = []
    for cl in CLS:
        s = av[av.cluster_id == cl]
        for _, _, nome in MODS:
            for v in (f"{nome} | sem ajuste", nome):
                p99 = s.groupby("estacao").apply(lambda x, v=v: x[v].quantile(0.99) - x.inmet.quantile(0.99))
                lin.append({"cluster": cl, "versao": v, "RMSE": rmse(s[v], s.inmet), "vies": float((s[v] - s.inmet).mean()),
                            "vies_P99_estacao": float(p99.mean()), "desvio_rel": float(s[v].std() / s.inmet.std())})
    return pd.DataFrame(lin)


def significancia(av: pd.DataFrame, pares: list[tuple[str, str]], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Diferença de RMSE (A − B; negativo = A melhor). IC 95% por bootstrap de blocos (ano, mês) e Wilcoxon
    pareado sobre o MSE de cada estação (estações com ≥ MIN_AVAL dias)."""
    rng = np.random.default_rng(seed)
    lin = []
    for cl in CLS:
        s = av[av.cluster_id == cl]
        for a_, b_ in pares:
            s2 = s.dropna(subset=[a_, b_])
            ea, eb = (s2[a_] - s2.inmet) ** 2, (s2[b_] - s2.inmet) ** 2
            g = pd.DataFrame({"bl": s2.bloco, "ea": ea, "eb": eb}).groupby("bl").agg(["sum", "count"])
            sa, sb, n = g[("ea", "sum")].to_numpy(), g[("eb", "sum")].to_numpy(), g[("ea", "count")].to_numpy()
            idx = rng.integers(0, len(n), size=(n_boot, len(n)))
            dif = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
            pe = pd.DataFrame({"est": s2.estacao, "ea": ea, "eb": eb}).groupby("est").agg(
                n=("ea", "size"), ea=("ea", "mean"), eb=("eb", "mean"))
            pe = pe[pe.n >= MIN_AVAL]
            w = wilcoxon(pe.ea, pe.eb) if len(pe) >= 5 else None
            lin.append({"cluster": cl, "A": a_, "B": b_, "dif_RMSE": rmse(s2[a_], s2.inmet) - rmse(s2[b_], s2.inmet),
                        "IC95_inf": float(np.percentile(dif, 2.5)), "IC95_sup": float(np.percentile(dif, 97.5)),
                        "blocos": len(n), "estacoes_wilcoxon": len(pe),
                        "A_melhor_em_estacoes": int((pe.ea < pe.eb).sum()),
                        "p_wilcoxon": float(w.pvalue) if w is not None else np.nan})
    return pd.DataFrame(lin)


def main() -> None:
    d = carregar()
    OUT.mkdir(parents=True, exist_ok=True)
    lim = limiares(d)
    lim.to_csv(OUT / "limiares_treino.csv")
    teste = d[d._split == "test"]
    nomes = ["ERA5", *[n for _, _, n in MODS]]

    ext = tabela_extremos(teste, lim, nomes)
    ext.to_csv(OUT / "extremos_limiar_treino.csv", index=False)
    for q in ("P95", "P99"):
        for met in ("POD", "FAR", "CSI"):
            print(f"\n{met} {q} (limiar do treino; teste 2000–2024)")
            print(ext[ext.limiar == q].pivot(index="versao", columns="cluster", values=met).reindex(nomes).round(2).to_string())
    print("\nestações e eventos P99 por cluster:")
    print(ext[(ext.limiar == "P99") & (ext.versao == "ERA5")].set_index("cluster")[["estacoes", "eventos"]].to_string())

    ab = ablacao(teste)
    ab.to_csv(OUT / "ablacao_ajuste_quantis.csv", index=False)
    ab_ext = tabela_extremos(teste, lim, [v for _, _, n in MODS for v in (f"{n} | sem ajuste", n)])
    ab_ext.to_csv(OUT / "ablacao_extremos.csv", index=False)
    print("\nAblação (teste 2000–2024), média dos 6 clusters:")
    m1 = ab.groupby("versao")[["RMSE", "vies", "vies_P99_estacao", "desvio_rel"]].mean()
    m2 = ab_ext[ab_ext.limiar == "P99"].groupby("versao")[["POD", "FAR", "CSI"]].mean()
    print(m1.join(m2).round(2).to_string())
    print("\nRMSE por cluster, sem ajuste × com ajuste:")
    print(ab.pivot(index="versao", columns="cluster", values="RMSE").round(2).to_string())

    pares = [("LSTM global anual", "ERA5"), ("LSTM global anual", "ML global anual"), ("ML global anual", "ERA5"),
             ("LSTM global anual", "ML local")]
    sg = significancia(teste, pares)
    sg.to_csv(OUT / "significancia_rmse.csv", index=False)
    print("\nSignificância (teste 2000–2024), A − B em RMSE:")
    print(sg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
