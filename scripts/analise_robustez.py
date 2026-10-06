"""Análises de robustez do estudo de features, sobre resíduos já gravados.

Custo de nuvem ZERO: nada é ajustado. Lê `units/<tag>/resid__*.parquet` e
`data/test.parquet` de um estudo (padrão: `cluster3_groups_modal`) e grava, em
`<estudo>/robustez/`, as tabelas abaixo.

  desempenho_cauda.csv     RMSE/viés em P90/P95/P99 e pinball, por arm (+ ERA5)
  efeitos_esquemas.csv     o MESMO efeito sob blocos ano-mês, evento, móveis e estacionário
  esquemas_resumo.csv      largura do IC e concordância de veredito entre esquemas
  diebold_mariano.csv      DM (HAC dentro de sequências contíguas) por comparação
  excedencia.csv           POD/FAR/CSI de excedência de P90/P95/P99
  estratos.csv             erro por estação, trimestre, regime e evento
  regimes_centroides.csv   o que cada regime sinótico representa
  eventos_extremos.csv     os maiores eventos e o que ERA5 e modelos previram

    python scripts/analise_robustez.py [--study cluster3_groups_modal]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.feature_study import robustness as rb  # noqa: E402
from src.feature_study.analysis import load_arms  # noqa: E402
from src.feature_study.config import MAIN_TAGS  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402

REF = "era5_gust_max"
KEY_ARMS = ("base", "full")
REGIME_VARS = ("w850", "vort850", "mslp_tend_3h", "grad_mslp_hpa_100km", "t850", "omega700", "cape")
N_BOOT = 2000


def _preparar(estudo: Path):
    dados = estudo / "data"
    if not (dados / "test.parquet").exists():
        raise FileNotFoundError(f"{dados / 'test.parquet'} ausente — baixe o teste do estudo")
    arms = load_arms(dados)
    tags = list(MAIN_TAGS)
    return (*rb.load_study(estudo, dados, tags, arms), arms, tags)


ERA5_ARM = "__era5__"
NOME_ARM = {ERA5_ARM: "ERA5 (rajada bruta)", "base": "modelo, arm base", "full": "modelo, arm full"}


def injeta_era5(resid, test, tags, models, comps):
    """Acrescenta a rajada ERA5 como um "arm" sem modelo, com a MESMA cobertura de
    cada (seed, modelo).

    É a correção de comparabilidade: cada modelo só existe nos trimestres em que foi
    eleito, e uma média de modelos sobre linhas diferentes não se compara com o ERA5
    avaliado no teste inteiro. Avaliado nas mesmas linhas de cada modelo, o ERA5 passa
    a ter a mesma ponderação por trimestre — e o ganho do modelo sobre ele ganha
    intervalo de confiança pelo mesmo bootstrap pareado."""
    y = test[TARGET_VAR].to_numpy(float)
    era = test[REF].to_numpy(float) - y
    for t in tags:
        for m in models:
            base = resid[(t, "base", m)]
            resid[(t, ERA5_ARM, m)] = np.where(np.isfinite(base), era, np.nan)
    return list(comps) + [("base_vs_era5", ERA5_ARM, "base"), ("full_vs_era5", ERA5_ARM, "full")]


def tabela_desempenho(resid, test, dias, tags, models, arms) -> pd.DataFrame:
    """Métricas pontuais por arm, médias sobre modelos e seeds — e a referência ERA5."""
    y = test[TARGET_VAR].to_numpy(float)
    uns = np.ones((1, dias.n_days))
    linhas = []
    nomes = [a.name for a in arms] + [ERA5_ARM]
    for nome in nomes:
        acc: dict[str, list] = {}
        for t in tags:
            for m in models:
                e = resid[(t, nome, m)]
                mask = np.isfinite(e)
                if not mask.any():
                    continue
                met = rb.metrics_from_totals(uns @ rb.day_sums(e, y, mask, dias))
                for k, v in met.items():
                    acc.setdefault(k, []).append(float(v[0]))
        if acc:
            linhas.append({"arm": NOME_ARM.get(nome, nome), **{k: float(np.mean(v)) for k, v in acc.items()}})
    return pd.DataFrame(linhas)


def esquemas(resid, test, dias, comps, tags, models, labels_evento) -> pd.DataFrame:
    esquemas_ = {
        "mes (ano-mês)": (rb.counts_by_blocks(rb.month_labels(test, dias), N_BOOT), int(rb.month_labels(test, dias).max() + 1)),
        "evento": (rb.counts_by_blocks(labels_evento, N_BOOT), int(labels_evento.max() + 1)),
        "bloco móvel L=3": (rb.counts_moving_block(dias, 3, N_BOOT), None),
        "bloco móvel L=7": (rb.counts_moving_block(dias, 7, N_BOOT), None),
        "bloco móvel L=14": (rb.counts_moving_block(dias, 14, N_BOOT), None),
        "estacionário L=7": (rb.counts_stationary(dias, 7, N_BOOT), None),
    }
    partes = []
    for nome, (counts, n_units) in esquemas_.items():
        t0 = time.time()
        partes.append(rb.paired_effects(resid, test, dias, comps, tags, models, counts,
                                        scheme=nome, n_units=n_units))
        print(f"   esquema {nome:18s} {time.time() - t0:5.1f}s", flush=True)
    return pd.concat(partes, ignore_index=True)


def resumo_esquemas(ef: pd.DataFrame) -> pd.DataFrame:
    """Largura do IC relativa ao esquema ano-mês e se o veredito muda."""
    base = ef[ef.scheme == "mes (ano-mês)"].set_index(["comparison", "metric"])
    out = []
    for (esq), g in ef.groupby("scheme"):
        g = g.set_index(["comparison", "metric"])
        larg = (g.ci_hi - g.ci_lo) / (base.ci_hi - base.ci_lo)
        mesmo = (g.verdict == base.verdict.reindex(g.index))
        out.append({"scheme": esq, "largura_ic_relativa_mediana": float(larg.median()),
                    "largura_min": float(larg.min()), "largura_max": float(larg.max()),
                    "veredito_igual_ao_mes_%": float(100 * mesmo.mean())})
    return pd.DataFrame(out)


def dm_tabela(resid, test, dias, comps, tags, models) -> pd.DataFrame:
    linhas = []
    for name, worse, better in comps:
        d, usados = rb.daily_loss_differential(resid, test, dias, worse, better, tags, models)
        r = rb.diebold_mariano(d, dias.run_id)
        por_modelo = []
        for m in models:
            dm, _ = rb.daily_loss_differential(resid, test, dias, worse, better, tags, [m])
            rm = rb.diebold_mariano(dm, dias.run_id)
            if np.isfinite(rm["p"]):
                por_modelo.append(rm)
        linhas.append({
            "comparison": name, "dm_media_modelos": r["dm"], "p": r["p"], "n_dias": r["n_dias"],
            "dif_media_erro2": r["media"],
            "modelos_testados": len(por_modelo),
            "modelos_p<0.05_melhor": sum(1 for x in por_modelo if x["p"] < 0.05 and x["dm"] > 0),
            "modelos_p<0.05_pior": sum(1 for x in por_modelo if x["p"] < 0.05 and x["dm"] < 0),
        })
    return pd.DataFrame(linhas)


def _preds(resid, test, tags, models, arm):
    """Gera (previsão, máscara de cobertura) por (seed, modelo): pred = resíduo + observado."""
    y = test[TARGET_VAR].to_numpy(float)
    for t in tags:
        for m in models:
            e = resid[(t, arm, m)]
            mask = np.isfinite(e)
            if mask.any():
                yield t, m, np.where(mask, y + e, np.nan), mask


def excedencia(resid, test, tags, models) -> pd.DataFrame:
    y = test[TARGET_VAR].to_numpy(float)
    linhas = []
    for q in rb.TAIL_QS:
        lim = float(np.quantile(y, q))
        etiqueta = f"P{int(round(q * 100))}"
        for arm in (ERA5_ARM, *KEY_ARMS):
            acc: dict[str, list] = {}
            for _, _, p, mask in _preds(resid, test, tags, models, arm):
                # o limiar é o mesmo para todos; a cobertura (igual à do ERA5) define as linhas
                c = rb.contingency(y[mask], p[mask], lim)
                for k, v in c.items():
                    acc.setdefault(k, []).append(v)
            linhas.append({"limiar": etiqueta, "valor_m/s": lim, "previsor": NOME_ARM[arm],
                           **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    return pd.DataFrame(linhas)


def calibracao(resid, test, tags, models) -> pd.DataFrame:
    """Reliability por faixa de previsão, médio sobre modelos e seeds (mesma cobertura)."""
    y = test[TARGET_VAR].to_numpy(float)
    linhas = []
    for arm in (ERA5_ARM, *KEY_ARMS):
        acc: dict[str, list] = {}
        for _, _, p, mask in _preds(resid, test, tags, models, arm):
            r = rb.reliability(y[mask], p[mask])
            for _, linha in r.iterrows():
                acc.setdefault(linha["faixa"], []).append((linha["n"], linha["prev_media"], linha["obs_media"]))
        for faixa, v in acc.items():
            a = np.array(v)
            linhas.append({"previsor": NOME_ARM[arm], "faixa": faixa, "n": a[:, 0].mean(),
                           "prev_media": a[:, 1].mean(), "obs_media": a[:, 2].mean()})
    out = pd.DataFrame(linhas)
    out["obs_menos_prev"] = out["obs_media"] - out["prev_media"]
    return out


def regimes(test, dias, k: int = 4):
    """Regimes sinóticos por k-means sobre as variáveis do grupo 3 (e `cape`), por DIA.

    Usa só preditores — o alvo não entra —, então os rótulos servem para estratificar
    a avaliação sem vazar a resposta. São descritivos: o rótulo `R<i>` é numerado por
    `w850` crescente e o significado físico sai da tabela de centróides, não do nome."""
    faltam = [c for c in REGIME_VARS if c not in test.columns]
    if faltam:
        raise KeyError(f"variáveis de regime ausentes do teste: {faltam}")
    por_dia = pd.DataFrame({c: np.bincount(dias.row_day, weights=test[c].to_numpy(float), minlength=dias.n_days)
                            for c in REGIME_VARS})
    n = np.bincount(dias.row_day, minlength=dias.n_days)
    por_dia = por_dia.div(n, axis=0)
    Z = StandardScaler().fit_transform(por_dia)
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    cent = pd.DataFrame(StandardScaler().fit(por_dia).inverse_transform(km.cluster_centers_), columns=REGIME_VARS)
    ordem = np.argsort(cent["w850"].to_numpy())
    renum = {int(v): f"R{i + 1}" for i, v in enumerate(ordem)}
    rotulo_dia = np.array([renum[int(c)] for c in km.labels_])
    cent.index = [renum[i] for i in range(k)]
    cent = cent.sort_index()
    cent["dias"] = pd.Series(rotulo_dia).value_counts().reindex(cent.index).to_numpy()
    return rotulo_dia, cent


def estratos(resid, test, dias, tags, models, rotulos: dict[str, np.ndarray]) -> pd.DataFrame:
    y = test[TARGET_VAR].to_numpy(float)
    lim90 = float(np.quantile(y, 0.9))
    linhas = []

    def acumula(pred, mask, lab, nome, estr, acc):
        m = mask & (lab == estr)
        if m.sum() < 30:
            return
        e = pred[m] - y[m]
        cauda = y[m] >= lim90
        acc.setdefault((nome, estr), []).append((
            int(m.sum()), float(np.sqrt(np.mean(e ** 2))),
            float(np.sqrt(np.mean(e[cauda] ** 2))) if cauda.sum() >= 10 else np.nan,
            float(np.mean(e[cauda])) if cauda.sum() >= 10 else np.nan,
            int(cauda.sum())))

    for dim, lab in rotulos.items():
        acc: dict = {}
        for estr in np.unique(lab):
            for arm in (ERA5_ARM, *KEY_ARMS):
                for _, _, p, mask in _preds(resid, test, tags, models, arm):
                    acumula(p, mask, lab, NOME_ARM[arm], estr, acc)
        for (nome, estr), v in acc.items():
            a = np.array(v, dtype=float)
            linhas.append({"dimensao": dim, "estrato": estr, "previsor": nome,
                           "n_linhas": a[:, 0].mean(), "n_cauda": a[:, 4].mean(),
                           "rmse": np.nanmean(a[:, 1]), "rmse_p90": np.nanmean(a[:, 2]),
                           "bias_p90": np.nanmean(a[:, 3])})
    return pd.DataFrame(linhas)


def eventos_extremos(resid, test, dias, tags, models, labels, ev_info, topn: int = 15) -> pd.DataFrame:
    y = test[TARGET_VAR].to_numpy(float)
    preds = {arm: [p for _, _, p, _ in _preds(resid, test, tags, models, arm)] for arm in KEY_ARMS}
    pm = {arm: np.nanmean(np.vstack(v), axis=0) for arm, v in preds.items()}
    era = test[REF].to_numpy(float)
    linhas = []
    for _, ev in ev_info.iterrows():
        dia_ev = np.flatnonzero(labels == ev["evento"])
        rows = np.flatnonzero(np.isin(dias.row_day, dia_ev))
        i = rows[np.nanargmax(y[rows])]
        linhas.append({"inicio": ev["inicio"].date(), "fim": ev["fim"].date(), "dias": ev["dias"],
                       "estacoes_acima_P90": ev["pico_estacoes_acima"], "estacao_do_pico": test["estacao"].iloc[i],
                       "obs_max_m/s": y[i], "ERA5_m/s": era[i], "modelo_base_m/s": pm["base"][i],
                       "modelo_full_m/s": pm["full"][i]})
    t = pd.DataFrame(linhas).sort_values("obs_max_m/s", ascending=False)
    return t.head(topn).reset_index(drop=True), t


def figura(saida: Path) -> Path:
    """Resumo visual, lido dos CSVs gravados (roda sem refazer a análise)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = {"ERA5 (rajada bruta)": "#c44", "modelo, arm base": "#2a7ab0", "modelo, arm full": "#5aa469"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5), constrained_layout=True)

    cal = pd.read_csv(saida / "calibracao_condicional.csv")
    a = ax[0, 0]
    for nome, g in cal.groupby("previsor"):
        a.plot(g.prev_media, g.obs_media, "o-", color=cores[nome], ms=4, label=nome)
    lim = [cal.prev_media.min() - 0.5, cal.prev_media.max() + 0.5]
    a.plot(lim, lim, "k--", lw=1, label="calibração perfeita")
    a.set_xlabel("previsão média na faixa (m/s)")
    a.set_ylabel("observado médio na faixa (m/s)")
    a.set_title("(a) Calibração condicional NA PREVISÃO\nfaixas dos quantis 0–10 %, …, 95–99 %, 99–100 %")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    exc = pd.read_csv(saida / "excedencia.csv")
    a = ax[0, 1]
    larg, xs = 0.25, np.arange(3)
    for i, (nome, c) in enumerate(cores.items()):
        g = exc[exc.previsor == nome]
        a.bar(xs + (i - 1) * larg, g.pod, larg, color=c, label=nome, edgecolor="black", linewidth=0.5)
        for x, v in zip(xs + (i - 1) * larg, g.pod):
            a.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    a.set_xticks(xs)
    a.set_xticklabels([f"{r.limiar}\n({r['valor_m/s']:.1f} m/s)" for _, r in exc[exc.previsor == "ERA5 (rajada bruta)"].iterrows()])
    a.set_ylabel("POD — fração dos extremos observados que foram previstos")
    a.set_title("(b) Detecção de excedência")
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")

    res = pd.read_csv(saida / "esquemas_resumo.csv")
    res = res[res.scheme != "mes (ano-mês)"]
    a = ax[1, 0]
    y = np.arange(len(res))
    a.barh(y, res.largura_ic_relativa_mediana, color="#7a7a7a", edgecolor="black")
    a.errorbar(res.largura_ic_relativa_mediana, y,
               xerr=[res.largura_ic_relativa_mediana - res.largura_min, res.largura_max - res.largura_ic_relativa_mediana],
               fmt="none", ecolor="k", capsize=3)
    a.axvline(1, color="#c44", ls="--", lw=1.2, label="blocos ano-mês (estudo)")
    a.set_yticks(y)
    a.set_yticklabels(res.scheme)
    a.set_xlabel("largura do IC ÷ largura com blocos ano-mês  (mediana; barras = mín–máx)")
    a.set_title("(c) O IC depende do tipo de bloco?")
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="x")

    est = pd.read_csv(saida / "estratos.csv")
    a = ax[1, 1]
    g = est[est.dimensao == "regime"]
    regs = sorted(g.estrato.unique())
    for i, (nome, c) in enumerate(cores.items()):
        v = [g[(g.estrato == r) & (g.previsor == nome)].rmse_p90.iloc[0] for r in regs]
        a.bar(np.arange(len(regs)) + (i - 1) * larg, v, larg, color=c, label=nome, edgecolor="black", linewidth=0.5)
    a.set_xticks(range(len(regs)))
    a.set_xticklabels(regs)
    a.set_ylabel("RMSE na cauda (observado ≥ P90), m/s")
    a.set_title("(d) Cauda por regime sinótico (R1–R4: ver regimes_centroides.csv)")
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle("Robustez do estudo de grupos — mesma cobertura de linhas para ERA5 e modelos", fontsize=13)
    destino = saida / "robustez.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="cluster3_groups_modal")
    ap.add_argument("--so-figura", action="store_true", help="só refaz a figura a partir dos CSVs")
    args = ap.parse_args()
    if args.so_figura:
        print(figura(RAIZ / "artifacts/feature_study" / args.study / "robustez"))
        return
    estudo = RAIZ / "artifacts/feature_study" / args.study
    saida = estudo / "robustez"
    saida.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    test, resid, models, comps, dias, arms, tags = _preparar(estudo)
    comps = injeta_era5(resid, test, tags, models, comps)
    print(f"{args.study}: {len(models)} modelos, {len(tags)} seeds, {len(comps)} comparações, "
          f"{dias.n_days} dias, {dias.run_id.max() + 1} sequências contíguas")

    print("\n[1/6] desempenho em cauda")
    desemp = tabela_desempenho(resid, test, dias, tags, models, arms)
    desemp.to_csv(saida / "desempenho_cauda.csv", index=False)
    sel = desemp[desemp.arm.isin(["ERA5 (rajada bruta)", "base", "full"])].set_index("arm")
    print(sel[["rmse", "rmse_p90", "rmse_p95", "rmse_p99", "bias_p90", "bias_p95", "bias_p99",
               "pinball_90", "pinball_95", "pinball_99"]].round(3).to_string())

    print("\n[2/6] esquemas de reamostragem (eventos, blocos móveis, estacionário)")
    rotulo_ev, ev_info = rb.event_labels(test, dias)
    print(f"   {len(ev_info)} eventos (≥2 estações no P90 próprio), duração mediana {ev_info.dias.median():.0f} d, "
          f"máx {ev_info.dias.max()} d; {int(rotulo_ev.max() + 1)} blocos no total (eventos + trechos calmos)")
    ef = esquemas(resid, test, dias, comps, tags, models, rotulo_ev)
    ef.to_csv(saida / "efeitos_esquemas.csv", index=False)
    res = resumo_esquemas(ef)
    res.to_csv(saida / "esquemas_resumo.csv", index=False)
    print(res.round(3).to_string(index=False))

    print("\n[3/6] Diebold–Mariano")
    dm = dm_tabela(resid, test, dias, comps, tags, models)
    dm.to_csv(saida / "diebold_mariano.csv", index=False)
    print(dm.round(4).to_string(index=False))

    print("\n[4/6] excedência")
    exc = excedencia(resid, test, tags, models)
    exc.to_csv(saida / "excedencia.csv", index=False)
    print(exc.round(3).to_string(index=False))

    print("\n[4b] calibração condicional na previsão (faixas dos quantis da própria previsão)")
    cal = calibracao(resid, test, tags, models)
    cal.to_csv(saida / "calibracao_condicional.csv", index=False)
    topo = cal[cal.faixa.isin(["0.90-0.95", "0.95-0.99", "0.99-1.00"])]
    print(topo.round(2).to_string(index=False))

    print("\n[5/6] regimes e estratos")
    rot_reg, cent = regimes(test, dias)
    cent.to_csv(saida / "regimes_centroides.csv")
    print(cent.round(2).to_string())
    em_evento = np.isin(rotulo_ev, ev_info["evento"].to_numpy())
    rotulos = {
        "estação": test["estacao"].astype(str).to_numpy(),
        "trimestre": test["season"].astype(str).to_numpy(),
        "regime": rot_reg[dias.row_day],
        "evento": np.where(em_evento[dias.row_day], "dentro de evento", "fora de evento"),
    }
    est = estratos(resid, test, dias, tags, models, rotulos)
    est.to_csv(saida / "estratos.csv", index=False)

    print("\n[6/6] eventos extremos")
    top, todos = eventos_extremos(resid, test, dias, tags, models, rotulo_ev, ev_info)
    top.to_csv(saida / "eventos_extremos.csv", index=False)
    todos.to_csv(saida / "eventos_todos.csv", index=False)
    print(top.round(2).to_string(index=False))
    print(f"\nfigura: {figura(saida)}")
    print(f"gravado em {saida}")


if __name__ == "__main__":
    main()
