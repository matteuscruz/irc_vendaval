"""Análises de robustez do estudo de features — todas sobre resíduos JÁ gravados.

Nada aqui ajusta modelo: o ganho é medir a MESMA pergunta com outras réguas, de
graça. Quatro blocos:

  1. **Métricas de cauda** além do P90: RMSE e viés em P95/P99 e a perda
     quantílica (pinball) da previsão pontual.
  2. **Reamostragem alternativa**: o bootstrap do estudo usa blocos (ano, mês).
     Aqui o mesmo efeito é reestimado com blocos por EVENTO meteorológico,
     blocos móveis e bootstrap estacionário, para ver se a conclusão depende da
     escolha do bloco.
  3. **Diebold–Mariano** sobre a diferença diária de perda, com variância de
     longo prazo (HAC) que só soma autocovariâncias dentro de sequências
     contíguas de dias.
  4. **Estratificação** por estação, trimestre, regime sinótico e evento, e
     detecção de excedência (POD/FAR/CSI) no lugar de só erro médio.

A unidade de reamostragem é o DIA (soma sobre as estações): a dependência entre
estações no mesmo dia é forte e quebrá-la num bootstrap por linha inflaria o
poder do teste.

Convenção do resíduo, a mesma de `worker.py`: `e = previsão_truncada − observado`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.feature_study.analysis import _verdict, comparisons, discover_models, load_residuals
from src.feature_study.config import BOOTSTRAP_SEED, SESOI_REL
from src.pipelines.common import TARGET_VAR

TAIL_QS = (0.90, 0.95, 0.99)
PINBALL_TAUS = (0.90, 0.95, 0.99)

METRIC_NAMES = (
    "rmse", "rmse_p90", "rmse_p95", "rmse_p99",
    "bias_p90", "bias_p95", "bias_p99",
    "pinball_90", "pinball_95", "pinball_99",
)
# Métrica de viés: o efeito é o DESLOCAMENTO com sinal (melhor − pior), como no
# estudo; nas demais, ganho = pior − melhor (positivo = o melhor arm é melhor).
_BIAS = {"bias_p90", "bias_p95", "bias_p99"}


# ── Dias como unidade de reamostragem ───────────────────────────────────────

@dataclass
class Dias:
    """Mapa linha → dia e a estrutura de contiguidade do calendário."""
    row_day: np.ndarray        # (n_rows,) índice do dia de cada linha
    dates: pd.DatetimeIndex    # (n_days,) datas únicas, ordenadas
    run_id: np.ndarray         # (n_days,) sequência contígua de dias (sem buraco)

    @property
    def n_days(self) -> int:
        return len(self.dates)


def make_days(test: pd.DataFrame) -> Dias:
    t = pd.DatetimeIndex(test["time"]).normalize()
    dates = pd.DatetimeIndex(np.unique(t.values))
    row_day = np.searchsorted(dates.values, t.values)
    quebra = np.r_[True, np.diff(dates.values).astype("timedelta64[D]").astype(int) > 1]
    return Dias(row_day=row_day, dates=dates, run_id=np.cumsum(quebra) - 1)


def month_labels(test: pd.DataFrame, dias: Dias) -> np.ndarray:
    """Bloco (ano, mês) de cada DIA, numerado na ordem de primeira aparição nas
    LINHAS — exatamente a numeração de `analysis.make_blocks`. Com a mesma
    numeração e o mesmo gerador, o sorteio é idêntico ao do estudo e o efeito
    reproduz o dele (verificado no script)."""
    key = test["year"].astype(int).to_numpy() * 100 + test["month"].astype(int).to_numpy()
    ids, _ = pd.factorize(key)
    por_dia = np.full(dias.n_days, -1, dtype=int)
    por_dia[dias.row_day] = ids
    return por_dia


def event_labels(
    test: pd.DataFrame, dias: Dias, *, k: int = 2, station_q: float = 0.90,
    gap_days: int = 1, quiet_chunk: int = 7,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Particiona o calendário de teste em EVENTOS e trechos calmos.

    Um dia é ATIVO se ao menos `k` estações observam rajada no seu próprio
    `station_q` (limiar por estação: a climatologia de rajada difere entre
    elas). Dias ativos separados por até `gap_days` dias viram um mesmo evento.
    O restante do calendário é cortado em trechos de até `quiet_chunk` dias, para
    que nenhum bloco calmo seja do tamanho de um mês.

    Devolve `(rótulo do bloco por dia, tabela dos eventos)`.
    """
    y = test[TARGET_VAR].to_numpy(float)
    est = test["estacao"].astype(str).to_numpy()
    lim = pd.Series(y).groupby(est).transform(lambda s: s.quantile(station_q)).to_numpy()
    exc = pd.Series((y >= lim).astype(int)).groupby(dias.row_day).sum().reindex(
        range(dias.n_days), fill_value=0).to_numpy()
    ativo = exc >= k

    rotulo = np.full(dias.n_days, -1, dtype=int)
    prox, ev_info = 0, []
    i = 0
    while i < dias.n_days:
        if not ativo[i]:
            i += 1
            continue
        j = i
        while True:  # estende o evento por dias ativos a até `gap_days` de distância, na mesma sequência
            nxt = next((m for m in range(j + 1, min(j + gap_days + 2, dias.n_days))
                        if ativo[m] and dias.run_id[m] == dias.run_id[i]), None)
            if nxt is None:
                break
            j = nxt
        rotulo[i:j + 1] = prox
        ev_info.append({"evento": prox, "inicio": dias.dates[i], "fim": dias.dates[j],
                        "dias": j - i + 1, "pico_estacoes_acima": int(exc[i:j + 1].max())})
        prox += 1
        i = j + 1

    # trechos calmos: sequências de dias sem rótulo, em fatias de `quiet_chunk`
    i = 0
    while i < dias.n_days:
        if rotulo[i] != -1:
            i += 1
            continue
        fim = i
        while (fim + 1 < dias.n_days and rotulo[fim + 1] == -1
               and dias.run_id[fim + 1] == dias.run_id[i] and fim + 1 - i < quiet_chunk):
            fim += 1
        rotulo[i:fim + 1] = prox
        prox += 1
        i = fim + 1
    return rotulo, pd.DataFrame(ev_info)


# ── Somas por dia e métricas ────────────────────────────────────────────────

def _quantil(y: np.ndarray, mask: np.ndarray, q: float) -> float:
    return float(np.quantile(y[mask], q))


def day_sums(e: np.ndarray, y: np.ndarray, mask: np.ndarray, dias: Dias) -> np.ndarray:
    """Matriz (n_dias, K) de somas, de onde saem todas as métricas por sorteio.

    Colunas: [n, Σe²] + por quantil de cauda [n_cauda, Σe, Σe²] + por τ [Σ pinball].
    Os limiares de cauda saem de `y` nas linhas em que o modelo existe, como no
    estudo (cada modelo só cobre os trimestres em que foi eleito)."""
    m = mask.astype(float)
    e0 = np.where(mask, e, 0.0)
    colunas = [m, e0 * e0]
    for q in TAIL_QS:
        cauda = ((y >= _quantil(y, mask, q)) & mask).astype(float)
        colunas += [cauda, e0 * cauda, e0 * e0 * cauda]
    u = -e0                                   # observado − previsão (zero fora da máscara)
    for tau in PINBALL_TAUS:
        colunas.append(np.maximum(tau * u, (tau - 1.0) * u))
    R = np.column_stack(colunas)
    out = np.zeros((dias.n_days, R.shape[1]))
    for c in range(R.shape[1]):
        out[:, c] = np.bincount(dias.row_day, weights=R[:, c], minlength=dias.n_days)
    return out


def metrics_from_totals(T: np.ndarray) -> dict[str, np.ndarray]:
    """`T`: (n_sorteios, K) = counts @ day_sums."""
    n = T[:, 0]
    out = {"rmse": np.sqrt(T[:, 1] / n)}
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, q in enumerate(TAIL_QS):
            nt, s1, s2 = T[:, 2 + 3 * i], T[:, 3 + 3 * i], T[:, 4 + 3 * i]
            tag = f"p{int(round(q * 100))}"
            out[f"rmse_{tag}"] = np.sqrt(s2 / nt)
            out[f"bias_{tag}"] = s1 / nt
        base = 2 + 3 * len(TAIL_QS)
        for j, tau in enumerate(PINBALL_TAUS):
            out[f"pinball_{int(round(tau * 100))}"] = T[:, base + j] / n
    return out


# ── Esquemas de reamostragem: todos devolvem contagens sobre DIAS ───────────

def counts_by_blocks(labels: np.ndarray, n_boot: int, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Multinomial sobre blocos, expandida para dias. Linha 0 = amostra original.

    Mesmo gerador e mesma chamada de `analysis.compute_effects`: com os blocos
    (ano, mês) o sorteio é idêntico ao do estudo."""
    nb = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    por_bloco = np.vstack([
        np.ones(nb),
        rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(float),
    ])
    return por_bloco[:, labels]


def counts_moving_block(dias: Dias, length: int, n_boot: int, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap de blocos móveis de `length` dias, SEM atravessar buracos do
    calendário (meses de teste não contíguos). Cada sorteio reconstrói ~N dias."""
    n = dias.n_days
    ok = np.array([s + length <= n and dias.run_id[s] == dias.run_id[s + length - 1]
                   for s in range(n)])
    starts = np.flatnonzero(ok)
    if len(starts) == 0:
        raise ValueError(f"nenhuma janela contígua de {length} dias")
    rng = np.random.default_rng(seed)
    nblk = math.ceil(n / length)
    off = np.arange(length)
    out = np.ones((n_boot + 1, n))
    for b in range(1, n_boot + 1):
        idx = (rng.choice(starts, size=nblk)[:, None] + off).ravel()
        out[b] = np.bincount(idx, minlength=n)
    return out


def counts_stationary(dias: Dias, mean_length: float, n_boot: int, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap estacionário (Politis–Romano): comprimentos geométricos de média
    `mean_length`. Um bloco termina ao fim da sequência contígua (não atravessa
    buracos), então o tamanho do sorteio é aproximadamente N, não exatamente N."""
    n = dias.n_days
    fim_run = np.zeros(n, dtype=int)
    for r in np.unique(dias.run_id):
        idx = np.flatnonzero(dias.run_id == r)
        fim_run[idx] = idx[-1]
    rng = np.random.default_rng(seed)
    out = np.ones((n_boot + 1, n))
    p = 1.0 / mean_length
    for b in range(1, n_boot + 1):
        cnt = np.zeros(n)
        total = 0
        while total < n:
            s = int(rng.integers(0, n))
            tam = int(min(rng.geometric(p), fim_run[s] - s + 1))
            cnt[s:s + tam] += 1
            total += tam
        out[b] = cnt
    return out


# ── Efeitos pareados com qualquer esquema ───────────────────────────────────

def paired_effects(
    resid: dict, test: pd.DataFrame, dias: Dias, comps, tags, models, counts: np.ndarray,
    *, scheme: str, metrics=METRIC_NAMES, n_units: int | None = None,
) -> pd.DataFrame:
    """Efeito pareado por comparação × métrica, para uma matriz de contagens.

    A definição é a do estudo (`analysis.compute_effects`): diferença por
    (seed, modelo) nas linhas em que ambos os arms existem; média sobre modelos,
    depois sobre seeds; `se² = se_boot² + se_rep²`."""
    y = test[TARGET_VAR].to_numpy(float)
    cache: dict = {}

    def draws(tag, arm, model):
        k = (tag, arm, model)
        if k not in cache:
            e = resid[k]
            mask = np.isfinite(e)
            if not mask.any():
                cache[k] = None
            else:
                cache[k] = (metrics_from_totals(counts @ day_sums(e, y, mask, dias)), mask)
        return cache[k]

    # escala do SESOI: 1% da métrica no arm de referência ("base")
    ancora = {}
    for m in metrics:
        v = [d[0][m][0] for t in tags for mod in models
             if (d := draws(t, "base", mod)) is not None]
        ancora[m] = float(np.mean(v)) if v else float("nan")

    rows = []
    for name, worse, better in comps:
        for metric in metrics:
            por_tag: dict[str, list] = {}
            n_mod: set = set()
            for t in tags:
                for mod in models:
                    w, b = draws(t, worse, mod), draws(t, better, mod)
                    if w is None or b is None or not np.array_equal(w[1], b[1]):
                        continue
                    dw, db = w[0][metric], b[0][metric]
                    por_tag.setdefault(t, []).append(db - dw if metric in _BIAS else dw - db)
                    n_mod.add(mod)
            if not por_tag:
                continue
            tag_means = np.array([np.mean(v, axis=0) for v in por_tag.values()])
            agg = tag_means.mean(axis=0)
            r = len(tag_means)
            se_boot = float(np.std(agg[1:], ddof=1))
            se_rep = float(np.std(tag_means[:, 0], ddof=1) / math.sqrt(r)) if r > 1 else 0.0
            se = math.sqrt(se_boot ** 2 + se_rep ** 2)
            point = float(agg[0])
            sesoi = SESOI_REL * abs(ancora[metric]) if np.isfinite(ancora[metric]) else float("nan")
            row = {"scheme": scheme, "comparison": name, "metric": metric, "effect": point,
                   "se_boot": se_boot, "se_rep": se_rep, "se": se,
                   "ci_lo": point - 1.96 * se, "ci_hi": point + 1.96 * se, "sesoi": sesoi,
                   "n_models": len(n_mod), "n_units": n_units}
            row["verdict"] = _verdict(row)
            rows.append(row)
    return pd.DataFrame(rows)


def load_study(out_dir, data_dir, tags, arms):
    """`(test, resid, models, comps, dias)` do estudo gravado em `out_dir`."""
    from src.feature_study.analysis import load_test

    test = load_test(data_dir)
    models = discover_models(out_dir, tags)
    names = [a.name for a in arms]
    resid = load_residuals(out_dir, tags, names, test, models)
    have = {a for a in names if any(np.isfinite(resid[(t, a, m)]).any() for t in tags for m in models)}
    return test, resid, models, comparisons(arms, have), make_days(test)


# ── Diebold–Mariano com HAC dentro de sequências contíguas ──────────────────

def daily_loss_differential(
    resid: dict, test: pd.DataFrame, dias: Dias, worse: str, better: str, tags, models,
) -> tuple[np.ndarray, int]:
    """Diferença diária média de erro² (pior − melhor), média sobre as estações do
    dia, sobre os modelos e sobre as seeds. NaN nos dias sem cobertura.

    Agregar por dia (e não por linha) evita tratar 9 estações do mesmo dia como 9
    observações independentes."""
    n = dias.n_days
    soma = np.zeros(n)
    cont = np.zeros(n)
    usados = 0
    for t in tags:
        for m in models:
            ew, eb = resid[(t, worse, m)], resid[(t, better, m)]
            ok = np.isfinite(ew) & np.isfinite(eb)
            if not ok.any():
                continue
            d = np.where(ok, ew * ew - eb * eb, 0.0)
            por_dia = np.bincount(dias.row_day, weights=d, minlength=n)
            n_dia = np.bincount(dias.row_day, weights=ok.astype(float), minlength=n)
            com = n_dia > 0
            soma[com] += por_dia[com] / n_dia[com]
            cont[com] += 1
            usados += 1
    out = np.full(n, np.nan)
    out[cont > 0] = soma[cont > 0] / cont[cont > 0]
    return out, usados


def diebold_mariano(d: np.ndarray, run_id: np.ndarray, max_lag: int = 5) -> dict:
    """Estatística DM sobre `d` (diferencial diário de perda; positivo = o 2º
    modelo é melhor).

    A variância de longo prazo soma autocovariâncias (pesos de Bartlett) apenas
    entre dias da MESMA sequência contígua: os meses de teste não são contíguos,
    e correlacionar dezembro com janeiro do ano seguinte, sem nada no meio,
    seria correlacionar o que não é adjacente. Correção de pequena amostra de
    Harvey–Leybourne–Newbold para horizonte 1; $p$ pela t de Student."""
    ok = np.isfinite(d)
    x, r = d[ok], run_id[ok]
    n = len(x)
    if n < 10:
        return {"dm": float("nan"), "p": float("nan"), "n_dias": n, "media": float("nan")}
    xb = x.mean()
    z = x - xb
    g0 = float(np.dot(z, z)) / n
    lrv = g0
    for lag in range(1, max_lag + 1):
        mesmo = r[lag:] == r[:-lag]
        g = float(np.dot(z[lag:][mesmo], z[:-lag][mesmo])) / n
        lrv += 2.0 * (1.0 - lag / (max_lag + 1)) * g
    lrv = max(lrv, 1e-12)
    dm = xb / math.sqrt(lrv / n)
    dm *= math.sqrt((n - 1) / n)
    p = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    return {"dm": float(dm), "p": p, "n_dias": n, "media": float(xb)}


# ── Detecção de excedência ──────────────────────────────────────────────────

def contingency(y: np.ndarray, pred: np.ndarray, limiar: float) -> dict:
    """POD, FAR, CSI e viés de frequência da previsão `pred >= limiar` contra a
    observação `y >= limiar`. O erro médio esconde se o modelo SABE que o dia é
    extremo; isto mede exatamente isso."""
    obs, prv = y >= limiar, pred >= limiar
    hits = int((obs & prv).sum())
    miss = int((obs & ~prv).sum())
    fa = int((~obs & prv).sum())
    pod = hits / (hits + miss) if hits + miss else float("nan")
    far = fa / (hits + fa) if hits + fa else float("nan")
    csi = hits / (hits + miss + fa) if hits + miss + fa else float("nan")
    fb = (hits + fa) / (hits + miss) if hits + miss else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "freq_bias": fb,
            "n_obs_exc": hits + miss, "n_prev_exc": hits + fa}


# ── Calibração condicional na PREVISÃO ──────────────────────────────────────

RELIAB_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def reliability(y: np.ndarray, pred: np.ndarray, edges=RELIAB_EDGES) -> pd.DataFrame:
    """Observado médio dentro de faixas de PREVISÃO (por quantis da própria previsão).

    Por que existe: "viés em P90" mede o erro condicionando no OBSERVADO alto, e
    isso dá viés negativo mesmo num previsor perfeitamente calibrado — quem
    seleciona os dias pelo resultado sempre pega o que a previsão ficou abaixo
    (regressão à média). A pergunta de calibração é a inversa: quando o previsor
    diz "extremo", o que de fato acontece? Condicionar na previsão não tem esse
    artefato; um previsor calibrado tem observado médio igual ao previsto em toda
    faixa."""
    cortes = np.quantile(pred, edges)
    idx = np.clip(np.searchsorted(cortes, pred, side="right") - 1, 0, len(edges) - 2)
    linhas = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        linhas.append({"faixa": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "n": int(m.sum()),
                       "prev_media": float(pred[m].mean()), "obs_media": float(y[m].mean())})
    return pd.DataFrame(linhas)
