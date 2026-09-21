"""Amostragem estratificada representativa do treino, com diagnósticos.

Princípios (cada um vem de uma falha concreta que já custou caro em outro lugar):

1. **Alocação proporcional** por estrato (estação × mês × faixa do alvo).
   Estratificar pelo alvo só é válido assim: a probabilidade de inclusão fica
   uniforme e apenas a variância cai. Sobre-amostrar a cauda "para cobrir o
   P90" deslocaria a distribuição de treino — não é feito.

2. **A amostra depende só de (seed, chave da linha, alvo)** — nunca das
   features. Seleção dentro do estrato por rank de um hash da chave. É isso que
   garante linhas idênticas entre todos os arms, requisito de qualquer
   comparação pareada, e é testável.

3. **O tamanho vem de uma tolerância** (limite DKW), mas é heurística: DKW
   assume iid, e as estações do mesmo cluster veem os mesmos dias, então o n
   efetivo é menor. Quem valida o tamanho é a curva de aprendizado do piloto.

4. **Gates estruturais bloqueiam; diagnósticos estatísticos são relatados.**
   As checagens do alvo passam por construção (é o que a estratificação
   garante) e valem como sanidade. As de feature medem só o acaso, porque as
   features não estão nos estratos — re-sortear até passar seria seleção. Por
   isso só uma falha GROSSEIRA (> 2× o limiar) bloqueia, e ela indica bug.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.feature_study.config import ALPHA, EPS, SEASONS
from src.pipelines.common import TARGET_VAR

KEY = ["estacao", "time"]
QUANTILES = (0.5, 0.9, 0.95, 0.99)


# ── Tamanho ─────────────────────────────────────────────────────────────────

def dkw_sample_size(eps: float = EPS, alpha: float = ALPHA) -> int:
    """n mínimo para que, com confiança 1−alpha, sup|F_amostra − F_pop| ≤ eps."""
    return int(math.ceil(math.log(2.0 / alpha) / (2.0 * eps ** 2)))


def dkw_epsilon(n: int, alpha: float = ALPHA, m: int = 1) -> float:
    """Limite DKW para uma amostra de n, com Bonferroni sobre m testes."""
    return math.sqrt(math.log(2.0 * m / alpha) / (2.0 * n))


# ── Alocação e sorteio ──────────────────────────────────────────────────────

def allocate_proportional(counts: pd.Series, n: int) -> pd.Series:
    """Alocação proporcional por maior resto: soma exatamente `n`, cada estrato
    a menos de 1 linha do valor esperado e nunca acima do seu tamanho."""
    counts = counts.astype(int)
    total = int(counts.sum())
    if n >= total:
        return counts
    exact = counts.to_numpy() * (n / total)
    base = np.floor(exact).astype(int)
    rem = n - int(base.sum())
    order = np.argsort(-(exact - base), kind="stable")
    extra = np.zeros(len(counts), dtype=int)
    extra[order[:rem]] = 1
    return pd.Series(base + extra, index=counts.index)


def row_hash(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Hash determinístico (siphash, não depende de PYTHONHASHSEED) da chave da
    linha misturada com a seed. Só usa estação e tempo."""
    t = pd.DatetimeIndex(df["time"]).asi8.astype(str)
    keys = (f"{seed}|" + df["estacao"].astype(str).to_numpy().astype(object)
            + "|" + t.astype(object))
    return pd.util.hash_array(np.asarray(keys, dtype=object))


def target_edges(y: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(y, dtype=float), [0.5, 0.9])


def draw_stratified(df: pd.DataFrame, n: int, seed: int, edges: np.ndarray) -> np.ndarray:
    """Posições (em `df`) das linhas sorteadas. `df` já precisa estar em ordem
    canônica; o resultado não depende de nenhuma coluna de feature."""
    N = len(df)
    if n >= N:
        return np.arange(N)
    band = np.digitize(df[TARGET_VAR].to_numpy(float), edges)
    strata = (df["estacao"].astype(str).to_numpy() + "|"
              + pd.DatetimeIndex(df["time"]).month.astype(str).to_numpy() + "|"
              + band.astype(str))
    codes, _ = pd.factorize(strata)
    counts = pd.Series(np.bincount(codes))
    alloc = allocate_proportional(counts, n).to_numpy()
    rank = pd.DataFrame({"s": codes, "h": row_hash(df, seed)}).groupby("s")["h"].rank(method="first").to_numpy()
    return np.flatnonzero(rank <= alloc[codes])


def group_sizes(pop: pd.DataFrame, n_target: int) -> dict:
    """Tamanho esperado da amostra por (split, trimestre)."""
    return {
        (sp, se): min(len(g), n_target)
        for (sp, se), g in pop.groupby(["_split", "season"])
    }


def build_replicate(trainval: pd.DataFrame, seed: int, n_target: int) -> pd.DataFrame:
    """Uma réplica: por (split, trimestre), amostra estratificada.

    Validação (~1 mil linhas/trimestre) < n ⇒ entra integral; a regra decide,
    não é caso especial. Faixas do alvo vêm sempre da população de TREINO do
    trimestre (a validação não define nada).
    """
    trainval = trainval.sort_values(KEY).reset_index(drop=True)
    parts = []
    for season in SEASONS:
        tr = trainval[(trainval["season"] == season) & (trainval["_split"] == "train")]
        if tr.empty:
            continue
        edges = target_edges(tr[TARGET_VAR].to_numpy())
        for split in ("train", "val"):
            g = trainval[(trainval["season"] == season) & (trainval["_split"] == split)]
            if g.empty:
                continue
            g = g.reset_index(drop=True)
            parts.append(g.iloc[draw_stratified(g, n_target, seed, edges)])
    return pd.concat(parts, ignore_index=True)


# ── Diagnósticos ────────────────────────────────────────────────────────────

def _flag(stat: float, thr: float) -> str:
    if not np.isfinite(stat):
        return "fail"
    return "ok" if stat <= thr else ("warn" if stat <= 2.0 * thr else "fail")


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    sd = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return abs(float(np.mean(a) - np.mean(b))) / sd if sd > 0 else 0.0


def _n_strata(pop: pd.DataFrame) -> int:
    edges = target_edges(pop[TARGET_VAR].to_numpy())
    band = np.digitize(pop[TARGET_VAR].to_numpy(float), edges)
    month = pd.DatetimeIndex(pop["time"]).month
    return int(pd.Series(pop["estacao"].astype(str).to_numpy() + "|" + month.astype(str) + "|" + band.astype(str)).nunique())


def representativeness_report(
    pop: pd.DataFrame, sample: pd.DataFrame, features, alpha: float = ALPHA,
    scope: str = "",
) -> pd.DataFrame:
    """Amostra vs. população (mesmo split e trimestre).

    Todo limiar é ESTATÍSTICO (escala com n) ou é a folga de arredondamento da
    alocação (`n_strata / n`, cada estrato erra menos de 1 linha). Limiar fixo
    disparava por acaso em amostra pequena — falha que só apareceria como
    teste instável. `kind`: `sanity` passa por construção, `reported` só
    relata; ambos bloqueiam apenas se > 2× o limiar (falha grosseira).
    """
    n = len(sample)
    if n >= len(pop):
        return pd.DataFrame([{
            "scope": scope, "check": "integral", "statistic": 0.0, "threshold": 0.0,
            "flag": "ok", "kind": "sanity",
        }])
    rows = []

    def add(check, stat, thr, kind):
        rows.append({"scope": scope, "check": check, "statistic": float(stat),
                     "threshold": float(thr), "flag": _flag(stat, thr), "kind": kind})

    slack = _n_strata(pop) / n
    eps_n = dkw_epsilon(n, alpha)
    y_p, y_s = pop[TARGET_VAR].to_numpy(float), sample[TARGET_VAR].to_numpy(float)
    add("target_ks", stats.ks_2samp(y_p, y_s).statistic, eps_n, "sanity")
    for q in QUANTILES:
        # No espaço das probabilidades: o quantil da amostra tem de cair perto de q
        # na CDF da população. Erro relativo no valor não é comparável entre
        # quantis (a cauda é esparsa por natureza).
        add(f"target_q{int(q * 100)}_prob_err", abs(float(np.mean(y_p <= np.quantile(y_s, q))) - q), eps_n, "sanity")
    p90 = np.quantile(y_p, 0.9)
    add("tail_share_diff", abs((y_s >= p90).mean() - (y_p >= p90).mean()), 0.01 + slack, "sanity")

    for col, name in (("estacao", "station"), ("month", "month"), ("year", "year")):
        gp = pop[col] if col in pop else getattr(pd.DatetimeIndex(pop["time"]), col)
        gs = sample[col] if col in sample else getattr(pd.DatetimeIndex(sample["time"]), col)
        sp_ = pd.Series(np.asarray(gp)).value_counts(normalize=True)
        ss_ = pd.Series(np.asarray(gs)).value_counts(normalize=True).reindex(sp_.index, fill_value=0.0)
        dev = float((ss_ - sp_).abs().max())
        if name == "year":  # não estratificado: variação binomial a 3 sigma
            pmax = float(sp_.max())
            add("year_share_max_dev", dev, 3.0 * math.sqrt(pmax * (1 - pmax) / n) + slack, "reported")
        else:
            add(f"{name}_share_max_dev", dev, 0.01 + slack, "sanity")

    feats = [f for f in features if f in pop.columns]
    eps_f = dkw_epsilon(n, alpha, m=max(len(feats), 1))
    smd_thr = max(0.1, 3.0 / math.sqrt(n))
    for f in feats:
        a, b = pop[f].to_numpy(float), sample[f].to_numpy(float)
        add(f"feature_ks::{f}", stats.ks_2samp(a, b).statistic, eps_f, "reported")
        add(f"feature_smd::{f}", _smd(a, b), smd_thr, "reported")
    return pd.DataFrame(rows)


def assert_no_gross_failure(report: pd.DataFrame) -> None:
    bad = report[report["flag"] == "fail"]
    if len(bad):
        raise ValueError(
            "amostra reprovada por falha GROSSEIRA (> 2× o limiar) — indica bug, não acaso:\n"
            + bad.head(10).to_string(index=False)
        )


def structural_violations(
    sample: pd.DataFrame, population: pd.DataFrame, test: pd.DataFrame,
    features, expected_sizes: dict | None = None,
) -> list[str]:
    """Gates estruturais (bloqueiam). Lista vazia = ok."""
    out = []
    if sample.duplicated(KEY).any():
        out.append("chaves (estacao,time) duplicadas na amostra")
    pk = set(zip(population["estacao"], population["time"]))
    sk = set(zip(sample["estacao"], sample["time"]))
    if not sk <= pk:
        out.append("amostra contém linhas fora da população")
    tk = set(zip(test["estacao"], test["time"]))
    if sk & tk:
        out.append(f"vazamento: {len(sk & tk)} chaves da amostra estão no teste")
    if set(sample["_split"]) & {"test", "out"}:
        out.append("amostra contém linhas de teste/out")
    missing = [f for f in features if f not in sample.columns]
    if missing:
        out.append(f"features ausentes na amostra: {missing[:5]}")
    else:
        nan = int(sample[list(features)].isna().any(axis=1).sum())
        if nan:
            out.append(f"{nan} linhas da amostra com NaN em alguma feature")
    if expected_sizes:
        got = sample.groupby(["_split", "season"]).size().to_dict()
        for k, want in expected_sizes.items():
            if got.get(k, 0) != want:
                out.append(f"tamanho de {k}: esperado {want}, obtido {got.get(k, 0)}")
    return out
