"""Estágio `aggregate`: quais features novas acrescentam informação.

Como o campeão do LazyPredict oscila entre arms (efeito da feature confundido
com troca de algoritmo), o efeito NÃO é medido no campeão. É medido, POR MODELO
e pareado, num conjunto de referência fixado a priori
(`config.REFERENCE_MODELS`), na mesma réplica e no mesmo trimestre.

Convenção de sinal ("ganho"): positivo = o arm MELHOR é melhor.
  RMSE, RMSE_P90 : ganho = erro(pior) − erro(melhor)
  R²             : ganho = R²(melhor) − R²(pior)
  Bias@P90       : deslocamento com sinal = bias(melhor) − bias(pior). Não é
                   "ganho": o viés é negativo e pode cruzar zero, então ler
                   |viés| esconderia o cruzamento.
Para um arm "add" o pior é a referência; para "drop" o pior é o próprio arm.

Incerteza: bootstrap em BLOCOS (ano, mês) do teste — as linhas não são
independentes (estações do mesmo dia, dias consecutivos). O MESMO sorteio de
blocos vale para todos os arms, modelos e réplicas, o que mantém o pareamento.
Por draw calcula-se o efeito por réplica e tira-se a média entre réplicas; soma-se
a variância ENTRE réplicas / R (sem empilhar réplicas como linhas independentes).
As seeds do modelo são fixas, então a variância entre réplicas é só de amostragem.

Decisão: SESOI = 1% do RMSE do base. "Sem informação" significa IC dentro de
±SESOI, e NÃO "IC cruza zero" (ausência de evidência ≠ evidência de ausência).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.feature_study.arms import SELECTION_AXIS, SELECTION_EXTRA, Arm, extra_comparisons
from src.feature_study.config import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CHAMPION_SLACK, LOSS_MODELS, MODEL_FAMILY,
    PRIMARY_METRIC, REFERENCE_MODELS, SESOI_REL,
)
from src.pipelines.common import TARGET_VAR

METRICS = ("rmse", "rmse_p90", "r2", "bias_p90")


# ── Carga ───────────────────────────────────────────────────────────────────

def load_arms(data_dir) -> list[Arm]:
    raw = json.loads((Path(data_dir) / "arms.json").read_text())
    return [Arm.from_dict(d) for d in raw]


def load_test(data_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "test.parquet").sort_values("row_id").reset_index(drop=True)


def discover_models(out_dir, tags) -> list[str]:
    """Modelos que têm resíduo gravado — a união das colunas de `resid__*.parquet`.

    É o que permite ao modo top-5 medir os modelos que a triagem de fato elegeu
    (MLP, GradientBoosting, BayesianRidge…), em vez de só os `REFERENCE_MODELS`.
    Num estudo com os 7 de referência a união é exatamente esses 7."""
    achados: set[str] = set()
    for tag in tags:
        for path in (Path(out_dir) / "units" / tag).glob("resid__*.parquet"):
            achados |= set(pq.ParquetFile(path).schema_arrow.names)
    achados.discard("row_id")
    return sorted(achados)


def load_residuals(out_dir, tags, arm_names, test: pd.DataFrame, models=REFERENCE_MODELS) -> dict:
    """{(tag, arm, model): resíduo alinhado a `test`}. NaN onde faltar — um
    vetor incompleto é descartado na análise, nunca preenchido."""
    pos = pd.Series(np.arange(len(test)), index=test["row_id"].to_numpy())
    out: dict = {}
    for tag in tags:
        for arm in arm_names:
            arrs = {m: np.full(len(test), np.nan) for m in models}
            for path in sorted((Path(out_dir) / "units" / tag).glob(f"resid__*__{arm}.parquet")):
                df = pd.read_parquet(path)
                idx = pos.loc[df["row_id"].to_numpy()].to_numpy()
                for m in models:
                    if m in df.columns:
                        arrs[m][idx] = df[m].to_numpy(float)
            for m, a in arrs.items():
                out[(tag, arm, m)] = a
    return out


# ── Blocos e somas ──────────────────────────────────────────────────────────

@dataclass
class Blocks:
    ids: np.ndarray
    nb: int
    tail: np.ndarray
    n_b: np.ndarray
    nt_b: np.ndarray
    sy_b: np.ndarray
    syy_b: np.ndarray


def make_blocks(test: pd.DataFrame, mask: np.ndarray | None = None) -> Blocks:
    """Somas por bloco (ano, mês). Com `mask`, só as linhas marcadas contam.

    Os ids de bloco são SEMPRE os globais, mesmo com máscara: o sorteio de blocos
    do bootstrap é um só para todos os modelos, e é isso que mantém o pareamento.
    Um bloco sem linhas mascaradas fica com contagem zero.

    A máscara existe porque, com a triagem top-5, cada modelo só tem resíduo nos
    trimestres em que foi eleito. Exigir cobertura total deixava um único modelo
    (a interseção dos quatro trimestres) e escondia o efeito dos demais."""
    key = test["year"].astype(int).to_numpy() * 100 + test["month"].astype(int).to_numpy()
    ids, uniq = pd.factorize(key)
    nb = len(uniq)
    y = test[TARGET_VAR].to_numpy(float)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    tail = (y >= np.quantile(y[m], 0.9)) & m
    return Blocks(
        ids=ids, nb=nb, tail=tail,
        n_b=np.bincount(ids[m], minlength=nb).astype(float),
        nt_b=np.bincount(ids[tail], minlength=nb).astype(float),
        sy_b=np.bincount(ids[m], weights=y[m], minlength=nb),
        syy_b=np.bincount(ids[m], weights=y[m] * y[m], minlength=nb),
    )


def _masked_draws(counts, e, test, cache: dict):
    """Métricas (por sorteio) de um vetor de resíduo nas linhas em que ele existe.
    Devolve `(draws, mask)` ou `None` se o vetor não tem nenhuma linha."""
    mask = np.isfinite(e)
    if not mask.any():
        return None
    k = mask.tobytes()
    if k not in cache:
        cache[k] = make_blocks(test, mask)
    return _draw_metrics(counts, np.where(mask, e, 0.0), cache[k]), mask


def _block_sums(e: np.ndarray, b: Blocks) -> np.ndarray:
    return np.column_stack([
        np.bincount(b.ids, weights=e * e, minlength=b.nb),
        np.bincount(b.ids[b.tail], weights=e[b.tail], minlength=b.nb),
        np.bincount(b.ids[b.tail], weights=e[b.tail] ** 2, minlength=b.nb),
    ])


def _draw_metrics(counts: np.ndarray, e: np.ndarray, b: Blocks) -> dict[str, np.ndarray]:
    """Métricas para cada sorteio (linha 0 de `counts` = amostra original)."""
    s2, s1t, s2t = (counts @ _block_sums(e, b)).T
    n, nt = counts @ b.n_b, counts @ b.nt_b
    sy, syy = counts @ b.sy_b, counts @ b.syy_b
    sst = syy - sy ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "rmse": np.sqrt(s2 / n),
            "rmse_p90": np.sqrt(s2t / nt),
            "bias_p90": s1t / nt,
            "r2": 1.0 - s2 / sst,
        }


def _delta(metric: str, worse: np.ndarray, better: np.ndarray) -> np.ndarray:
    if metric in ("rmse", "rmse_p90"):
        return worse - better
    return better - worse  # r2 (ganho) e bias_p90 (deslocamento com sinal)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / (np.arange(n) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


# ── Efeitos ─────────────────────────────────────────────────────────────────

def comparisons(arms: list[Arm], available: set[str]) -> list[tuple[str, str, str]]:
    """(nome, pior, melhor) — só as cujos dois arms têm resultado."""
    out = []
    for a in arms:
        if not a.reference:
            continue
        worse, better = (a.reference, a.name) if a.role == "add" else (a.name, a.reference)
        out.append((a.name, worse, better))
    out += extra_comparisons(arms)
    return [c for c in out if c[1] in available and c[2] in available]


def compute_effects(
    out_dir, data_dir, tags, arms: list[Arm] | None = None,
    n_boot: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
    models=None,
) -> pd.DataFrame:
    """Tabela de efeitos pareados, por comparação × família × métrica.

    `models=None` mede todos os modelos com resíduo gravado (`discover_models`)."""
    models = discover_models(out_dir, tags) if models is None else list(models)
    arms = arms or load_arms(data_dir)
    test = load_test(data_dir)
    blocks = make_blocks(test)
    arm_names = [a.name for a in arms]
    resid = load_residuals(out_dir, tags, arm_names, test, models)

    have = {a for a in arm_names
            if any(np.isfinite(resid[(t, a, m)]).any() for t in tags for m in models)}
    comps = comparisons(arms, have)
    if not comps:
        raise ValueError("nenhuma comparação disponível — faltam resíduos dos arms")

    rng = np.random.default_rng(seed)
    counts = np.vstack([
        np.ones(blocks.nb),
        rng.multinomial(blocks.nb, np.full(blocks.nb, 1.0 / blocks.nb), size=n_boot).astype(float),
    ])

    cache: dict = {}
    blocos_por_mascara: dict = {}

    def metrics_of(tag, arm, model):
        k = (tag, arm, model)
        if k not in cache:
            cache[k] = _masked_draws(counts, resid[k], test, blocos_por_mascara)
        return cache[k]

    families = sorted({MODEL_FAMILY.get(m, "other") for m in models}) + ["all"]
    rows = []
    for name, worse, better in comps:
        for metric in METRICS:
            per: dict = {}  # (tag, model) -> (B+1,)
            for t in tags:
                for m in models:
                    w, bt = metrics_of(t, worse, m), metrics_of(t, better, m)
                    # os dois lados têm de cobrir EXATAMENTE as mesmas linhas;
                    # senão a diferença misturaria trimestres diferentes
                    if w is not None and bt is not None and np.array_equal(w[1], bt[1]):
                        per[(t, m)] = _delta(metric, w[0][metric], bt[0][metric])
            for fam in families:
                # O default TEM de ser o mesmo dos dois lados: `families` acima
                # usa `.get(m, "other")`, então sem o default aqui um modelo
                # fora de MODEL_FAMILY nunca casaria com a própria linha
                # `family="other"` — ela sairia vazia, em silêncio. Latente no
                # conjunto fixo de referência; real com o top-5 dinâmico, onde
                # GradientBoosting/Bagging/AdaBoost são candidatos.
                keep = {k: v for k, v in per.items()
                        if fam == "all" or MODEL_FAMILY.get(k[1], "other") == fam}
                if not keep:
                    continue
                by_tag = {}
                for (t, _m), v in keep.items():
                    by_tag.setdefault(t, []).append(v)
                tag_means = np.array([np.mean(v, axis=0) for v in by_tag.values()])  # (R, B+1)
                agg = tag_means.mean(axis=0)
                point = float(agg[0])
                se_boot = float(np.std(agg[1:], ddof=1)) if n_boot > 1 else float("nan")
                r = len(tag_means)
                se_rep = float(np.std(tag_means[:, 0], ddof=1) / math.sqrt(r)) if r > 1 else 0.0
                se = math.sqrt(se_boot ** 2 + se_rep ** 2)
                z = point / se if se > 0 else float("nan")
                rows.append({
                    "comparison": name, "worse": worse, "better": better,
                    "family": fam, "metric": metric, "effect": point,
                    "se_boot": se_boot, "se_rep": se_rep, "se": se,
                    "ci_lo": point - 1.96 * se, "ci_hi": point + 1.96 * se,
                    "z": z, "p": math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan"),
                    "n_replicates": r, "n_models": len({k[1] for k in keep}),
                    # Cada trimestre testa num ÚNICO mês, então são ~24 blocos
                    # (um por ano). Com tão poucas unidades independentes o IC
                    # é estreito por construção e tende a ser anticonservador —
                    # quem lê o intervalo precisa ver de quantos blocos ele veio.
                    "n_blocos": blocks.nb,
                })
    eff = pd.DataFrame(rows)

    # Eixo de cada comparação. As `extra_comparisons` não têm arm próprio e são
    # sempre do eixo de features.
    axis_of = {a.name: a.axis for a in arms}
    eff["axis"] = eff["comparison"].map(axis_of).fillna("features")
    eff.loc[eff["comparison"].isin(SELECTION_EXTRA), "axis"] = SELECTION_AXIS
    anchor_of = {a.name: a.reference for a in arms}

    eff["q_bh"] = np.nan
    eff["sesoi"] = np.nan
    eff["verdict"] = ""
    for axis, grupo in eff.groupby("axis"):
        metric = PRIMARY_METRIC.get(axis, "rmse")
        # SESOI ancorado na MESMA métrica com que o eixo é julgado, medida no
        # arm de referência. O eixo de features mantém ("base", "rmse") — é o
        # que preserva os valores já publicados.
        anchors = {anchor_of.get(c) or "base" for c in grupo["comparison"]}
        escala = np.mean([v for v in (_anchor_metric(counts, test, blocos_por_mascara, resid, tags, models, a, metric)
                                      for a in sorted(anchors)) if np.isfinite(v)] or [np.nan])
        eff.loc[grupo.index, "sesoi"] = SESOI_REL * escala if np.isfinite(escala) else np.nan

        # Benjamini-Hochberg POR EIXO: são famílias de hipóteses distintas, e o
        # BH multiplica o p pelo tamanho da família — juntá-las mudaria os
        # q-valores do eixo de features só por existir um eixo novo.
        julgado = grupo.index[(grupo["metric"] == metric) & (grupo["family"] == "all")]
        com_p = julgado[eff.loc[julgado, "p"].notna()]
        if len(com_p):
            eff.loc[com_p, "q_bh"] = benjamini_hochberg(eff.loc[com_p, "p"].to_numpy())
        eff.loc[julgado, "verdict"] = [_verdict(r) for _, r in eff.loc[julgado].iterrows()]
    return eff


def _anchor_metric(counts, test, cache, resid, tags, models, arm, metric="rmse") -> float:
    """Escala do arm de referência na métrica com que o eixo é julgado."""
    vals = []
    for t in tags:
        for m in models:
            e = resid.get((t, arm, m))
            r = None if e is None else _masked_draws(counts[:1], e, test, cache)
            if r is not None:
                vals.append(r[0][metric][0])
    return float(np.mean(vals)) if vals else float("nan")


def _verdict(r) -> str:
    s = r["sesoi"]
    if not np.isfinite(s) or not np.isfinite(r["se"]):
        return "indeterminado"
    if r["ci_lo"] > s:
        return "acrescenta"
    if r["ci_hi"] < -s:
        return "prejudica"
    if r["ci_lo"] >= -s and r["ci_hi"] <= s:
        return "sem informação"
    return "inconclusivo"


def loss_frontier(effects: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por braço de perda: o que se ganha na cauda e o que se PAGA no
    global.

    A cauda não é de graça — uma perda assimétrica deixa de estimar a média
    condicional e piora o RMSE por construção. Pôr ganho e preço na mesma linha
    é o que torna a tabela uma fronteira de escolha em vez de um argumento de um
    lado só. `tail_gain_per_rmse_cost` é a taxa de troca; abaixo de 1 a cauda
    sai cara.
    """
    if effects.empty or "axis" not in effects or not (effects["axis"] == "loss").any():
        return pd.DataFrame()
    eff = effects[(effects["axis"] == "loss") & (effects["family"] == "all")]
    por_metrica = {m: g.set_index("comparison") for m, g in eff.groupby("metric")}
    cauda = por_metrica["rmse_p90"]

    out = pd.DataFrame({
        "reference": cauda["worse"],
        "loss": [c.split("__")[-1] for c in cauda.index],
        "rmse_p90_gain": cauda["effect"],
        "ci_lo": cauda["ci_lo"], "ci_hi": cauda["ci_hi"],
        "q_bh": cauda["q_bh"], "sesoi": cauda["sesoi"], "verdict": cauda["verdict"],
        "n_models": cauda["n_models"],
    })
    # Sinal: `effect` já é ganho (positivo = melhor). O RMSE global é preço, por
    # isso entra invertido.
    out["rmse_cost"] = -por_metrica["rmse"]["effect"].reindex(out.index)
    out["r2_cost"] = -por_metrica["r2"]["effect"].reindex(out.index)
    out["bias_p90_shift"] = por_metrica["bias_p90"]["effect"].reindex(out.index)
    out["tail_gain_per_rmse_cost"] = out["rmse_p90_gain"] / out["rmse_cost"].replace(0, np.nan)
    return out.reset_index().sort_values("rmse_p90_gain", ascending=False)


def loss_by_model(out_dir, data_dir, tags, arms=None, models=REFERENCE_MODELS) -> pd.DataFrame:
    """Efeito de cada perda POR MODELO (ponto, sem IC).

    A média da família esconde que os modelos não respondem igual: o XGBoost
    reage bem menos a `tau` que CatBoost e LightGBM, e diluiria a média sem
    que ninguém visse. Esconder um modelo que não responde seria selecionar
    pelo resultado; mostrá-lo por linha é o contrário disso.
    """
    arms = arms or load_arms(data_dir)
    de_perda = [a for a in arms if a.axis == "loss"]
    if not de_perda:
        return pd.DataFrame()
    test = load_test(data_dir)
    blocks = make_blocks(test)
    resid = load_residuals(out_dir, tags, [a.name for a in arms], test, models)
    counts = np.ones((1, blocks.nb))

    rows = []
    for a in de_perda:
        for m in models:
            for t in tags:
                ew, eb = resid.get((t, a.reference, m)), resid.get((t, a.name, m))
                if ew is None or eb is None or not (np.isfinite(ew).all() and np.isfinite(eb).all()):
                    continue
                w, b = (_draw_metrics(counts, e, blocks) for e in (ew, eb))
                rows.append({"comparison": a.name, "reference": a.reference, "loss": a.loss,
                             "model": m, "tag": t,
                             **{k: float(_delta(k, w[k], b[k])[0]) for k in METRICS}})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["comparison", "reference", "loss", "model"], as_index=False)[list(METRICS)].mean()


CHAMPION_RULES = ("r2", "rmse_p90", "r2_slack_then_rmse_p90")


def champion_rules(out_dir, tags, slack: float = CHAMPION_SLACK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara REGRAS de escolha de campeão sobre os resultados que já existem.

    Maximizar R² é minimizar erro quadrático, cujo minimizador é a média
    condicional — a seleção por R² prefere sistematicamente o modelo mais
    COMPRIMIDO, que é o oposto do que um produto de extremos precisa. A regra
    com folga corrige isso sem aceitar um modelo ruim: entre os que estão a até
    `slack` do melhor R², fica o de menor RMSE_P90. Sem a folga, a melhor cauda
    pode ser um modelo com R² negativo (medido no DJF: RANSAC, R² = −0,349).

    Escolha SEMPRE na validação, pontuação SEMPRE no teste — escolher no teste
    seria medir o próprio teste. É medição: nada aqui altera a produção, que
    segue em `src/dataset/creation/best_model_selector.py`.
    """
    lb = models_leaderboard(out_dir, tags)
    if lb.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = lb[lb["split"] == "val"]
    test = lb[lb["split"] == "test"].set_index(["arm", "season", "model"])

    def escolhe(g: pd.DataFrame, rule: str) -> str:
        g = g.dropna(subset=["R2", "RMSE_P90"])
        if g.empty:
            return ""
        if rule == "r2":
            return g.loc[g["R2"].idxmax(), "model"]
        if rule == "rmse_p90":
            return g.loc[g["RMSE_P90"].idxmin(), "model"]
        elegiveis = g[g["R2"] >= g["R2"].max() - slack]
        return elegiveis.loc[elegiveis["RMSE_P90"].idxmin(), "model"]

    rows = []
    for (arm, season), g in val.groupby(["arm", "season"]):
        for rule in CHAMPION_RULES:
            escolhido = escolhe(g, rule)
            chave = (arm, season, escolhido)
            if not escolhido or chave not in test.index:
                continue
            t = test.loc[chave]
            rows.append({"arm": arm, "season": season, "rule": rule, "model": escolhido,
                         **{f"test_{c}": float(t[c]) for c in ("R2", "RMSE", "Bias_P90", "RMSE_P90")}})
    picks = pd.DataFrame(rows)
    if picks.empty:
        return picks, pd.DataFrame()
    resumo = picks.groupby("rule", as_index=False).agg(
        n=("model", "size"), n_modelos=("model", "nunique"),
        **{c: (c, "mean") for c in picks.columns if c.startswith("test_")})
    return picks, resumo


def effects_by_season(out_dir, data_dir, tags, arms=None, models=None) -> pd.DataFrame:
    """Ganho de RMSE por trimestre (só ponto, sem IC — poucos blocos por trimestre)."""
    models = discover_models(out_dir, tags) if models is None else list(models)
    arms = arms or load_arms(data_dir)
    test = load_test(data_dir)
    resid = load_residuals(out_dir, tags, [a.name for a in arms], test, models)
    have = {a.name for a in arms if any(np.isfinite(resid[(t, a.name, m)]).any() for t in tags for m in models)}
    rows = []
    for name, worse, better in comparisons(arms, have):
        for season in sorted(test["season"].unique()):
            mask = (test["season"] == season).to_numpy()
            gains = []
            for t in tags:
                for m in models:
                    ew, eb = resid[(t, worse, m)][mask], resid[(t, better, m)][mask]
                    if np.isfinite(ew).all() and np.isfinite(eb).all():
                        gains.append(math.sqrt(np.mean(ew ** 2)) - math.sqrt(np.mean(eb ** 2)))
            if gains:
                rows.append({"comparison": name, "season": season, "gain_rmse": float(np.mean(gains))})
    return pd.DataFrame(rows)


# ── Ranking ─────────────────────────────────────────────────────────────────

def ranking_features(effects: pd.DataFrame, arms: list[Arm]) -> pd.DataFrame:
    """Por feature: A (ganho isolado, base+f vs base) e L (perda ao remover,
    full vs full−f). L ≈ 0 com A > 0 sugere redundância com as demais novas —
    leitura heurística, não um estimador."""
    e = effects[(effects["metric"] == "rmse") & (effects["family"] == "all")].set_index("comparison")
    rows = []
    for a in arms:
        if a.kind != "add":
            continue
        add = e.loc[a.name] if a.name in e.index else None
        d_name = f"drop__{a.subject}"
        drop = e.loc[d_name] if d_name in e.index else None
        rows.append({
            "feature": a.subject,
            "A_gain_rmse": None if add is None else add["effect"],
            "A_ci_lo": None if add is None else add["ci_lo"],
            "A_ci_hi": None if add is None else add["ci_hi"],
            "A_q_bh": None if add is None else add["q_bh"],
            "A_verdict": "" if add is None else add["verdict"],
            "L_loss_rmse": None if drop is None else drop["effect"],
            "L_ci_lo": None if drop is None else drop["ci_lo"],
            "L_ci_hi": None if drop is None else drop["ci_hi"],
            "L_verdict": "" if drop is None else drop["verdict"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("A_gain_rmse", ascending=False) if len(out) else out


def ranking_groups(effects: pd.DataFrame) -> pd.DataFrame:
    e = effects[(effects["metric"] == "rmse") & (effects["family"] == "all")]
    keep = e[e["comparison"].str.startswith(("add_grp__", "drop_grp__", "ctrl__", "sel__"))
             | e["comparison"].isin(["real_vs_perm_static", "full_vs_base", *SELECTION_EXTRA])]
    return keep[["comparison", "effect", "ci_lo", "ci_hi", "q_bh", "sesoi", "verdict"]].reset_index(drop=True)


# ── Descrição (não usada em inferência) ─────────────────────────────────────

def models_leaderboard(out_dir, tags) -> pd.DataFrame:
    """Métricas dos 39 modelos por arm — tabela DESCRITIVA. Inclui modelos
    degenerados; o efeito é medido só no conjunto de referência."""
    frames = [pd.read_parquet(p) for t in tags for p in sorted((Path(out_dir) / "units" / t).glob("metrics__*.parquet"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    cols = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
    if "n_nonfinite" not in df.columns:  # unidades anteriores à coluna
        df["n_nonfinite"] = 0
    keys = ["arm", "season", "model", "split"]
    out = df.groupby(keys, as_index=False)[cols + ["fit_seconds"]].mean()
    # Modelos que divergiram (predição NaN/inf) aparecem com métrica vazia; a
    # contagem os torna localizáveis em vez de sumirem na média.
    return out.merge(df.groupby(keys, as_index=False)["n_nonfinite"].sum(), on=keys)


def models_leaderboard_by_seed(out_dir, tags) -> pd.DataFrame:
    """Mesma tabela do leaderboard, SEM promediar: uma linha por seed (coluna
    `tag`). É daqui que se mede a dispersão entre seeds."""
    frames = []
    for t in tags:
        for p in sorted((Path(out_dir) / "units" / t).glob("metrics__*.parquet")):
            df = pd.read_parquet(p)
            df["tag"] = t
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    cols = ["tag", "arm", "season", "model", "split", "model_seed", "R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
    return pd.concat(frames, ignore_index=True)[cols]


def seed_sensitivity(out_dir, tags) -> pd.DataFrame:
    """Quanto o RMSE de teste de cada modelo varia SÓ por trocar a seed (mesmos
    dados, mesmo arm, mesmo trimestre). Desvio 0 = modelo determinístico: a seed
    não o afeta e repeti-lo não acrescenta informação."""
    by_seed = models_leaderboard_by_seed(out_dir, tags)
    if by_seed.empty or by_seed["tag"].nunique() < 2:
        return pd.DataFrame()
    test = by_seed[by_seed["split"] == "test"]
    sd = test.groupby(["model", "arm", "season"])["RMSE"].std(ddof=1)
    out = sd.groupby("model").agg(sd_rmse_mean="mean", sd_rmse_max="max").reset_index()
    out["estocastico"] = out["sd_rmse_max"] > 1e-9
    out["n_seeds"] = by_seed["tag"].nunique()
    return out.sort_values("sd_rmse_mean", ascending=False).reset_index(drop=True)


def champions(out_dir, tags) -> pd.DataFrame:
    """Campeão por arm (melhor R² de validação) — só descritivo: a troca de
    campeão entre arms confunde o efeito da feature com o do algoritmo."""
    lb = models_leaderboard(out_dir, tags)
    if lb.empty:
        return lb
    val = lb[lb["split"] == "val"].sort_values("R2", ascending=False).drop_duplicates(["arm", "season"])
    test = lb[lb["split"] == "test"].set_index(["arm", "season", "model"])
    val = val.assign(**{f"test_{c}": [test.loc[(r.arm, r.season, r.model), c] if (r.arm, r.season, r.model) in test.index else np.nan
                                      for r in val.itertuples()] for c in ("R2", "RMSE", "Bias_P90", "RMSE_P90")})
    return val[["arm", "season", "model", "R2", "test_R2", "test_RMSE", "test_Bias_P90", "test_RMSE_P90"]].rename(columns={"R2": "val_R2"})


# Dois modelos cujas métricas de VALIDAÇÃO diferem menos que isto são tratados
# como o mesmo: `LinearRegression`, `Ridge`, `RidgeCV` e `TransformedTargetRegressor`
# (que por padrão é uma regressão linear) saem com R² e RMSE_P90 idênticos até a
# terceira casa, e ocupavam quatro das cinco vagas de JJA.
EQUIV_TOL_R2 = 0.002
EQUIV_TOL_P90 = 0.01


def top_models_by_season(
    out_dir, tags, *, arm: str = "base", k: int = 5,
    slack: float = CHAMPION_SLACK, split_select: str = "val",
    equiv_tol_r2: float = EQUIV_TOL_R2, equiv_tol_p90: float = EQUIV_TOL_P90,
) -> pd.DataFrame:
    """Os `k` melhores modelos POR TRIMESTRE, sobre um único arm.

    É a triagem que substitui rodar os 39 regressores em todos os arms. A regra
    é a mesma de `champion_rules["r2_slack_then_rmse_p90"]`, só que devolvendo
    `k` em vez de 1: entre os que estão a até `slack` do melhor R², ordena por
    RMSE_P90 crescente. Maximizar R² sozinho prefere o modelo mais comprimido,
    que é o oposto do que um produto de extremos precisa; a folga evita o
    extremo oposto, de eleger uma cauda boa com R² negativo.

    Escolha SEMPRE na validação (`split_select`), métricas de teste apenas
    REPORTADAS ao lado — escolher no teste seria medir o próprio teste.

    Sobre a mudança metodológica: `config.py` documenta a decisão de FIXAR os
    modelos a priori, porque escolher o top-K pelo arm base geraria regressão à
    média contra as features. Isso continua valendo para `REFERENCE_MODELS`.
    Aqui o desenho é outro e a condição do bootstrap é preservada: o MESMO
    conjunto de `k` vale para todos os arms daquele trimestre, então o
    pareamento de `compute_effects` (mesmo modelo, trimestre e seed dos dois
    lados) continua intacto. O que se perde é a garantia de que a lista não foi
    tocada pelos dados — o viés resultante é CONSERVADOR contra as features
    novas, já que um modelo que só brilhasse COM elas nunca é visto.
    """
    lb = models_leaderboard(out_dir, tags)
    if lb.empty:
        return pd.DataFrame()

    sel = lb[(lb["arm"] == arm) & (lb["split"] == split_select)].dropna(subset=["R2", "RMSE_P90"])
    # Modelo que divergiu em qualquer seed não entra na triagem: ele voltaria a
    # divergir nos outros arms e levaria um resultado inteiro consigo.
    sel = sel[sel["n_nonfinite"] == 0]
    if sel.empty:
        return pd.DataFrame()
    teste = lb[(lb["arm"] == arm) & (lb["split"] == "test")].set_index(["season", "model"])
    sd = seed_sensitivity(out_dir, tags)
    sd_of = sd.set_index("model")["sd_rmse_mean"].to_dict() if not sd.empty else {}

    linhas = []
    for season, g in sel.groupby("season"):
        elegiveis = g[g["R2"] >= g["R2"].max() - slack].sort_values("RMSE_P90")
        # Se a folga admitir menos que `k`, completa por R² — melhor um quinto
        # modelo medíocre do que um trimestre com menos modelos que os outros,
        # o que desbalancearia a média por trimestre da análise.
        candidatos = list(elegiveis["model"])
        resto = g[~g["model"].isin(candidatos)].sort_values("R2", ascending=False)
        candidatos += list(resto["model"])

        # Modelos equivalentes contam UMA vez. Sem isto, cinco vagas podem ser
        # duas ideias: o conjunto de "5 modelos" inflaria `n_models` e a
        # diversidade que motivou a triagem não existiria.
        por_modelo = g.set_index("model")[["R2", "RMSE_P90"]]
        escolhidos, omitidos = [], {}
        for modelo in candidatos:
            r2, p90 = por_modelo.loc[modelo, "R2"], por_modelo.loc[modelo, "RMSE_P90"]
            gemeo = next((e for e in escolhidos
                          if abs(por_modelo.loc[e, "R2"] - r2) <= equiv_tol_r2
                          and abs(por_modelo.loc[e, "RMSE_P90"] - p90) <= equiv_tol_p90), None)
            if gemeo is not None:
                omitidos.setdefault(gemeo, []).append(modelo)
            else:
                escolhidos.append(modelo)
            if len(escolhidos) == k:
                break
        for rank, modelo in enumerate(escolhidos, start=1):
            linha_val = g[g["model"] == modelo].iloc[0]
            t = teste.loc[(season, modelo)] if (season, modelo) in teste.index else None
            linhas.append({
                "season": season, "rank": rank, "model": modelo,
                "R2_val": float(linha_val["R2"]), "RMSE_P90_val": float(linha_val["RMSE_P90"]),
                "R2_test": float(t["R2"]) if t is not None else np.nan,
                "RMSE_test": float(t["RMSE"]) if t is not None else np.nan,
                "RMSE_P90_test": float(t["RMSE_P90"]) if t is not None else np.nan,
                # Gap val→teste: com ~680 colunas e 9 estações os controles
                # negativos detectam ganho espúrio de FEATURE, não sobreajuste
                # global. Este gap é o termômetro que falta.
                "gap_R2_val_test": float(linha_val["R2"] - t["R2"]) if t is not None else np.nan,
                "n_seeds": len(tags),
                "sd_RMSE_seeds": float(sd_of.get(modelo, 0.0)),
                "equivalentes_omitidos": ", ".join(omitidos.get(modelo, [])),
            })
    return pd.DataFrame(linhas).sort_values(["season", "rank"]).reset_index(drop=True)


def top_models_payload(top: pd.DataFrame, *, arm="base", k=5, slack=CHAMPION_SLACK,
                       tags=(), split_select="val") -> dict:
    """JSON congelado da triagem — é ele que viaja até os containers do `fit`.

    Congelar em vez de recalcular é deliberado: se cada unidade refizesse a
    escolha, uma seed a mais no volume mudaria o conjunto de modelos no meio do
    fan-out e metade dos arms sairia com modelos diferentes da outra metade.
    """
    from src.feature_study.worker import _models_fingerprint

    by_season = {s: list(g.sort_values("rank")["model"]) for s, g in top.groupby("season")}
    todos = sorted({m for ms in by_season.values() for m in ms})
    return {
        "schema": 1, "arm": arm, "k": k, "slack": slack,
        "rule": "r2_slack_then_rmse_p90", "split_select": split_select,
        "tags": list(tags), "models_fp": _models_fingerprint(todos),
        "by_season": by_season,
    }


def compare_units(out_dir, tag_a: str, tag_b: str) -> float:
    """Maior diferença absoluta de métrica entre duas execuções da MESMA unidade
    (teste de determinismo do piloto).

    Guiada pelos arquivos de `tag_b` (a REPETIÇÃO): `tag_a` pode ter muito mais
    unidades — a pasta do run principal também guarda o resto do estudo — e só
    as repetidas são comparáveis. Falta em `tag_a` é erro."""
    worst = 0.0
    found = sorted((Path(out_dir) / "units" / tag_b).glob("metrics__*.parquet"))
    if not found:
        raise FileNotFoundError(f"nenhuma unidade em units/{tag_b}")
    for pb in found:
        pa = Path(out_dir) / "units" / tag_a / pb.name
        if not pa.exists():
            raise FileNotFoundError(pa)
        a, b = pd.read_parquet(pa), pd.read_parquet(pb)
        cols = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]
        merged = a.merge(b, on=["arm", "season", "model", "split"], suffixes=("_a", "_b"))
        for c in cols:
            worst = max(worst, float(np.nanmax(np.abs(merged[f"{c}_a"] - merged[f"{c}_b"]))))
    return worst


def plot_ranking(rank: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rank.empty:
        return
    r = rank.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.34 * len(r) + 1.2)), sharey=True)
    for ax, (val, lo, hi, title) in zip(axes, (
        ("A_gain_rmse", "A_ci_lo", "A_ci_hi", "A — ganho isolado (base + f vs base)"),
        ("L_loss_rmse", "L_ci_lo", "L_ci_hi", "L — perda ao remover (full vs full − f)"),
    )):
        v = r[val].astype(float).to_numpy()
        err = np.vstack([v - r[lo].astype(float).to_numpy(), r[hi].astype(float).to_numpy() - v])
        ax.barh(r["feature"], v, xerr=err, color="#2a78d6" if val.startswith("A") else "#c98a1f", capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Δ RMSE (m/s), positivo = a feature ajuda")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_aggregate(out_dir, data_dir, tags, label: str = "main", n_boot: int = BOOTSTRAP_DRAWS) -> dict:
    """Escreve `summary/<label>/*` e devolve os DataFrames principais."""
    out_dir = Path(out_dir)
    arms = load_arms(data_dir)
    dest = out_dir / "summary" / label
    dest.mkdir(parents=True, exist_ok=True)

    effects = compute_effects(out_dir, data_dir, tags, arms, n_boot=n_boot)
    rank_f, rank_g = ranking_features(effects, arms), ranking_groups(effects)
    by_season = effects_by_season(out_dir, data_dir, tags, arms)

    effects.to_csv(dest / "effects.csv", index=False)
    rank_f.to_csv(dest / "ranking_features.csv", index=False)
    rank_g.to_csv(dest / "ranking_groups.csv", index=False)
    by_season.to_csv(dest / "effects_by_season.csv", index=False)
    models_leaderboard(out_dir, tags).to_csv(dest / "models_leaderboard.csv", index=False)
    by_seed = models_leaderboard_by_seed(out_dir, tags)
    if not by_seed.empty and by_seed["tag"].nunique() > 1:      # com 1 seed não há o que comparar
        by_seed.to_csv(dest / "models_leaderboard_by_seed.csv", index=False)
        seed_sensitivity(out_dir, tags).to_csv(dest / "seed_sensitivity.csv", index=False)
    champions(out_dir, tags).to_csv(dest / "champions.csv", index=False)
    plot_ranking(rank_f, dest / "ranking_features.png")

    # Regras de campeão: medição, roda sempre (independe do eixo de perda).
    picks, resumo = champion_rules(out_dir, tags)
    if not picks.empty:
        picks.to_csv(dest / "champion_rules.csv", index=False)
        resumo.to_csv(dest / "champion_rules_summary.csv", index=False)

    # Saídas do eixo de perda só quando ele existe — uma execução só de features
    # continua gravando exatamente o que gravava antes.
    frontier = loss_frontier(effects)
    if not frontier.empty:
        frontier.to_csv(dest / "loss_frontier.csv", index=False)
        loss_by_model(out_dir, data_dir, tags, arms).to_csv(dest / "loss_by_model.csv", index=False)

    (dest / "meta.json").write_text(json.dumps({"tags": list(tags), "n_boot": n_boot,
                                                "reference_models": list(REFERENCE_MODELS),
                                                "sesoi_rel": SESOI_REL,
                                                "primary_metric": PRIMARY_METRIC,
                                                "loss_models": list(LOSS_MODELS),
                                                "champion_slack": CHAMPION_SLACK}, indent=2))
    return {"effects": effects, "ranking_features": rank_f, "ranking_groups": rank_g,
            "loss_frontier": frontier, "champion_rules": resumo}
