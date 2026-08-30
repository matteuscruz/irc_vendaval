"""Seleção do melhor modelo (pipeline × configuração de ablation) por
cluster × trimestre climático, lendo os artefatos LOCAIS já treinados
(results.csv / fitted_models / predictions_by_station.csv).

Porta a lógica já prototipada no dashboard (irc_vendaval_dashboard/app.py:
`_best_combo_per_cluster`, `_sdf_from_cluster_winners`,
`_ablation_select_split`, `_resolve_all_season`, `derive_quarterly_results`)
pra ler os artefatos deste repo em vez do parquet sincronizado do dashboard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.common import compute_metrics, month_to_season

_MODEL_STEM_RE = re.compile(r"^best_model_c(\d+)(?:_(DJF|MAM|JJA|SON))?$")

PIPELINES = ["lazy", "mlp", "lstm"]
ARMS = ["original", "synthetic", "newfeatures", "all", "basin", "all_basin"]
SEASONS_ORDER = ["DJF", "MAM", "JJA", "SON"]
METRICS = ["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"]

# Mesma direção de scripts/_ablation_common.py::METRIC_DIRECTIONS — duplicado
# aqui (não importado) porque src/ não deve depender de scripts/ (scripts/
# não é um pacote instalável, só um diretório de utilitários standalone).
METRIC_DIRECTIONS = {
    "R2": "higher", "RMSE": "lower", "Bias": "zero",
    "Bias_P90": "zero", "RMSE_P90": "lower",
}

# arm -> feature_groups, mesma matriz de scripts/_ablation_common.py::ABLATION_MATRIX.
ARM_FEATURE_GROUPS = {
    "original": "original",
    "synthetic": "original",
    "newfeatures": "original,era5_18z,bt55",
    "all": "original,era5_18z,bt55",
    "basin": "original,era5_basin",
    "all_basin": "original,era5_18z,bt55,era5_basin",
}

# pipeline -> raiz local dos artefatos, relativa a --artifacts-root.
PIPELINE_ROOTS = {
    "lazy": Path("lazy_modal/lazy_clusters"),
    "mlp": Path("mlp_modal/mlp_clusters"),
    "lstm": Path("modal/experiments"),
}


@dataclass(frozen=True)
class Combo:
    pipeline: str
    arm: str
    dir: Path
    feature_groups: str

    @property
    def results_csv(self) -> Path:
        return self.dir / "results.csv"

    @property
    def models_dir(self) -> Path:
        return self.dir / "fitted_models"

    @property
    def preds_by_station(self) -> Path:
        return self.dir / "predictions" / "predictions_by_station.csv"

    def available_cluster_seasons(self) -> set[tuple[int, str | None]]:
        """(cluster_id, season) com modelo salvo — season=None é o modelo
        pooled (ano inteiro; sempre o caso pra mlp/lstm, que não gravam
        arquivo por trimestre). Parseia só o nome do arquivo (sem
        joblib.load) por velocidade — roda a cada winner table. `newfeatures`
        e `all`(lstm) só têm modelo pra 5-6 clusters, por causa do
        `--restrict-coverage` (estações fora da cobertura era5_18z/bt55 são
        descartadas antes do treino)."""
        if not self.models_dir.exists():
            return set()
        stems = [p.stem for p in self.models_dir.glob("best_model_c*.joblib")]
        stems += [p.stem for p in self.models_dir.glob("best_model_c*.keras")]
        out: set[tuple[int, str | None]] = set()
        for s in stems:
            m = _MODEL_STEM_RE.match(s)
            if not m:
                continue
            out.add((int(m.group(1)), m.group(2)))
        return out

    def available_clusters(self) -> set[int]:
        """Clusters com pelo menos um modelo salvo (qualquer trimestre)."""
        return {cid for cid, _ in self.available_cluster_seasons()}


def _to_int_cluster(cid) -> int | None:
    """cluster_id pode vir como '1', 1 ou '1.0'. Rótulos de cluster-merge
    (ex.: '1-2') não são suportados aqui (o grid usa os 14 clusters
    originais) — retorna None pra serem descartados."""
    try:
        return int(float(cid))
    except (TypeError, ValueError):
        return None


def discover_combos(artifacts_root: str | Path = "artifacts") -> list[Combo]:
    """Varre o filesystem por (pipeline, arm) com `results.csv` disponível —
    nunca hardcoda quais braços existem: `basin`/`all_basin` hoje não têm
    artefatos locais (nunca rodados), loga aviso em vez de assumir."""
    root = Path(artifacts_root)
    combos: list[Combo] = []
    missing: list[str] = []
    for pipeline in PIPELINES:
        base = root / PIPELINE_ROOTS[pipeline]
        for arm in ARMS:
            combo_dir = base / arm
            if (combo_dir / "results.csv").exists():
                combos.append(Combo(
                    pipeline=pipeline, arm=arm, dir=combo_dir,
                    feature_groups=ARM_FEATURE_GROUPS[arm],
                ))
            else:
                missing.append(f"{pipeline}/{arm}")
    if missing:
        print(
            f"[best_model_selector] AVISO: {len(missing)} combinação(ões) "
            f"sem results.csv local, ignoradas: {missing}"
        )
    return combos


def load_results(combos: list[Combo]) -> pd.DataFrame:
    """Concatena results.csv de todos os combos — pipeline/arm são
    sobrescritos com o valor descoberto por discover_combos (mais robusto a
    inconsistências do que confiar cegamente no que já está no CSV)."""
    frames = []
    for c in combos:
        if not c.results_csv.exists():
            continue
        df = pd.read_csv(c.results_csv)
        df["pipeline"] = c.pipeline
        df["arm"] = c.arm
        df["cluster_id"] = df["cluster_id"].apply(_to_int_cluster)
        df = df.dropna(subset=["cluster_id"])
        df["cluster_id"] = df["cluster_id"].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_station_predictions(combo: Combo) -> pd.DataFrame:
    """Carrega predictions_by_station.csv com 'time' parseado e cluster_id
    normalizado; deduplica linhas repetidas (bug conhecido do lazy: 2 linhas
    por (estacao,time,split) — vem do run cluster-global + run
    por-trimestre, sem coluna que diferencie as duas origens)."""
    if not combo.preds_by_station.exists():
        return pd.DataFrame()
    df = pd.read_csv(combo.preds_by_station, parse_dates=["time"])
    df["cluster_id"] = df["cluster_id"].apply(_to_int_cluster)
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    dedup_cols = [c for c in ("estacao", "time", "split") if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)
    return df


def derive_quarterly_from_predictions(combo: Combo, results: pd.DataFrame) -> pd.DataFrame:
    """Só pra combos cujo results.csv não tem granularidade de trimestre
    nativa (hoje: mlp, que reporta só season='ALL') — deriva DJF/MAM/JJA/SON
    a partir de predictions_by_station.csv, usando a métrica REAL de
    src.pipelines.common.compute_metrics (não uma reimplementação)."""
    seasons_here = results[
        (results["pipeline"] == combo.pipeline) & (results["arm"] == combo.arm)
    ]["season"].unique()
    if any(s != "ALL" for s in seasons_here):
        return pd.DataFrame()  # já tem granularidade nativa — não duplica.

    preds = load_station_predictions(combo)
    if preds.empty or "time" not in preds.columns:
        return pd.DataFrame()
    if "split" not in preds.columns:
        # predictions_by_station.csv de um schema mais antigo (sem coluna
        # 'split') — não dá pra derivar season×split com confiança; pula
        # esse combo em vez de derrubar o winner table inteiro.
        print(
            f"[best_model_selector] AVISO: {combo.pipeline}/{combo.arm} — "
            "predictions_by_station.csv sem coluna 'split', pulando derivação "
            "de trimestre pra esse combo."
        )
        return pd.DataFrame()
    preds = preds.copy()
    preds["season"] = month_to_season(preds["time"].dt.month)

    rows = []
    for (cid, season, split), g in preds.groupby(["cluster_id", "season", "split"]):
        if len(g) < 2:
            continue
        metrics = compute_metrics(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
        rows.append({
            "pipeline": combo.pipeline, "arm": combo.arm, "experiment": combo.arm,
            "cluster_id": cid, "season": season, "split": split,
            "n_samples": len(g), **metrics,
        })
    return pd.DataFrame(rows)


def load_all_results_with_derived(combos: list[Combo]) -> pd.DataFrame:
    """`load_results` + `derive_quarterly_from_predictions` pros combos que
    precisarem — ponto de entrada único usado pelo CLI e pelo grid_generator."""
    results = load_results(combos)
    derived = [derive_quarterly_from_predictions(c, results) for c in combos]
    derived = [d for d in derived if not d.empty]
    if derived:
        results = pd.concat([results] + derived, ignore_index=True)
    return results


def _select_split(df: pd.DataFrame) -> pd.DataFrame:
    """Por (pipeline, arm), prefere split='test'; senão usa o único disponível."""
    if df.empty:
        return df
    frames = []
    for _, g in df.groupby(["pipeline", "arm"]):
        splits = set(g["split"].unique())
        chosen = "test" if "test" in splits else sorted(splits)[0]
        frames.append(g[g["split"] == chosen])
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Agrega sobre season (média ponderada por n_samples)."""
    if df.empty:
        return df
    rows = []
    for key, g in df.groupby(group_cols):
        w = g["n_samples"]
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        for m in METRICS:
            row[m] = float(np.average(g[m], weights=w)) if w.sum() > 0 else float("nan")
        row["n_samples"] = int(w.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_all_season(all_results: pd.DataFrame) -> pd.DataFrame:
    """Linhas efetivas pra season='ALL': usa a linha nativa quando o combo
    já reporta uma (lazy/mlp), e DERIVA 'ALL' pra quem só reporta por
    trimestre (lstm não tem ALL nativo) — agregando as 4 linhas de
    trimestre (split='test', ponderado por n_samples) por
    (pipeline, arm, cluster_id)."""
    if all_results.empty:
        return all_results
    native = all_results[all_results["season"] == "ALL"]
    covered = set(zip(native["pipeline"], native["arm"]))
    quarterly = all_results[
        (all_results["season"] != "ALL")
        & ~all_results.apply(lambda r: (r["pipeline"], r["arm"]) in covered, axis=1)
    ]
    quarterly_test = _select_split(quarterly)
    if quarterly_test.empty:
        return native
    derived = _aggregate(quarterly_test, ["pipeline", "arm", "cluster_id"])
    if derived.empty:
        return native
    derived["season"] = "ALL"
    derived["split"] = "test"
    return pd.concat([native, derived], ignore_index=True)


def _sort_candidates(g: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Ordena candidatos do melhor pro pior, conforme a direção da métrica."""
    direction = METRIC_DIRECTIONS.get(metric, "higher")
    if direction == "higher":
        return g.sort_values(metric, ascending=False)
    if direction == "lower":
        return g.sort_values(metric, ascending=True)
    return g.reindex(g[metric].abs().sort_values().index)  # "zero": mais perto de 0


def select_winners(
    results: pd.DataFrame, metric: str = "R2", season: str | None = None,
) -> dict[int, tuple[str, str]]:
    """Pra cada cluster, a combinação (pipeline, arm) vencedora pela métrica
    dada. season=None usa a visão agregada 'ALL' (ano inteiro); senão usa só
    as linhas (nativas ou derivadas) daquele trimestre específico."""
    if results.empty:
        return {}
    if season is None:
        selected = _select_split(resolve_all_season(results))
    else:
        selected = _select_split(results[results["season"] == season])
    if selected.empty:
        return {}
    winners: dict[int, tuple[str, str]] = {}
    for cid, g in selected.groupby("cluster_id"):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        best = _sort_candidates(g, metric).iloc[0]
        winners[int(cid)] = (best["pipeline"], best["arm"])
    return winners


def select_winner_table(
    results: pd.DataFrame, metric: str = "R2", seasons: list[str] = SEASONS_ORDER,
) -> pd.DataFrame:
    """Tabela com 1 linha por (cluster_id, season) — inclui também season
    'ALL' (ano inteiro), útil como referência/auditoria mesmo não sendo
    usada diretamente pelo grid (que é sempre por trimestre)."""
    rows = []
    for season in ["ALL", *seasons]:
        season_arg = None if season == "ALL" else season
        selected = (
            _select_split(resolve_all_season(results)) if season_arg is None
            else _select_split(results[results["season"] == season_arg])
        )
        if selected.empty:
            continue
        for cid, g in selected.groupby("cluster_id"):
            g = g.dropna(subset=[metric])
            if g.empty:
                continue
            best = _sort_candidates(g, metric).iloc[0]
            rows.append({
                "cluster_id": int(cid), "season": season,
                "pipeline": best["pipeline"], "arm": best["arm"],
                "metric": metric, "metric_value": float(best[metric]),
                "n_samples": int(best["n_samples"]),
            })
    return pd.DataFrame(rows)


def apply_fitted_model_fallback(
    table: pd.DataFrame, combos: list[Combo], results: pd.DataFrame, metric: str,
) -> pd.DataFrame:
    """Quando o vencedor não tem modelo salvo pro cluster (gap do
    `--restrict-coverage` em `newfeatures`/`all`-lstm), desce pro próximo
    candidato do ranking até achar um com `fitted_models`. Registra a troca
    em `fallback_from` (None se o vencedor original já tinha modelo)."""
    if table.empty:
        table = table.copy()
        table["has_fitted_model"] = pd.Series(dtype=bool)
        table["fallback_from"] = pd.Series(dtype=object)
        return table

    combo_by_key = {(c.pipeline, c.arm): c for c in combos}
    table = table.copy()
    table["has_fitted_model"] = False
    table["fallback_from"] = None

    for idx, row in table.iterrows():
        cid, season = row["cluster_id"], row["season"]
        season_arg = None if season == "ALL" else season
        combo = combo_by_key.get((row["pipeline"], row["arm"]))
        if combo is not None:
            avail = combo.available_cluster_seasons()
            # Prefere o modelo específico do trimestre; cai pro pooled
            # (season=None) só se não houver um por-trimestre salvo — cobre
            # tanto lazy (que agora pode ter os dois) quanto mlp/lstm (que só
            # têm pooled, já sempre presente em `avail` como (cid, None)).
            if (cid, season_arg) in avail or (cid, None) in avail:
                table.at[idx, "has_fitted_model"] = True
                continue

        selected = (
            _select_split(resolve_all_season(results)) if season_arg is None
            else _select_split(results[results["season"] == season_arg])
        )
        g = selected[selected["cluster_id"] == cid].dropna(subset=[metric])
        ranked = _sort_candidates(g, metric) if not g.empty else g

        found = False
        for _, cand in ranked.iterrows():
            cand_combo = combo_by_key.get((cand["pipeline"], cand["arm"]))
            if cand_combo is None:
                continue
            cand_avail = cand_combo.available_cluster_seasons()
            if (cid, season_arg) in cand_avail or (cid, None) in cand_avail:
                table.at[idx, "pipeline"] = cand["pipeline"]
                table.at[idx, "arm"] = cand["arm"]
                table.at[idx, "metric_value"] = float(cand[metric])
                table.at[idx, "n_samples"] = int(cand["n_samples"])
                table.at[idx, "has_fitted_model"] = True
                table.at[idx, "fallback_from"] = f"{row['pipeline']}/{row['arm']}"
                found = True
                break
        if not found:
            print(
                f"[best_model_selector] AVISO: cluster {cid} trimestre {season} "
                "— nenhum candidato com modelo salvo, ficará sem correção nesse recorte."
            )
    return table


def build_winner_table(
    artifacts_root: str | Path = "artifacts", metric: str = "R2",
) -> tuple[pd.DataFrame, list[Combo]]:
    """Ponto de entrada único: descobre combos, carrega resultados (+
    derivados), monta a tabela de vencedores por (cluster, trimestre) com
    fallback pra modelo salvo. Usado pelo CLI e pelo grid_generator."""
    combos = discover_combos(artifacts_root)
    results = load_all_results_with_derived(combos)
    table = select_winner_table(results, metric=metric)
    table = apply_fitted_model_fallback(table, combos, results, metric)
    return table, combos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tabela de vencedores (pipeline × configuração) por cluster × trimestre"
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--metric", default="R2", choices=METRICS)
    args = parser.parse_args()

    winner_table, found_combos = build_winner_table(args.artifacts_root, args.metric)
    print(f"[best_model_selector] {len(found_combos)} combinação(ões) descoberta(s).")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(winner_table.to_string(index=False))
