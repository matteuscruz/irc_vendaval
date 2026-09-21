"""Estágio `prepare`: dados → amostras representativas em parquet.

Roda uma vez (no Modal, com o merge completo do ERA5) e grava tudo o que os
workers precisam; os workers não voltam a tocar em NetCDF.

  data/meta.json, arms.json        constantes, contagens, arms
  data/test.parquet                teste COMPLETO (nunca amostrado)
  data/sample_full.parquet         treino + validação INTEGRAIS (é o que o estudo usa)
  data/sample_r{i}.parquet         só se `seeds` for dado: réplicas amostradas
  data/sample_n{N}_r0.parquet      só se `pilot_sizes` for dado
  data/representativeness.csv      só se houver amostras: diagnósticos vs. população

A amostragem está desligada por padrão (ver `config.SAMPLE_SEEDS`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.feature_study.arms import (
    DEFAULT_ARM_SETS, add_control_columns, build_arms, usable_new_features,
)
from src.feature_study.config import (
    ALPHA, CLUSTER_ID, EPS, EXCLUDED_DUPLICATES, LOSS_MODELS, MODEL_SEED, MODEL_SEEDS,
    PILOT_SIZES, SAMPLE_SEEDS, SEASONS,
)
from src.feature_study.sampling import (
    assert_no_gross_failure, build_replicate, dkw_sample_size, group_sizes,
    representativeness_report, structural_violations,
)
from src.pipelines.common import (
    KNOWN_STATIC_FEATURES, TARGET_VAR, month_to_season, resolve_feature_groups,
    select_complete_rows,
)

TRAINVAL = ("train", "val")


def _population(raw_dir: str, shp_dir: str, cluster_id: int):
    """Frame do cluster sob a UNIÃO de todas as features (mesmas linhas para
    todos os arms) + contagens do que a exigência de completude descartou.

    Carrega com o grupo `original` (as colunas `nf_*` vêm no frame de qualquer
    modo) e aplica a completude da união aqui, em vez de usar
    `feature_groups="original,new_features"` — assim dá para contar o que ela
    remove, o que o loader esconde.
    """
    from src.pipelines.cluster_lazy import load_lazy_training_frame

    df0 = load_lazy_training_frame(raw_dir, shp_dir, "original")
    base = resolve_feature_groups("original")
    new, static = usable_new_features(df0.columns, KNOWN_STATIC_FEATURES)
    if not new:
        raise ValueError("nenhuma feature nova no frame — confira dataset/raw/new_features")

    is_c = df0["cluster_id"].astype(str) == str(cluster_id)
    before = df0[is_c].groupby("_split").size().to_dict()
    union = base + new
    complete = select_complete_rows(df0, union, label="feature_study")

    clusters_kept = sorted(complete["cluster_id"].astype(str).unique())
    if str(cluster_id) not in clusters_kept:
        raise ValueError(f"cluster {cluster_id} não sobreviveu à regra de cobertura (restaram {clusters_kept})")
    dropped_clusters = sorted(set(df0["cluster_id"].astype(str)) - set(clusters_kept))

    pop = complete[complete["cluster_id"].astype(str) == str(cluster_id)].copy()
    after = pop.groupby("_split").size().to_dict()
    info = {
        "rows_before_completeness": {k: int(v) for k, v in before.items()},
        "rows_after_completeness": {k: int(v) for k, v in after.items()},
        "clusters_dropped_by_coverage": dropped_clusters,
    }
    return pop, base, new, static, info


def prepare(
    raw_dir: str, shp_dir: str, out_dir, *,
    eps: float = EPS, alpha: float = ALPHA, seeds=SAMPLE_SEEDS,
    pilot_sizes=PILOT_SIZES, arm_sets=DEFAULT_ARM_SETS, cluster_id: int = CLUSTER_ID,
) -> dict:
    out = Path(out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)

    pop, base, new, static, info = _population(raw_dir, shp_dir, cluster_id)
    pop = add_control_columns(pop, static)
    pop["season"] = month_to_season(pop["time"].dt.month)
    pop["year"] = pop["time"].dt.year
    pop["month"] = pop["time"].dt.month
    pop["row_id"] = range(len(pop))

    arms = build_arms(base, new, static, arm_sets)
    feature_cols = sorted({f for a in arms for f in a.features})
    keep = ["row_id", "estacao", "time", "season", "year", "month", "_split", TARGET_VAR, *feature_cols]
    pop = pop[keep]

    test = pop[pop["_split"] == "test"].reset_index(drop=True)
    trainval = pop[pop["_split"].isin(TRAINVAL)].reset_index(drop=True)
    if test.empty or trainval.empty:
        raise ValueError("sem treino/validação ou sem teste após a completude")
    test.to_parquet(out / "test.parquet", index=False)
    trainval.to_parquet(out / "sample_full.parquet", index=False)

    n_target = dkw_sample_size(eps, alpha) if seeds else None
    if seeds:
        print(f"[prepare] n por trimestre (DKW eps={eps}, alpha={alpha}) = {n_target}", flush=True)
    else:
        print("[prepare] amostragem desligada: o estudo usa o treino COMPLETO", flush=True)

    reports = []
    sizes = {}
    samples = {f"r{i}": (seed, n_target) for i, seed in enumerate(seeds)}
    if pilot_sizes and not seeds:
        raise ValueError("pilot_sizes exige ao menos uma seed em `seeds`")
    for n in pilot_sizes:
        samples[f"n{n}_r0"] = (seeds[0], int(n))

    # Amostras de execuções anteriores ficariam no diretório sem que nada as
    # usasse — e uma amostra velha ao lado da nova convida a analisar a errada.
    keep_names = {f"sample_{tag}.parquet" for tag in samples} | {"sample_full.parquet"}
    for stale in out.glob("sample_*.parquet"):
        if stale.name not in keep_names:
            stale.unlink()
            print(f"[prepare] removida amostra antiga: {stale.name}", flush=True)

    for tag, (seed, n) in samples.items():
        s = build_replicate(trainval, seed, n)
        exp_sizes = group_sizes(trainval, n)
        bad = structural_violations(s, trainval, test, feature_cols, exp_sizes)
        if bad:
            raise ValueError(f"amostra {tag} reprovada nos gates estruturais: {bad}")
        s.to_parquet(out / f"sample_{tag}.parquet", index=False)
        sizes[tag] = {f"{sp}/{se}": int(v) for (sp, se), v in s.groupby(["_split", "season"]).size().items()}
        for season in SEASONS:
            pt = trainval[(trainval["_split"] == "train") & (trainval["season"] == season)]
            st = s[(s["_split"] == "train") & (s["season"] == season)]
            if pt.empty:
                continue
            rep = representativeness_report(pt, st, feature_cols, alpha, scope=f"{tag}/{season}")
            rep.insert(0, "tag", tag)
            reports.append(rep)
        print(f"[prepare] amostra {tag}: {len(s)} linhas (seed {seed}, n={n})", flush=True)

    rep_path = out / "representativeness.csv"
    if reports:
        rep_all = pd.concat(reports, ignore_index=True)
        rep_all.to_csv(rep_path, index=False)
        assert_no_gross_failure(rep_all)
        flags = rep_all["flag"].value_counts().to_dict()
    else:
        rep_path.unlink(missing_ok=True)
        flags = {}

    (out / "arms.json").write_text(json.dumps([a.to_dict() for a in arms], indent=2))
    meta = {
        "cluster_id": cluster_id, "model_seed": MODEL_SEED, "model_seeds": list(MODEL_SEEDS), "sample_seeds": list(seeds),
        "sampling": bool(seeds), "eps": eps, "alpha": alpha, "n_target_per_season": n_target,
        "pilot_sizes": list(pilot_sizes), "arm_sets": list(arm_sets), "n_arms": len(arms),
        "loss_arms": sorted(a.name for a in arms if a.axis == "loss"),
        "loss_models": list(LOSS_MODELS),
        "train_rows_per_season": {
            se: int(((trainval["_split"] == "train") & (trainval["season"] == se)).sum())
            for se in SEASONS
        },
        "base_features": base, "new_features": new, "static_features": static,
        "excluded_duplicates": list(EXCLUDED_DUPLICATES),
        "n_test_rows": len(test), "n_trainval_rows": len(trainval),
        "sample_sizes": sizes, "flags": flags,
        **info,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[prepare] {len(arms)} arms, teste {len(test)} linhas, "
          f"treino por trimestre {meta['train_rows_per_season']}", flush=True)
    return meta
