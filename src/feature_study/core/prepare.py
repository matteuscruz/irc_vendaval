"""Estágio `prepare`: dados → população em parquet.

Roda uma vez (no Modal ou local) e grava tudo o que os workers precisam; os workers
não voltam a tocar nos arquivos de origem.

  data/meta.json, arms.json        constantes, contagens, arms
  data/test.parquet                teste COMPLETO
  data/sample_full.parquet         treino + validação INTEGRAIS (é o que o estudo usa)

O estudo treina no treino COMPLETO, sem amostragem: o cluster 3 tem poucas linhas por
trimestre, e com seeds fixas a única incerteza é o bootstrap em blocos do teste.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_study.core.arms import DEFAULT_ARM_SETS, add_control_columns, build_arms
from src.feature_study.core.config import CLUSTER_ID, MODEL_SEED, MODEL_SEEDS, SEASONS
from src.feature_study.data import groups_source as gs
from src.feature_study.data.purge import purge_days
from src.pipeline.data.splits import LABEL_DTYPE, purge_keep
from src.pipelines.common import (
    ERA5_GUST_PROXY, TARGET_VAR, month_to_season, select_complete_rows,
)

TRAINVAL = ("train", "val")

# Identificam a estação e ficam no parquet para o mapa espacial (coordenada).
CONTEXT_COLUMNS = ("latitude", "longitude")

# Colunas de REFERÊNCIA, preservadas no parquet mas que nunca entram em arm
# nenhum — não são features, são o que se compara contra.
#
# A referência é a RAJADA ERA5 (`era5_gust_max`), não `wind_mag_max`. ATENÇÃO à
# diferença: `wind_mag_max` é o máximo do VENTO MÉDIO a 10 m, não uma rajada — tem
# viés de −5 m/s contra o INMET e faz o modelo parecer corrigir muito mais do que
# corrige. Medido no teste, a rajada ERA5 bruta tem viés de −0,03 m/s (RMSE 2,65
# contra 5,85). Toda comparação "modelo vs ERA5" deve usar a rajada.
#
# Perder a coluna é silencioso: nada falha, só o `era5_proxy` das predições sai
# todo NaN e o mapa de correlação fica vazio.
REFERENCE_CANDIDATES = (gs.REFERENCE_COLUMN, ERA5_GUST_PROXY)


def purge_split_boundaries(pop: pd.DataFrame, gap_days: int) -> tuple[pd.DataFrame, int]:
    """Aplica a purga temporal do projeto (`splits.purge_keep`) nas fronteiras
    de mês, devolvendo `(frame purgado, linhas removidas)`.

    O vazamento que isto fecha é o menos óbvio dos dois. Não é 1º de janeiro
    (teste) puxando dezembro (treino) — isso é a realidade operacional, um
    modelo em produção tem mesmo o passado recente. É o INVERSO: 1º-3 de
    fevereiro são TREINO, e o `tp_roll72h` deles alcança 30-31 de janeiro, que
    é TESTE. A linha de treino passa a conter informação de dias de teste. O
    mesmo em maio, agosto e novembro — todo mês de teste é seguido por um de
    treino.

    O rótulo é função apenas do calendário (mês de teste, ou bloco (ano, mês)
    de validação), nunca da estação — é o que `splits.resolve_month_block_split`
    garante ao sortear a validação por blocos inteiros. Por isso ele é
    recalculado a partir das regras, e não lido linha a linha: um dia ausente
    do frame (falha do INMET) continua existindo no ERA5 e alimenta a janela.
    """
    if gap_days <= 0:
        return pop, 0

    meses_teste = set(pop.loc[pop["_split"] == "test", "month"].unique())
    blocos_val = {
        (int(y), int(m))
        for y, m in pop.loc[pop["_split"] == "val", ["year", "month"]].drop_duplicates().to_numpy()
    }

    def rotulos(times: pd.Series) -> np.ndarray:
        m = times.dt.month.to_numpy()
        codigo = times.dt.year.to_numpy() * 100 + m
        val_codes = [y * 100 + mm for y, mm in blocos_val]
        return np.where(
            np.isin(m, list(meses_teste)), "test",
            np.where(np.isin(codigo, val_codes), "val", "train"),
        ).astype(LABEL_DTYPE)

    janela = np.column_stack([
        rotulos(pop["time"] - pd.Timedelta(days=k)) for k in range(gap_days, 0, -1)
    ] + [pop["_split"].to_numpy().astype(LABEL_DTYPE)])
    keep = purge_keep(janela, pop["_split"].to_numpy().astype(LABEL_DTYPE))
    return pop[keep].reset_index(drop=True), int((~keep).sum())


def _population_era5_groups(raw_dir: str, cluster_id: int, cache_dir):
    """População da estrutura nova: um ponto por (estação, dia), na hora do pico
    da rajada ERA5, alvo INMET, split idêntico ao do estudo anterior.

    Devolve `(df0, base, new, static, group_of, info)`. Nada aqui lê o ERA5
    bruto antigo (`ERA5_Stratified`, `ERA5_Features_Basin`…): só os quatro
    parquets por grupo e o `INMET_Stratified.nc` do alvo — é por isso que o
    modo roda numa máquina sem aqueles arquivos.
    """
    from src.pipelines.common import default_month_block_split

    if not gs.is_available(raw_dir):
        raise FileNotFoundError(
            f"faltam parquets em {gs.group_dir(raw_dir)} — esperados: {sorted(gs.GROUP_FILES.values())}"
        )
    alvo, eixo = gs.load_target(raw_dir)
    com_alvo = set(alvo[gs.STATION].unique())
    estacoes = sorted(com_alvo & set(gs.available_stations(raw_dir)))
    if not estacoes:
        raise ValueError("nenhuma estação tem alvo INMET em 2000–2024 e features nos parquets")

    daily, meta_g = gs.build_daily_table(raw_dir, estacoes, cache_dir=cache_dir)
    group_of = meta_g["group_of"]

    df0 = daily.merge(alvo, on=[gs.STATION, gs.TIME], how="inner")
    # Split no eixo de dias ATÉ 2024: com 2025 os blocos de validação mudam
    # (15 de 32), e o estudo deixaria de ser comparável ao anterior.
    split = default_month_block_split(np.unique(eixo.values))
    df0["_split"] = split.label(pd.DatetimeIndex(df0[gs.TIME]))
    df0["cluster_id"] = str(cluster_id)

    base = sorted(c for c, g in group_of.items() if g == "grupo1")
    estaticas = sorted(c for c, g in group_of.items()
                       if g == gs.STATIC_GROUP)
    new = sorted(c for c, g in group_of.items() if g in ("grupo2", "grupo3")) + estaticas
    info = {
        "row_design": meta_g["row_design"], "peak_column": meta_g["peak_column"],
        "stations": estacoes, "dropped_null_columns": meta_g["dropped_null_columns"],
        "group_of": group_of,
        "split_val_units": sorted([int(y), int(m)] for y, m in split.val_units),
    }
    return df0, base, new, estaticas, group_of, info


def _population(raw_dir: str, cluster_id: int, cache_dir=None):
    """Frame do cluster sob a UNIÃO de todas as features (mesmas linhas para todos
    os arms) + contagens do que a exigência de completude descartou.

    A completude da união é aplicada aqui, e não no loader, para poder contar o
    que ela remove — o que o loader esconderia."""
    df0, base, new, static, _, extra_info = _population_era5_groups(raw_dir, cluster_id, cache_dir)
    if not new:
        raise ValueError("nenhuma feature nova no frame — confira dataset/raw/feature_study_cluster_3")

    is_c = df0["cluster_id"].astype(str) == str(cluster_id)
    before = df0[is_c].groupby("_split").size().to_dict()
    complete = select_complete_rows(df0, base + new, label="feature_study")

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
        **extra_info,
    }
    return pop, base, new, static, info


def prepare(raw_dir: str, out_dir, *, arm_sets=DEFAULT_ARM_SETS, cluster_id: int = CLUSTER_ID) -> dict:
    out = Path(out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)

    pop, base, new, static, info = _population(raw_dir, cluster_id, cache_dir=Path(out_dir) / "_cache")
    group_of = info["group_of"]
    pop = add_control_columns(pop, static)
    pop["season"] = month_to_season(pop["time"].dt.month)
    pop["year"] = pop["time"].dt.year
    pop["month"] = pop["time"].dt.month

    # Gap ÚNICO, derivado do conjunto de features (união = o arm `full`), nunca
    # uma constante solta. Único de propósito: `compute_effects` é pareado por
    # `row_id`, então todos os arms têm de compartilhar exatamente as mesmas
    # linhas — um gap por arm quebraria o pareamento sem erro visível.
    gap = purge_days(base + new)
    pop, purgadas = purge_split_boundaries(pop, gap)
    if gap:
        print(f"[prepare] purga temporal: gap de {gap} dia(s), {purgadas} linhas removidas "
              "das fronteiras de mês", flush=True)
    pop["row_id"] = range(len(pop))

    arms = build_arms(base, new, static, arm_sets, group_of=group_of)
    feature_cols = sorted({f for a in arms for f in a.features})
    referencia = next((c for c in REFERENCE_CANDIDATES if c in pop.columns), None)
    if referencia is None and not any(c in feature_cols for c in REFERENCE_CANDIDATES):
        raise ValueError(
            f"nenhuma coluna de referência ERA5 na população (esperada: {REFERENCE_CANDIDATES}) — "
            "sem ela não há mapa espacial nem comparação contra o ERA5, e a perda seria silenciosa"
        )
    contexto = [c for c in (*CONTEXT_COLUMNS, referencia)
                if c and c in pop.columns and c not in feature_cols]
    keep = ["row_id", "estacao", "time", "season", "year", "month", "_split", TARGET_VAR,
            *contexto, *feature_cols]
    pop = pop[keep]

    test = pop[pop["_split"] == "test"].reset_index(drop=True)
    trainval = pop[pop["_split"].isin(TRAINVAL)].reset_index(drop=True)
    if test.empty or trainval.empty:
        raise ValueError("sem treino/validação ou sem teste após a completude")
    test.to_parquet(out / "test.parquet", index=False)
    trainval.to_parquet(out / "sample_full.parquet", index=False)

    (out / "arms.json").write_text(json.dumps([a.to_dict() for a in arms], indent=2))
    meta = {
        "cluster_id": cluster_id, "model_seed": MODEL_SEED, "model_seeds": list(MODEL_SEEDS),
        "arm_sets": list(arm_sets), "n_arms": len(arms),
        # Gap da purga temporal e quanto ele custou. Vai para o meta porque é
        # uma decisão do desenho: quem ler o estudo depois precisa saber quantos
        # dias de fronteira saíram, e por qual feature.
        "purge_days": gap, "rows_purged": purgadas,
        "train_rows_per_season": {
            se: int(((trainval["_split"] == "train") & (trainval["season"] == se)).sum())
            for se in SEASONS
        },
        "base_features": base, "new_features": new, "static_features": static,
        "n_test_rows": len(test), "n_trainval_rows": len(trainval),
        **info,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[prepare] {len(arms)} arms, teste {len(test)} linhas, "
          f"treino por trimestre {meta['train_rows_per_season']}", flush=True)
    return meta
