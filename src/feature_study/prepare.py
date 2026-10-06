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

import numpy as np
import pandas as pd

from src.feature_study.arms import (
    DEFAULT_ARM_SETS, add_control_columns, build_arms, expand_arm_sets,
    usable_new_features,
)
from src.feature_study.config import (
    ALPHA, CLUSTER_ID, EPS, EXCLUDED_DUPLICATES, LOSS_MODELS, MODEL_SEED, MODEL_SEEDS,
    PILOT_SIZES, SAMPLE_SEEDS, SEASONS,
)
from src.feature_study.hourly_flat import (
    HOURLY_FLAT_NEW_PREFIX, hourly_flat_columns, merge_hourly_flat, purge_days,
)
from src.feature_study import groups_source as gs
from src.feature_study.hourly_source import merge_hourly_features
from src.feature_study.sampling import (
    assert_no_gross_failure, build_replicate, dkw_sample_size, group_sizes,
    representativeness_report, structural_violations,
)
from src.pipeline.data.splits import LABEL_DTYPE, purge_keep
from src.pipelines.common import (
    ERA5_GUST_PROXY, KNOWN_STATIC_FEATURES, TARGET_VAR, month_to_season,
    resolve_feature_groups, select_complete_rows,
)

TRAINVAL = ("train", "val")

# Identificam a estação e entram na base do modo RAW. Sem elas o modelo perde o
# identificador que as 40 features diárias carregavam, e o mapa espacial fica
# sem coordenada para plotar.
RAW_CONTEXT_COLUMNS = ("latitude", "longitude")

# Colunas de REFERÊNCIA, preservadas no parquet mas que nunca entram em arm
# nenhum — não são features, são o que se compara contra.
#
# A referência é a rajada ERA5 da estrutura nova (`era5_gust_max`) ou, nos modos
# antigos, `wind_mag_max`. ATENÇÃO à diferença: `wind_mag_max` é o máximo do
# VENTO MÉDIO a 10 m, não uma rajada — tem viés de −5 m/s contra o INMET e faz o
# modelo parecer corrigir muito mais do que corrige. Medido no teste, a rajada
# ERA5 bruta tem viés de −0,03 m/s (RMSE 2,65 contra 5,85). Toda comparação
# "modelo vs ERA5" deve usar a rajada.
#
# Perder a coluna é silencioso: nada falha, só o `era5_proxy` das predições sai
# todo NaN e o mapa de correlação fica vazio.
REFERENCE_CANDIDATES = (gs.REFERENCE_COLUMN, ERA5_GUST_PROXY)


def _raw_hourly_pool(df0: pd.DataFrame, raw_dir: str, cluster_id: int):
    """Modo `hourly_raw`: a BASE deixa de ser diária.

    Nada aqui é agregado. A base vira o perfil horário cru do ERA5 padrão
    (`hf_*`, de `test_cluster_3_hourly.nc`) e as features novas vêm igualmente
    horárias (`hfn_*`, extraídas da grade em `new_features/`). As 7 estáticas
    já estão em `df0` e entram como estão: são geografia, sem dimensão
    temporal, e não há nada de "diário" nelas para trocar.

    Devolve `(df0 com as colunas achatadas, base, novas, estáticas)`.
    """
    from src.data.new_features import merge_new_features_hourly_flat

    df0 = merge_hourly_flat(df0, raw_dir)
    # Só as estações do cluster estudado: o frame diário traz todas (237 na
    # execução real) e a extração horária custaria 26× mais para um dado que a
    # regra de cobertura descarta logo em seguida.
    do_cluster = df0.loc[df0["cluster_id"].astype(str) == str(cluster_id), "estacao"].unique()
    df0 = merge_new_features_hourly_flat(df0, raw_dir, only_stations=do_cluster)

    base = hourly_flat_columns() + list(RAW_CONTEXT_COLUMNS)
    novas_horarias = sorted(c for c in df0.columns if str(c).startswith(HOURLY_FLAT_NEW_PREFIX))
    _, static = usable_new_features(df0.columns, KNOWN_STATIC_FEATURES)
    faltando = [c for c in base if c not in df0.columns]
    if faltando:
        raise ValueError(f"colunas da base horária ausentes do frame: {faltando[:5]}… ({len(faltando)})")
    return df0, base, novas_horarias + static, static


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


def _population(raw_dir: str, shp_dir: str, cluster_id: int, arm_sets=DEFAULT_ARM_SETS,
                cache_dir=None):
    """Frame do cluster sob a UNIÃO de todas as features (mesmas linhas para
    todos os arms) + contagens do que a exigência de completude descartou.

    Carrega com o grupo `original` (as colunas `nf_*` vêm no frame de qualquer
    modo) e aplica a completude da união aqui, em vez de usar
    `feature_groups="original,new_features"` — assim dá para contar o que ela
    remove, o que o loader esconde.

    Com `"hourly" in arm_sets`, SOMA a segunda fonte de features novas —
    `hourly_source.py`, o arquivo horário por estação
    (`test_cluster_3_hourly.nc`), que não passa pelo `new_features/` porque
    não tem grade lat/lon para o loader validar — às features diárias da
    grade. Com `"hourly_only"`, TROCA: só as `nf_h_*` entram no pool de
    features novas ("o horário sozinho ajuda?", não "o horário ajuda além do
    que já temos?"). Sem nenhum dos dois (padrão), o arquivo horário nem é
    lido — a população fica exatamente como se essa fonte não existisse.
    """
    arm_sets = expand_arm_sets(arm_sets)   # valida e recusa modos que trocam a base

    extra_info: dict = {}
    if "era5_groups" in arm_sets:
        # não passa pelo `load_lazy_training_frame`: aquele exige o ERA5 bruto antigo
        df0, base, new, static, _, extra_info = _population_era5_groups(raw_dir, cluster_id, cache_dir)
    else:
        from src.pipelines.cluster_lazy import load_lazy_training_frame
        df0 = load_lazy_training_frame(raw_dir, shp_dir, "original")
    if "era5_groups" in arm_sets:
        pass
    elif "hourly_raw" in arm_sets:
        df0, base, new, static = _raw_hourly_pool(df0, raw_dir, cluster_id)
    else:
        if "hourly" in arm_sets or "hourly_only" in arm_sets:
            df0 = merge_hourly_features(df0, raw_dir)
        base = resolve_feature_groups("original")
        new, static = usable_new_features(df0.columns, KNOWN_STATIC_FEATURES)
        if "hourly_only" in arm_sets:
            new = [f for f in new if f.startswith("nf_h_")]
            static = [f for f in static if f.startswith("nf_h_")]  # nenhuma feature horária é estática
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
        **extra_info,
    }
    return pop, base, new, static, info


def prepare(
    raw_dir: str, shp_dir: str, out_dir, *,
    eps: float = EPS, alpha: float = ALPHA, seeds=SAMPLE_SEEDS,
    pilot_sizes=PILOT_SIZES, arm_sets=DEFAULT_ARM_SETS, cluster_id: int = CLUSTER_ID,
) -> dict:
    out = Path(out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)

    pop, base, new, static, info = _population(
        raw_dir, shp_dir, cluster_id, arm_sets, cache_dir=Path(out_dir) / "_cache",
    )
    group_of = info.get("group_of")
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
    # lat/lon entram explicitamente: se algum dia saírem da lista de features
    # (o mlp v4 já fez isso no caminho dele), o mapa espacial perderia a
    # coordenada sem que nada aqui acusasse.
    referencia = next((c for c in REFERENCE_CANDIDATES if c in pop.columns), None)
    if referencia is None and not any(c in feature_cols for c in REFERENCE_CANDIDATES):
        raise ValueError(
            f"nenhuma coluna de referência ERA5 na população (esperada: {REFERENCE_CANDIDATES}) — "
            "sem ela não há mapa espacial nem comparação contra o ERA5, e a perda seria silenciosa"
        )
    contexto = [c for c in (*RAW_CONTEXT_COLUMNS, referencia)
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
        # Gap da purga temporal e quanto ele custou. Vai para o meta porque é
        # uma decisão do desenho, não um detalhe: quem ler o estudo depois
        # precisa saber quantos dias de fronteira saíram, e por qual feature.
        "purge_days": gap, "rows_purged": purgadas,
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
