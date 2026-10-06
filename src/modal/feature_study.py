"""Modal: estudo de informação de features novas (cluster 3), em 3 estágios.

    prepare    1 container: carrega dados e grava o treino/teste em parquet.
    fit        fan-out: cada trimestre em SEU container (trimestre × lote de
               arms), LazyPredict inteiro (39 modelos) sobre o treino COMPLETO.
    aggregate  1 container: efeitos pareados, IC por blocos, ranking.

Sem amostragem: o cluster 3 tem só ~6 mil linhas de treino por trimestre. Com seeds
fixas e dados completos não há réplicas.

Somente o cluster 3 e seeds fixas (`src/feature_study/core/config.py`): nada disso é
flag. Tudo vive no volume de artefatos em `feature_study/<--study>/`.

    modal run src/modal/feature_study.py --stage prepare
    modal run src/modal/feature_study.py --stage pilot       # custo e determinismo
    modal run src/modal/feature_study.py --stage fit         # ~32 containers
    modal run src/modal/feature_study.py --stage aggregate
    modal run src/modal/feature_study.py --stage all         # prepare → fit → aggregate

O `fit` é idempotente: se algum container falhar, repita o mesmo comando — as
unidades já gravadas são puladas.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

APP_NAME = "irc-vendaval-feature-study"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

# Relativo à RAIZ do volume (o `lazy_modal/` local é só o espelho do download).
# Namespace do estudo dentro do volume. O modo RAW horário tem OUTRA população
# (a base deixa de ser diária, e a purga corta linhas), então seus `row_id` não
# correspondem aos do estudo diário: gravar por cima misturaria dois estudos e
# `load_residuals` leria resíduos desalinhados sem nenhum erro. Cada desenho no
# seu diretório.
STUDY_ROOT = "feature_study"
DEFAULT_STUDY = "cluster3"

# Fan-out: cada unidade paga o import de catboost/xgboost/lightgbm, então os
# arms vão em lotes. O teto evita estourar o limite de concorrência da conta.
CHUNK_SIZE = 5
MAX_CONTAINERS = 40
# Nome do JSON congelado pela triagem. Vive em `summary/` e é lido por cada
# container do `fit` quando `--models top5`.
TOP_MODELS_FILE = "top_models.json"
THREADS = "4"

_local_root = Path(__file__).parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libhdf5-dev", "libnetcdf-dev", "gdal-bin", "libgdal-dev", "libproj-dev")
    .pip_install(
        "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.6.0", "matplotlib>=3.9.0",
        "xarray>=2024.1.0", "netCDF4>=1.6.0", "geopandas>=0.14.0", "pyshp>=2.3.0",
        "scipy>=1.10.0", "shapely>=2.0.0",
        # As imagens das outras pipelines NÃO têm pyarrow (saiu junto com o BT55);
        # aqui ele é o formato de troca entre os estágios.
        "pyarrow>=14.0.0",
    )
    .pip_install("lazypredict[boost]>=0.3.0", "xgboost>=2.0.0", "lightgbm>=4.0.0")
    .env({
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        # CatBoost/XGBoost ignoram o cgroup e abririam threads por todos os
        # núcleos do host. Fixar mantém o custo previsível e a execução estável.
        "OMP_NUM_THREADS": THREADS,
        "MKL_NUM_THREADS": THREADS,
        "OPENBLAS_NUM_THREADS": THREADS,
    })
    .add_local_file(str(_local_root / "main.py"), remote_path=f"{REMOTE_APP_DIR}/main.py", copy=True)
    .add_local_dir(str(_local_root / "src"), remote_path=f"{REMOTE_APP_DIR}/src", copy=True)
)

# Imagem da LSTM: TensorFlow com GPU. A do estudo tabular não tem TensorFlow, e a da LSTM de
# produção não tem pyarrow nem lazypredict — por isso uma terceira, só para este estágio.
lstm_image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    .apt_install("libhdf5-dev", "libnetcdf-dev", "gdal-bin", "libgdal-dev", "libproj-dev")
    .pip_install(
        "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.6.0", "matplotlib>=3.9.0",
        "xarray>=2024.1.0", "netCDF4>=1.6.0", "geopandas>=0.14.0", "pyshp>=2.3.0",
        "scipy>=1.10.0", "shapely>=2.0.0", "pyarrow>=14.0.0", "pyyaml>=6.0.3", "pydantic>=2.13.3",
    )
    .pip_install("lazypredict[boost]>=0.3.0", "xgboost>=2.0.0", "lightgbm>=4.0.0")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_file(str(_local_root / "main.py"), remote_path=f"{REMOTE_APP_DIR}/main.py", copy=True)
    .add_local_dir(str(_local_root / "src"), remote_path=f"{REMOTE_APP_DIR}/src", copy=True)
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
_VOLUMES = {REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume}


def _study(study: str = DEFAULT_STUDY) -> Path:
    return Path(REMOTE_ARTIFACTS_DIR) / STUDY_ROOT / study


def _commit_safe() -> None:
    """Commit com 1 retry após reload (escrita concorrente de várias unidades)."""
    try:
        artifact_volume.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[modal] commit falhou ({exc}); reload + retry...")
        artifact_volume.reload()
        artifact_volume.commit()


def _bootstrap() -> None:
    if REMOTE_APP_DIR not in sys.path:
        sys.path.insert(0, REMOTE_APP_DIR)


# ── Estágio 1 ────────────────────────────────────────────────────────────────

@app.function(
    image=image, volumes=_VOLUMES, timeout=21600, memory=32768,
    nonpreemptible=True,  # o merge do ERA5 não tem checkpoint aqui
)
def prepare_remote(arm_sets: str = "anchors,groups,controls", study: str = DEFAULT_STUDY) -> dict:
    _bootstrap()
    from src.feature_study.core.prepare import prepare

    artifact_volume.reload()
    meta = prepare(
        f"{REMOTE_DATASET_DIR}/raw", _study(study),
        arm_sets=tuple(s for s in arm_sets.split(",") if s),
    )
    _commit_safe()
    return {k: meta[k] for k in (
        "n_arms", "n_test_rows", "n_trainval_rows", "train_rows_per_season",
        "rows_before_completeness", "rows_after_completeness",
        "clusters_dropped_by_coverage", "purge_days", "rows_purged",
    )}


@app.function(image=image, volumes=_VOLUMES, timeout=600, memory=4096)
def list_arms(study: str = DEFAULT_STUDY) -> list[str]:
    _bootstrap()
    from src.feature_study.core.analysis import load_arms

    artifact_volume.reload()
    return [a.name for a in load_arms(_study(study) / "data")]


@app.function(image=image, volumes=_VOLUMES, timeout=600, memory=4096)
def add_arms_remote(selection_json: str, study: str = DEFAULT_STUDY) -> list[str]:
    """Acrescenta os arms de seleção a `arms.json` SEM refazer o `prepare`.

    O `prepare` regeneraria dados e `row_id`; aqui só entram arms novos, e os
    resultados já ajustados (`base`, `full`...) continuam pareando com eles."""
    _bootstrap()
    import json

    from src.feature_study.core.analysis import load_arms
    from src.feature_study.selection.selected import append_arms, build_selected_arms

    artifact_volume.reload()
    data = _study(study) / "data"
    full = next(a for a in load_arms(data) if a.name == "full")
    nomes = append_arms(data, build_selected_arms(full.features, json.loads(selection_json)))
    _commit_safe()
    return nomes


# ── Estágio 2 ────────────────────────────────────────────────────────────────

@app.function(
    image=image, volumes=_VOLUMES, timeout=7200, memory=8192, cpu=4,
    max_containers=MAX_CONTAINERS, retries=2,
)
def fit_unit(tag: str, season: str, arm_names: list[str], models: str = "all",
             out_tag: str | None = None, seed: int | None = None,
             study: str = DEFAULT_STUDY) -> dict:
    """Um trimestre, uma amostra, uma seed, um lote de arms — no próprio container."""
    _bootstrap()
    from src.feature_study.core.analysis import load_arms
    from src.feature_study.core.config import MODEL_SEED
    from src.feature_study.core.worker import run_unit
    seed = MODEL_SEED if seed is None else seed

    artifact_volume.reload()
    study = _study(study)
    by_name = {a.name: a for a in load_arms(study / "data")}
    unknown = [n for n in arm_names if n not in by_name]
    if unknown:
        raise ValueError(f"arms desconhecidos: {unknown}")

    # `top5` é resolvido AQUI, dentro do container, lendo o JSON congelado pela
    # triagem — mesmo padrão dos arms logo acima. O driver não precisa do
    # volume montado, e a unidade já sabe seu próprio trimestre.
    escolhidos, campeao = models, ""
    if models == "top5":
        escolhidos = _top5_for(study, season)
        # rank-1 do trimestre: é o único cujo estimador ajustado é persistido,
        # para destravar o mapa em GRADE sem retreinar nada.
        campeao = escolhidos[0]

    t0 = time.time()
    print(f"[modal] ▶ {out_tag or tag}/{season}: {len(arm_names)} arm(s), "
          f"modelos={escolhidos if models == 'top5' else models}, seed={seed}", flush=True)
    written = run_unit(study / "data", study, tag, season, [by_name[n] for n in arm_names],
                       models=escolhidos, out_tag=out_tag, seed=seed, champion=campeao)
    _commit_safe()

    import pandas as pd
    m = pd.concat([pd.read_parquet(p) for p in written if p.name.startswith("metrics__")])
    per_arm = m.groupby("arm")[["fit_seconds", "peak_rss_mb"]].first()
    print(f"[modal] ✓ {out_tag or tag}/{season} em {time.time() - t0:.0f}s", flush=True)
    return {"tag": tag, "season": season, "arms": arm_names, "seconds": round(time.time() - t0, 1),
            "fit_seconds": per_arm["fit_seconds"].to_dict(),
            "peak_rss_mb": float(per_arm["peak_rss_mb"].max())}


@app.function(
    image=lstm_image, volumes=_VOLUMES, gpu="T4", timeout=7200, memory=12288, cpu=4,
    max_containers=MAX_CONTAINERS, retries=1,
)
def fit_lstm_unit(season: str, arm_names: list[str], seed: int, window_hours: int = 24,
                  study: str = DEFAULT_STUDY) -> dict:
    """Um trimestre, uma seed, todos os arms pedidos, com a LSTM — no próprio container.

    Mesma população e mesmos arms do `fit_unit`; a saída vai para `units/<tag>_lstm/`.
    """
    _bootstrap()
    from src.feature_study.core.analysis import load_arms
    from src.feature_study.core.config import seed_tag
    from src.feature_study.core.worker_lstm import run_unit_lstm

    artifact_volume.reload()
    study_dir = _study(study)
    by_name = {a.name: a for a in load_arms(study_dir / "data")}
    unknown = [n for n in arm_names if n not in by_name]
    if unknown:
        raise ValueError(f"arms desconhecidos: {unknown}")
    out_tag = seed_tag(seed) + "_lstm"

    t0 = time.time()
    print(f"[modal] ▶ {out_tag}/{season}: {len(arm_names)} arm(s), janela {window_hours} h, seed={seed}", flush=True)
    written = run_unit_lstm(study_dir / "data", study_dir, f"{REMOTE_DATASET_DIR}/raw", "full", season,
                            [by_name[n] for n in arm_names], out_tag=out_tag, seed=seed,
                            window_hours=window_hours)
    _commit_safe()
    print(f"[modal] ✓ {out_tag}/{season} em {time.time() - t0:.0f}s ({len(written)} arquivos)", flush=True)
    return {"tag": out_tag, "season": season, "arms": arm_names, "seconds": round(time.time() - t0, 1)}


def _top5_for(study, season: str) -> list[str]:
    """Lista de modelos do trimestre, lida do JSON congelado pela triagem.

    Falha ALTO se o arquivo não existir ou não tiver o trimestre: rodar o
    `fit` com `top5` antes do `screen` produziria silenciosamente um estudo com
    outro conjunto de modelos em cada trimestre.
    """
    import json

    caminho = study / "summary" / TOP_MODELS_FILE
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe — rode `--stage screen` antes de `--stage fit --models top5`"
        )
    payload = json.loads(caminho.read_text())
    modelos = payload.get("by_season", {}).get(season)
    if not modelos:
        raise ValueError(f"triagem sem modelos para {season} em {caminho}")
    return list(modelos)


# ── Estágio 2b: triagem ──────────────────────────────────────────────────────

@app.function(image=image, volumes=_VOLUMES, timeout=1800, memory=8192)
def screen_remote(tags: list[str], arm: str = "base", k: int = 5,
                  study: str = DEFAULT_STUDY) -> dict:
    """Congela os `k` melhores modelos por trimestre a partir do arm `base`.

    Roda UMA vez e grava o resultado; o `fit` seguinte apenas lê. Recalcular a
    cada unidade faria uma seed a mais no volume mudar o conjunto de modelos no
    meio do fan-out.
    """
    _bootstrap()
    import json

    from src.feature_study.core.analysis import top_models_by_season, top_models_payload

    artifact_volume.reload()
    study = _study(study)
    top = top_models_by_season(study, tags, arm=arm, k=k)
    if top.empty:
        raise ValueError(f"nenhuma métrica do arm {arm!r} em units/ — rode o `fit` do arm base antes")

    destino = study / "summary"
    destino.mkdir(parents=True, exist_ok=True)
    top.to_parquet(destino / "top_models.parquet", index=False)
    payload = top_models_payload(top, arm=arm, k=k, tags=tags)
    (destino / TOP_MODELS_FILE).write_text(json.dumps(payload, indent=2))
    _commit_safe()

    for season, modelos in payload["by_season"].items():
        print(f"[screen] {season}: {', '.join(modelos)}", flush=True)
    instavel = top[top["sd_RMSE_seeds"] > 0.05]
    if not instavel.empty:
        print(f"[screen] AVISO: {len(instavel)} escolha(s) com sd do RMSE entre seeds > 0.05 — "
              "triagem instável, considere aumentar k", flush=True)
    return payload


# ── Estágio 3 ────────────────────────────────────────────────────────────────

@app.function(image=image, volumes=_VOLUMES, timeout=3600, memory=16384)
def aggregate_remote(tags: list[str], label: str = "main", n_boot: int = 2000,
                     study: str = DEFAULT_STUDY) -> list[str]:
    _bootstrap()
    from src.feature_study.core.analysis import run_aggregate

    artifact_volume.reload()
    study = _study(study)
    run_aggregate(study, study / "data", tags, label=label, n_boot=n_boot)
    _commit_safe()
    dest = study / "summary" / label
    return sorted(str(p.relative_to(REMOTE_ARTIFACTS_DIR)) for p in dest.rglob("*") if p.is_file())


@app.function(image=image, volumes=_VOLUMES, timeout=600, memory=4096)
def compare_remote(tag_a: str, tag_b: str, study: str = DEFAULT_STUDY) -> float:
    _bootstrap()
    from src.feature_study.core.analysis import compare_units

    artifact_volume.reload()
    return compare_units(_study(study), tag_a, tag_b)


# ── Orquestração local ───────────────────────────────────────────────────────

def _chunks(names: list[str], size: int) -> list[list[str]]:
    return [names[i:i + size] for i in range(0, len(names), size)]


def _run_fit(seeds: list[int], arm_names: list[str], models: str, chunk_size: int,
             study: str = DEFAULT_STUDY, triage: bool = False) -> None:
    """Uma réplica de treino por seed: mesma amostra (`full`), pasta de saída própria.

    Com `triage=True` a saída vai para `units/<tag>_triage/`, separada do estudo:
    os dois gravam o mesmo nome de arquivo para o arm `base`, e sem a separação o
    estudo (5 modelos) sobrescrevia a triagem (39) — como aconteceu no
    `cluster3_raw`."""
    from src.feature_study.core.config import seed_tag, triage_tag
    tag_de = triage_tag if triage else seed_tag

    seasons = ("DJF", "MAM", "JJA", "SON")
    args = [("full", s, chunk, models, tag_de(seed), seed, study)
            for seed in seeds for s in seasons for chunk in _chunks(arm_names, chunk_size)]
    print(f"\nfit: {len(args)} container(s) = {len(seeds)} seed(s) {seeds} × {len(seasons)} trimestres × "
          f"{len(_chunks(arm_names, chunk_size))} lote(s) de até {chunk_size} arms "
          f"(máx. {MAX_CONTAINERS} simultâneos)", flush=True)
    results = list(fit_unit.starmap(args, order_outputs=False, return_exceptions=True))
    failed = [r for r in results if isinstance(r, BaseException)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    secs = [r["seconds"] for r in ok]
    print(f"fit: {len(ok)}/{len(args)} ok"
          + (f" | container: mediana {sorted(secs)[len(secs) // 2]:.0f}s, "
             f"total {sum(secs) / 3600:.1f} container-h" if secs else ""), flush=True)
    if failed:
        print(f"fit: {len(failed)} FALHARAM (repita o comando; unidades prontas são puladas):")
        for f in failed[:5]:
            print(f"   {type(f).__name__}: {str(f)[:300]}")
        raise SystemExit(1)


def _run_fit_lstm(seeds: list[int], arm_names: list[str], window_hours: int, study: str) -> None:
    """Uma unidade por (seed × trimestre), com todos os arms no mesmo container: as janelas são
    lidas uma vez por unidade e reaproveitadas por todos os arms."""
    seasons = ("DJF", "MAM", "JJA", "SON")
    args = [(s, arm_names, seed, window_hours, study) for seed in seeds for s in seasons]
    print(f"\nfit-lstm: {len(args)} container(s) = {len(seeds)} seed(s) {seeds} × {len(seasons)} trimestres, "
          f"{len(arm_names)} arm(s) cada, janela de {window_hours} h", flush=True)
    results = list(fit_lstm_unit.starmap(args, order_outputs=False, return_exceptions=True))
    failed = [r for r in results if isinstance(r, BaseException)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    secs = [r["seconds"] for r in ok]
    print(f"fit-lstm: {len(ok)}/{len(args)} ok"
          + (f" | container: mediana {sorted(secs)[len(secs) // 2]:.0f}s, "
             f"total {sum(secs) / 3600:.2f} container-h" if secs else ""), flush=True)
    if failed:
        print(f"fit-lstm: {len(failed)} FALHARAM (repita o comando; unidades prontas são puladas):")
        for f in failed[:5]:
            print(f"   {type(f).__name__}: {str(f)[:300]}")
        raise SystemExit(1)


def _download(paths: list[str], local_dir: str) -> int:
    vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
    root = Path(local_dir)
    n = 0
    for rel in paths:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"".join(vol.read_file(rel)))
        print(f"  ↓ {rel}")
        n += 1
    return n


def _pilot(arm_names: list[str], study: str = DEFAULT_STUDY) -> None:
    """Custo e determinismo — antes de gastar o fan-out.

    UMA unidade com os 39 modelos, duas vezes: tempo, memória e se o resultado é
    idêntico (seeds fixas). A primeira execução grava `units/full/` com os 39
    modelos, então o `fit` seguinte a reaproveita em vez de refazer.
    """
    seasons = ("DJF", "MAM", "JJA", "SON")
    print("\n[pilot] custo e determinismo (base, DJF, 39 modelos, 2×)...", flush=True)
    # As DUAS tuplas levam `study` explícito: uma tupla curta cairia no default
    # e o `full_repeat` iria gravar dentro de OUTRO estudo, silenciosamente.
    a, b = fit_unit.starmap([("full", "DJF", ["base"], "all", None, None, study),
                             ("full", "DJF", ["base"], "all", "full_repeat", None, study)])
    diff = compare_remote.remote("full", "full_repeat", study)
    t = a["fit_seconds"]["base"]
    print(f"  tempo de 1 arm × 1 trimestre com 39 modelos: {t:.0f}s | pico de memória {a['peak_rss_mb']:.0f} MB")
    print(f"  determinismo: maior diferença entre as duas execuções = {diff:.3g}"
          + ("  ✓ idêntico" if diff == 0 else "  ✗ NÃO reproduz — investigar antes do fan-out"))
    est_h = t * len(arm_names) * len(seasons) / 3600
    print(f"  extrapolação do run completo: ≈ {est_h:.1f} container-horas "
          f"({len(arm_names)} arms × 4 trimestres) — estimativa, não medida")


@app.local_entrypoint()
def main(
    stage: str,
    arm_sets: str = "anchors,groups,controls",
    arms: str = "",
    models: str = "all",
    seeds: str = "",
    tags: str = "",
    label: str = "main",
    study: str = DEFAULT_STUDY,
    n_boot: int = 2000,
    chunk_size: int = CHUNK_SIZE,
    download: bool = True,
    local_dir: str = "artifacts",
    triage: bool = False,
    selection: str = "config/selected_features_val12.json",
    lstm: bool = False,
    window_hours: int = 24,
) -> None:
    """
    --stage      prepare | add-arms | pilot | fit | fit-lstm | screen | aggregate | all
    --lstm       No `aggregate`: agrega as unidades da LSTM (`<tag>_lstm`) em `summary/lstm`.
    --window-hours  No `fit-lstm`: janela da LSTM, em horas até a hora do pico (padrão 24).
                 `fit-lstm` usa a MESMA população e os MESMOS arms do estudo tabular, com a
                 LSTM no lugar dos modelos; fora do `all`. Rode `--arms` para restringir.
    --selection  JSON congelado da seleção de variáveis (só no add-arms). `add-arms`
                 acrescenta os arms `sel__*` a arms.json SEM refazer o prepare; grátis.
    --arm-sets   anchors,groups,controls (padrão) — só no prepare.
                 `anchors` = base + full, exigido por todos os outros.
                 `singles`  = a varredura base+f / full−f, uma por feature. OPT-IN:
                              é ela que produz ~100 arms. Rode-a depois, restrita ao
                              grupo que venceu. `core` é o alias de anchors+singles.
    --arms       Restringe o fit a arms específicos (nomes separados por vírgula)
    --models     all (39, padrão) | reference (7) | fast3 (3) | top5 (a triagem)
    --seeds      Seeds do fit, separadas por vírgula (padrão: as 5 de MODEL_SEEDS; unidades prontas são puladas)
    --tags       Réplicas do aggregate (padrão: as de todas as seeds de MODEL_SEEDS)
    --label      Nome da pasta de resumo (padrão main)
    --study      Namespace no volume (padrão cluster3). O modo RAW horário tem OUTRA
                 população e outros row_id — use um nome próprio (ex.: cluster3_raw),
                 senão os dois estudos se misturam sem nenhum erro.
    --n-boot     Sorteios do bootstrap em blocos (padrão 2000)
    --chunk-size Arms por container no fit (padrão 5)
    --triage     No `fit`, grava em units/<tag>_triage/ (triagem dos 39 modelos no arm base).
                 No `screen`, é o padrão. Mantém a triagem auditável: sem isto o estudo
                 sobrescreve a leaderboard de 39 modelos do arm base.
    --no-download  Não baixa o resumo
    """
    if stage not in ("prepare", "add-arms", "pilot", "fit", "fit-lstm", "screen", "aggregate", "all"):
        raise SystemExit(f"--stage inválido: {stage!r}")
    # `main` roda na máquina local: o pacote `src` precisa estar no caminho.
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.feature_study.core.config import MAIN_TAGS, MODEL_SEEDS, seed_tag, triage_tag
    seed_list = [int(s) for s in seeds.split(",") if s] or list(MODEL_SEEDS)
    tag_list = [t for t in tags.split(",") if t] or (
        [seed_tag(s) + "_lstm" for s in seed_list] if lstm else list(MAIN_TAGS))
    if lstm and label == "main":
        label = "lstm"
    # `screen` lê a pasta da TRIAGEM (<tag>_triage) por padrão. Para uma triagem
    # antiga, gravada junto do estudo, passe `--tags` explicitamente.
    screen_tags = [t for t in tags.split(",") if t] or [triage_tag(s) for s in seed_list]

    if stage in ("prepare", "all"):
        print("prepare: carregando dados e amostrando...", flush=True)
        meta = prepare_remote.remote(arm_sets, study)
        print(f"prepare ok: {meta['n_arms']} arms | teste {meta['n_test_rows']} linhas | treino completo")
        print(f"  treino por trimestre: {meta['train_rows_per_season']}")
        print(f"  linhas antes/depois da completude: {meta['rows_before_completeness']} → "
              f"{meta['rows_after_completeness']}")
        print(f"  clusters descartados pela cobertura: {meta['clusters_dropped_by_coverage']}")
        print(f"  purga temporal: gap de {meta['purge_days']} dia(s), "
              f"{meta['rows_purged']} linha(s) removida(s) das fronteiras de mês")

    if stage == "add-arms":
        nomes = add_arms_remote.remote((Path(root) / selection).read_text(), study)
        print(f"add-arms ok: {len(nomes)} arms em {STUDY_ROOT}/{study}/data/arms.json: {', '.join(nomes)}")

    if stage == "fit-lstm":
        all_arms = list_arms.remote(study)
        chosen = [a for a in arms.split(",") if a] or all_arms
        unknown = [a for a in chosen if a not in all_arms]
        if unknown:
            raise SystemExit(f"arms desconhecidos: {unknown}. Disponíveis: {all_arms}")
        _run_fit_lstm(seed_list, chosen, window_hours, study)

    if stage in ("pilot", "fit", "all"):
        all_arms = list_arms.remote(study)
        chosen = [a for a in arms.split(",") if a] or all_arms
        unknown = [a for a in chosen if a not in all_arms]
        if unknown:
            raise SystemExit(f"arms desconhecidos: {unknown}. Disponíveis: {all_arms}")
        if stage == "pilot":
            _pilot(all_arms, study)
        else:
            _run_fit(seed_list, chosen, models, chunk_size, study, triage)

    # `screen` fica FORA de `all` de propósito: ele só faz sentido entre um
    # `fit --arms base --models all` e um `fit --models top5`. Dentro de `all`
    # rodaria depois de todos os arms já ajustados, escolhendo modelos que
    # ninguém mais usaria.
    if stage == "screen":
        print("\nscreen: escolhendo os melhores modelos por trimestre...", flush=True)
        payload = screen_remote.remote(screen_tags, study=study)
        for season, modelos in payload["by_season"].items():
            print(f"  {season}: {', '.join(modelos)}")
        print(f"  congelado em {STUDY_ROOT}/{study}/summary/{TOP_MODELS_FILE} "
              f"(impressão digital {payload['models_fp']})")

    if stage in ("aggregate", "all"):
        print("\naggregate...", flush=True)
        created = aggregate_remote.remote(tag_list, label, n_boot, study)
        print(f"{len(created)} arquivo(s) de resumo.")
        if download:
            extra = [f"{STUDY_ROOT}/{study}/data/{f}" for f in ("meta.json", "arms.json")]
            if lstm:
                # as unidades da LSTM são pequenas e trazem a importância por grupo/coluna
                vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
                for t in tag_list:
                    extra += [e.path for e in vol.listdir(f"{STUDY_ROOT}/{study}/units/{t}")
                              if e.path.endswith(".parquet")]
            n = _download(created + extra, local_dir)
            print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}/{STUDY_ROOT}/{study}")
