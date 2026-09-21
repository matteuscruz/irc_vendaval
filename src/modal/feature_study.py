"""Modal: estudo de informação de features novas (cluster 3), em 3 estágios.

    prepare    1 container: carrega dados e grava o treino/teste em parquet.
    fit        fan-out: cada trimestre em SEU container (trimestre × lote de
               arms), LazyPredict inteiro (39 modelos) sobre o treino COMPLETO.
    aggregate  1 container: efeitos pareados, IC por blocos, ranking.

Sem amostragem: o cluster 3 tem só ~6 mil linhas de treino por trimestre (ver
`src/feature_study/config.py`). Com seeds fixas e dados completos não há réplicas.

Somente o cluster 3 e seeds fixas (`src/feature_study/config.py`): nada disso é
flag. Tudo vive no volume de artefatos em `feature_study/cluster3/`.

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
STUDY_DIR = "feature_study/cluster3"

# Fan-out: cada unidade paga o import de catboost/xgboost/lightgbm, então os
# arms vão em lotes. O teto evita estourar o limite de concorrência da conta.
CHUNK_SIZE = 5
MAX_CONTAINERS = 40
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

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
_VOLUMES = {REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume}


def _study() -> Path:
    return Path(REMOTE_ARTIFACTS_DIR) / STUDY_DIR


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
def prepare_remote(arm_sets: str = "core,groups,controls") -> dict:
    _bootstrap()
    from src.feature_study.prepare import prepare

    artifact_volume.reload()
    meta = prepare(
        f"{REMOTE_DATASET_DIR}/raw", f"{REMOTE_DATASET_DIR}/shp", _study(),
        arm_sets=tuple(s for s in arm_sets.split(",") if s),
    )
    _commit_safe()
    return {k: meta[k] for k in (
        "n_arms", "sampling", "n_test_rows", "n_trainval_rows", "train_rows_per_season",
        "rows_before_completeness", "rows_after_completeness",
        "clusters_dropped_by_coverage",
    )}


@app.function(image=image, volumes=_VOLUMES, timeout=600, memory=4096)
def list_arms() -> list[str]:
    _bootstrap()
    from src.feature_study.analysis import load_arms

    artifact_volume.reload()
    return [a.name for a in load_arms(_study() / "data")]


# ── Estágio 2 ────────────────────────────────────────────────────────────────

@app.function(
    image=image, volumes=_VOLUMES, timeout=7200, memory=8192, cpu=4,
    max_containers=MAX_CONTAINERS, retries=2,
)
def fit_unit(tag: str, season: str, arm_names: list[str], models: str = "all",
             out_tag: str | None = None, seed: int | None = None) -> dict:
    """Um trimestre, uma amostra, uma seed, um lote de arms — no próprio container."""
    _bootstrap()
    from src.feature_study.analysis import load_arms
    from src.feature_study.config import MODEL_SEED
    from src.feature_study.worker import run_unit
    seed = MODEL_SEED if seed is None else seed

    artifact_volume.reload()
    study = _study()
    by_name = {a.name: a for a in load_arms(study / "data")}
    unknown = [n for n in arm_names if n not in by_name]
    if unknown:
        raise ValueError(f"arms desconhecidos: {unknown}")

    t0 = time.time()
    print(f"[modal] ▶ {out_tag or tag}/{season}: {len(arm_names)} arm(s), modelos={models}, seed={seed}", flush=True)
    written = run_unit(study / "data", study, tag, season, [by_name[n] for n in arm_names],
                       models=models, out_tag=out_tag, seed=seed)
    _commit_safe()

    import pandas as pd
    m = pd.concat([pd.read_parquet(p) for p in written if p.name.startswith("metrics__")])
    per_arm = m.groupby("arm")[["fit_seconds", "peak_rss_mb"]].first()
    print(f"[modal] ✓ {out_tag or tag}/{season} em {time.time() - t0:.0f}s", flush=True)
    return {"tag": tag, "season": season, "arms": arm_names, "seconds": round(time.time() - t0, 1),
            "fit_seconds": per_arm["fit_seconds"].to_dict(),
            "peak_rss_mb": float(per_arm["peak_rss_mb"].max())}


# ── Estágio 3 ────────────────────────────────────────────────────────────────

@app.function(image=image, volumes=_VOLUMES, timeout=3600, memory=16384)
def aggregate_remote(tags: list[str], label: str = "main", n_boot: int = 2000) -> list[str]:
    _bootstrap()
    from src.feature_study.analysis import run_aggregate

    artifact_volume.reload()
    study = _study()
    run_aggregate(study, study / "data", tags, label=label, n_boot=n_boot)
    _commit_safe()
    dest = study / "summary" / label
    return sorted(str(p.relative_to(REMOTE_ARTIFACTS_DIR)) for p in dest.rglob("*") if p.is_file())


@app.function(image=image, volumes=_VOLUMES, timeout=600, memory=4096)
def compare_remote(tag_a: str, tag_b: str) -> float:
    _bootstrap()
    from src.feature_study.analysis import compare_units

    artifact_volume.reload()
    return compare_units(_study(), tag_a, tag_b)


# ── Orquestração local ───────────────────────────────────────────────────────

def _chunks(names: list[str], size: int) -> list[list[str]]:
    return [names[i:i + size] for i in range(0, len(names), size)]


def _run_fit(seeds: list[int], arm_names: list[str], models: str, chunk_size: int) -> None:
    """Uma réplica de treino por seed: mesma amostra (`full`), pasta de saída própria."""
    from src.feature_study.config import seed_tag

    seasons = ("DJF", "MAM", "JJA", "SON")
    args = [("full", s, chunk, models, seed_tag(seed), seed)
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


def _pilot(arm_names: list[str]) -> None:
    """Custo e determinismo — antes de gastar o fan-out.

    UMA unidade com os 39 modelos, duas vezes: tempo, memória e se o resultado é
    idêntico (seeds fixas). A primeira execução grava `units/full/` com os 39
    modelos, então o `fit` seguinte a reaproveita em vez de refazer.
    """
    seasons = ("DJF", "MAM", "JJA", "SON")
    print("\n[pilot] custo e determinismo (base, DJF, 39 modelos, 2×)...", flush=True)
    a, b = fit_unit.starmap([("full", "DJF", ["base"], "all", None),
                             ("full", "DJF", ["base"], "all", "full_repeat")])
    diff = compare_remote.remote("full", "full_repeat")
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
    arm_sets: str = "core,groups,controls",
    arms: str = "",
    models: str = "all",
    seeds: str = "",
    tags: str = "",
    label: str = "main",
    n_boot: int = 2000,
    chunk_size: int = CHUNK_SIZE,
    download: bool = True,
    local_dir: str = "artifacts",
) -> None:
    """
    --stage      prepare | pilot | fit | aggregate | all
    --arm-sets   core,groups,controls[,losses] (só no prepare; padrão: os três, 37 arms).
                 `losses` acrescenta o EIXO DE PERDA (10 arms: base/full × 4 expectilas +
                 Huber). É opt-in porque multiplica o fan-out.
    --arms       Restringe o fit a arms específicos (nomes separados por vírgula)
    --models     all (39, padrão) | reference (7) | fast3 (3)
    --seeds      Seeds do fit, separadas por vírgula (padrão: as 5 de MODEL_SEEDS; unidades prontas são puladas)
    --tags       Réplicas do aggregate (padrão: as de todas as seeds de MODEL_SEEDS)
    --label      Nome da pasta de resumo (padrão main)
    --n-boot     Sorteios do bootstrap em blocos (padrão 2000)
    --chunk-size Arms por container no fit (padrão 5)
    --no-download  Não baixa o resumo
    """
    if stage not in ("prepare", "pilot", "fit", "aggregate", "all"):
        raise SystemExit(f"--stage inválido: {stage!r}")
    # `main` roda na máquina local: o pacote `src` precisa estar no caminho.
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.feature_study.config import MAIN_TAGS, MODEL_SEEDS
    tag_list = [t for t in tags.split(",") if t] or list(MAIN_TAGS)
    seed_list = [int(s) for s in seeds.split(",") if s] or list(MODEL_SEEDS)

    if stage in ("prepare", "all"):
        print("prepare: carregando dados e amostrando...", flush=True)
        meta = prepare_remote.remote(arm_sets)
        print(f"prepare ok: {meta['n_arms']} arms | teste {meta['n_test_rows']} linhas | "
              f"amostragem: {'ligada' if meta['sampling'] else 'desligada (treino completo)'}")
        print(f"  treino por trimestre: {meta['train_rows_per_season']}")
        print(f"  linhas antes/depois da completude: {meta['rows_before_completeness']} → "
              f"{meta['rows_after_completeness']}")
        print(f"  clusters descartados pela cobertura: {meta['clusters_dropped_by_coverage']}")

    if stage in ("pilot", "fit", "all"):
        all_arms = list_arms.remote()
        chosen = [a for a in arms.split(",") if a] or all_arms
        unknown = [a for a in chosen if a not in all_arms]
        if unknown:
            raise SystemExit(f"arms desconhecidos: {unknown}. Disponíveis: {all_arms}")
        if stage == "pilot":
            _pilot(all_arms)
        else:
            _run_fit(seed_list, chosen, models, chunk_size)

    if stage in ("aggregate", "all"):
        print("\naggregate...", flush=True)
        created = aggregate_remote.remote(tag_list, label, n_boot)
        print(f"{len(created)} arquivo(s) de resumo.")
        if download:
            extra = [f"{STUDY_DIR}/data/{f}" for f in ("meta.json", "arms.json")]
            n = _download(created + extra, local_dir)
            print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}/{STUDY_DIR}")
