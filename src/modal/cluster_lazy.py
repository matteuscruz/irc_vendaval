"""
IRC Vendaval — Modal deployment (LazyPredict por cluster, fan-out)
==================================================================
Executa a pipeline LazyRegressor × cluster espacial na nuvem (Modal),
**um container por cluster em paralelo**, e baixa os artefatos.

Arquitetura resiliente:
  1. discover_clusters  → resolve exp + lista clusters (1 container leve)
  2. process_cluster    → 1 container por cluster (.starmap, paralelo);
                          cada um escreve _partial/cluster_N.csv + plots e
                          faz commit no volume ao terminar
  3. aggregate_clusters → junta os parciais em CSV final + heatmap + meta

Se um cluster estourar tempo/memória, os demais não são afetados e seus
artefatos já estão commitados (baixe com --only-download).

Usa os mesmos dados NetCDF (raw/) e shapefiles (shp/), compartilhando o
volume "irc-vendaval-dataset".

Uso
---
# Rodar (recomenda-se --detach p/ sobreviver a quedas de conexão local)
modal run --detach src/modal/cluster_lazy.py

# Só baixar artefatos de uma run já concluída
modal run src/modal/cluster_lazy.py --only-download \\
    --local-dir artifacts/lazy_modal

# Forçar re-upload dos dados
modal run src/modal/cluster_lazy.py --force-dataset-upload
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# ── Constantes ───────────────────────────────────────────────────────────────

APP_NAME = "irc-vendaval-lazy-clusters"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

# Mesmo sentinela da pipeline LSTM — reutiliza upload já existente
DATASET_SENTINELS = [
    "raw/INMET_Stratified.nc",
    "raw/ERA5_Stratified.nc",
    "raw/ERA5_Features_Basin_2000_2026.nc",
]

_local_root = Path(__file__).parent.parent.parent

# ── Imagem ───────────────────────────────────────────────────────────────────
# Pipeline CPU-only: sem TensorFlow. Imagem leve com lazypredict + geo.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # HDF5 + NetCDF (necessários para netCDF4/xarray)
        "libhdf5-dev",
        "libnetcdf-dev",
        # GDAL (necessário para geopandas/fiona)
        "gdal-bin",
        "libgdal-dev",
        "libproj-dev",
    )
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "xarray>=2024.1.0",
        "netCDF4>=1.6.0",
        "geopandas>=0.14.0",
        "pyshp>=2.3.0",
        "scipy>=1.10.0",
        "cartopy>=0.22.0",
        "shapely>=2.0.0",
    )
    .pip_install(
        "lazypredict[boost]>=0.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
    )
    .env({
        "MPLBACKEND": "Agg",
        # Sem isso, stdout fica block-buffered quando não é um TTY (caso do
        # Modal) — prints de progresso/tempo ficam presos no buffer até ele
        # encher ou o processo terminar, escondendo em que ponto exato uma
        # etapa longa (~horas) está travada/lenta enquanto ela roda.
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_file(
        str(_local_root / "main.py"),
        remote_path=f"{REMOTE_APP_DIR}/main.py",
        copy=True,
    )
    .add_local_dir(
        str(_local_root / "src"),
        remote_path=f"{REMOTE_APP_DIR}/src",
        copy=True,
    )
)

# ── Volumes ──────────────────────────────────────────────────────────────────

artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME, create_if_missing=True
)
dataset_volume = modal.Volume.from_name(
    DATASET_VOLUME_NAME, create_if_missing=True
)

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App(APP_NAME)


# ── Gestão do dataset ────────────────────────────────────────────────────────

def _dataset_exists_in_volume() -> bool:
    try:
        entries = dataset_volume.listdir("/", recursive=True)
        paths = {e.path.lstrip("/") for e in entries}
        # listdir recursivo lista arquivos, não a pasta em si — prefixo cobre
        # tanto sentinels de arquivo quanto de diretório.
        return all(
            any(p == s or p.startswith(s + "/") for p in paths)
            for s in DATASET_SENTINELS
        )
    except Exception:
        return False


def _sync_new_features(local_raw: Path) -> None:
    """Sobe arquivos de dataset/raw/new_features ausentes (ou com tamanho
    diferente) no volume — features novas entram sem re-enviar o dataset
    inteiro. Caches locais (`_*`) não sobem."""
    local_root = local_raw / "new_features"
    if not local_root.is_dir():
        return
    try:
        remote = {
            e.path.lstrip("/"): e.size
            for e in dataset_volume.listdir("/raw/new_features", recursive=True)
        }
    except Exception:
        remote = {}
    pending = [
        f for f in sorted(local_root.rglob("*.nc"))
        if not f.name.startswith("_")
        and remote.get(f"raw/{f.relative_to(local_raw).as_posix()}") != f.stat().st_size
    ]
    if not pending:
        return
    print(f"Enviando {len(pending)} arquivo(s) novo(s) de new_features para o volume...")
    with dataset_volume.batch_upload(force=True) as upload:
        for f in pending:
            upload.put_file(str(f), f"/raw/{f.relative_to(local_raw).as_posix()}")


def _ensure_dataset(force: bool = False) -> None:
    """Garante que raw/ e shp/ estão no volume Modal."""
    local_raw = _local_root / "dataset" / "raw"
    local_shp = _local_root / "dataset" / "shp"

    for d in (local_raw, local_shp):
        if not d.exists():
            raise FileNotFoundError(
                f"'{d}' não encontrado localmente. "
                "Certifique-se de que o dataset está em dataset/."
            )

    if not force and _dataset_exists_in_volume():
        print(
            f"Dataset já disponível no volume "
            f"'{DATASET_VOLUME_NAME}' — pulando upload."
        )
        _sync_new_features(local_raw)
        return

    action = "Re-enviando" if force else "Enviando"
    nc_mb = sum(
        f.stat().st_size for f in local_raw.glob("*.nc")
    ) / 1024**2
    print(
        f"{action} dataset para volume '{DATASET_VOLUME_NAME}' "
        f"({nc_mb:.0f} MB de NetCDF)..."
    )

    with dataset_volume.batch_upload(force=True) as upload:
        upload.put_directory(str(local_raw), "/raw")
        upload.put_directory(str(local_shp), "/shp")

    print("Upload concluído.")


# ── Execução: fan-out (1 container por cluster) ───────────────────────────────

_VOLUMES = {
    REMOTE_ARTIFACTS_DIR: artifact_volume,
    REMOTE_DATASET_DIR: dataset_volume,
}


@app.function(
    image=image, volumes=_VOLUMES, timeout=21600, memory=32768,
    nonpreemptible=True,  # merge sem checkpoint — preempção reinicia do zero
)
def build_dataset_cache(interp_method: str = "nearest") -> str:
    """Gera o cache do merge ERA5 completo (INMET × ERA5_Stratified ×
    ERA5-Basin) UMA vez, no volume Modal — reaproveitado por Lazy/MLP/LSTM via
    NetCDFLoader.load_extended(), evitando recalcular o merge caro em cada
    uma das dezenas de invocações da matriz de ablation.

    `interp_method="bilinear"` (usado pela LSTM v2) grava um cache separado,
    `era5_merged_cache_bilinear.nc` — o `nearest` de lazy/mlp não muda.
    """
    dataset_volume.reload()

    sys.path.insert(0, REMOTE_APP_DIR)
    from src.data.netcdf_loader import NetCDFLoader

    raw_dir = Path(REMOTE_DATASET_DIR) / "raw"
    cache_path = raw_dir / NetCDFLoader.merged_cache_name(interp_method)
    # Checkpoint do merge das 4 fontes (ANTES do rebuild_original_from_basin,
    # que é onde timeout/preempção têm historicamente batido) — se essa
    # função for re-executada depois de uma falha, pula direto pro passo caro
    # seguinte em vez de refazer os ~5-10min de carregamento das 4 fontes.
    # Sufixado pelo método: um checkpoint nearest não pode alimentar o bilinear.
    suffix = "" if interp_method == "nearest" else f"_{interp_method}"
    checkpoint_path = raw_dir / f"era5_merge_checkpoint{suffix}.nc"

    print(f"[modal] Gerando cache do merge ERA5 {cache_path.name} "
          f"(interp={interp_method}, ignora cache existente)...")
    _t0 = time.time()
    ds_inmet, ds_era5 = NetCDFLoader(str(raw_dir)).load_extended(
        use_cache=False,
        checkpoint_path=checkpoint_path,
        on_checkpoint_saved=dataset_volume.commit,
        interp_method=interp_method,
    )
    print(f"[modal] Merge concluído em {time.time() - _t0:.0f}s: "
          f"{len(ds_inmet.estacao)} estações, "
          f"{len(ds_era5.time)} timesteps, {len(ds_era5.data_vars)} variáveis.")

    if cache_path.exists():
        cache_path.unlink()
    _t1 = time.time()
    NetCDFLoader.without_new_features(ds_era5).to_netcdf(cache_path)
    print(f"[modal] Cache salvo em {time.time() - _t1:.0f}s: {cache_path} "
          f"({cache_path.stat().st_size / 1024**2:.1f} MB)")

    # Checkpoint intermediário não serve mais depois do cache final pronto —
    # remove pra não confundir uma próxima chamada de --build-cache "do zero".
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    dataset_volume.commit()
    print(f"[modal] Cache commitado no volume '{DATASET_VOLUME_NAME}'.")
    return str(cache_path)


def _base_cmd(
    cluster_merge: str | None,
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    feature_groups: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",  # stdout sem buffer — sem isso, print()s do subprocess só
               # aparecem quando o processo termina (bufferizado por padrão
               # quando a saída não é um terminal, como aqui via pipe).
        f"{REMOTE_APP_DIR}/main.py",
        "cluster_lazy",
        "--raw-dir", f"{REMOTE_DATASET_DIR}/raw",
        "--shp-dir", f"{REMOTE_DATASET_DIR}/shp",
        "--output-dir", f"{REMOTE_ARTIFACTS_DIR}/lazy_clusters",
    ]
    if cluster_merge:
        cmd += ["--cluster-merge", cluster_merge]
    if not stratify_seasons:
        cmd += ["--no-stratify-seasons"]
    if n_neighbor_clusters != 1:
        cmd += ["--n-neighbor-clusters", str(n_neighbor_clusters)]
    if eval_window != "monthly":
        cmd += ["--eval-window", eval_window]
    if feature_groups:
        cmd += ["--feature-groups", feature_groups]
    return cmd


def _commit_safe() -> None:
    """Commit do volume com 1 retry após reload (escrita concorrente)."""
    try:
        artifact_volume.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[modal] commit falhou ({exc}); reload + retry...")
        artifact_volume.reload()
        artifact_volume.commit()


@app.function(image=image, volumes=_VOLUMES, timeout=3600, memory=16384)
def discover_clusters(
    cluster_merge: str | None = None,
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    exp_name: str | None = None,
    feature_groups: str | None = None,
) -> tuple[str, list[str]]:
    """Carrega os dados, resolve o exp e lista os clusters presentes."""
    import subprocess

    extra = ["--list-clusters"]
    if exp_name:
        extra += ["--exp-name", exp_name]
    base = _base_cmd(cluster_merge, stratify_seasons,
                     n_neighbor_clusters, eval_window, feature_groups)
    res = subprocess.run(
        base + extra,
        cwd=REMOTE_APP_DIR, text=True, capture_output=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(
            f"--list-clusters falhou (exit {res.returncode}):\n{res.stderr}"
        )

    resolved_exp: str | None = None
    clusters: list[str] = []
    for line in res.stdout.splitlines():
        if line.startswith("EXP_NAME:"):
            resolved_exp = line.split(":", 1)[1].strip()
        elif line.startswith("CLUSTERS_JSON:"):
            clusters = json.loads(line.split(":", 1)[1])
    if not resolved_exp or not clusters:
        raise RuntimeError("Falha ao resolver exp/clusters.")

    _commit_safe()
    print(f"[modal] Experimento: {resolved_exp} | clusters: {clusters}")
    return resolved_exp, clusters


@app.function(image=image, volumes=_VOLUMES, timeout=10800, memory=16384)
def process_cluster(
    cid: str,
    exp_name: str,
    cluster_merge: str | None = None,
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    feature_groups: str | None = None,
) -> tuple[str, bool]:
    """Treina todos os modelos de UM cluster no seu próprio container."""
    import subprocess

    artifact_volume.reload()
    print(f"[modal] ▶ Cluster {cid} (exp {exp_name})")
    base = _base_cmd(cluster_merge, stratify_seasons,
                     n_neighbor_clusters, eval_window, feature_groups)
    # Streaming linha-a-linha (em vez de capture_output + print no final) —
    # com 14 clusters rodando em paralelo (.starmap), cada linha vem
    # prefixada com o cluster de origem pra dar pra acompanhar ao vivo qual
    # cluster/trimestre está em qual ponto do treino.
    proc = subprocess.Popen(
        base + ["--exp-name", exp_name, "--cluster-id", str(cid)],
        cwd=REMOTE_APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    for line in proc.stdout:
        print(f"[c{cid}] {line}", end="", flush=True)
    proc.wait()
    ok = proc.returncode == 0
    if not ok:
        print(f"[modal] ✗ Cluster {cid} FALHOU (exit {proc.returncode}).")
    _commit_safe()
    if ok:
        print(f"[modal] ✓ Cluster {cid} salvo.")
    return str(cid), ok


@app.function(image=image, volumes=_VOLUMES, timeout=1800, memory=16384)
def aggregate_clusters(
    exp_name: str,
    cluster_merge: str | None = None,
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    feature_groups: str | None = None,
    ablation_group: str | None = None,
) -> list[str]:
    """Junta os parciais → CSV final + heatmap + run_meta."""
    import subprocess

    artifact_volume.reload()
    print("[modal] Construindo agregados (CSV + heatmap + meta)...")
    base = _base_cmd(cluster_merge, stratify_seasons,
                     n_neighbor_clusters, eval_window, feature_groups)
    extra = ["--exp-name", exp_name, "--aggregate-only"]
    if ablation_group:
        extra += ["--ablation-group", ablation_group]
    res = subprocess.run(
        base + extra,
        cwd=REMOTE_APP_DIR, text=True, capture_output=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(f"[modal] Agregação falhou:\n{res.stderr}")
    _commit_safe()

    exp_dir = Path(REMOTE_ARTIFACTS_DIR) / "lazy_clusters" / exp_name
    created = sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR))
        for p in exp_dir.rglob("*") if p.is_file()
    )
    print(f"[modal] {len(created)} artefato(s) no experimento.")
    return created


@app.function(image=image, volumes=_VOLUMES, timeout=7200, memory=16384)
def generate_spatial_maps(
    exp_name: str,
    year: str = "2023",
    smoothing: str = "gaussian",
) -> list[str]:
    """Usa os melhores modelos do LazyPredict para gerar mapa corrigido.

    Carrega os artefatos joblib de fitted_models/ e interpola para grade ERA5.

    Saída salva em:
      <exp>/spatial_maps/era5_corrigido_YYYY_<smoothing>.nc
    """
    import subprocess

    artifact_volume.reload()

    exp_dir = Path(REMOTE_ARTIFACTS_DIR) / "lazy_clusters" / exp_name
    models_dir = exp_dir / "fitted_models"
    maps_dir = exp_dir / "spatial_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists() or not list(models_dir.glob("*.joblib")):
        print(f"[modal] ⚠ Nenhum modelo salvo em {models_dir} — pulando mapas.")
        return []

    out_nc = maps_dir / f"era5_corrigido_{year}_{smoothing}.nc"

    print(f"[modal] Gerando mapa corrigido: {year} ({smoothing})...")
    print(f"[modal] Modelos: {models_dir}")
    res = subprocess.run(
        [
            sys.executable,
            "-m", "src.inference.spatial_correction",
            "--raw-dir", f"{REMOTE_DATASET_DIR}/raw",
            "--shp-dir", f"{REMOTE_DATASET_DIR}/shp",
            "--models-dir", str(models_dir),
            "--out-dir", str(maps_dir),
            "--year", year,
            "--smoothing", smoothing,
        ],
        cwd=REMOTE_APP_DIR, text=True, capture_output=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(f"[modal] ✗ Inferência espacial FALHOU:\n{res.stderr}")
    else:
        print(f"[modal] ✓ NetCDF salvo: {out_nc}")

    _commit_safe()

    created = sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR))
        for p in maps_dir.rglob("*") if p.is_file()
    )
    print(f"[modal] {len(created)} artefato(s) espaciais.")
    return created


# ── Download de artefatos ────────────────────────────────────────────────────

def _download_artifacts(local_dir: str, paths: list[str]) -> int:
    vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for rel in paths:
        remote_path = f"/{rel.lstrip('/')}"
        local_path = local_root / rel.lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = list(vol.read_file(remote_path))
        local_path.write_bytes(b"".join(chunks))
        print(f"  ↓ {rel}")
        count += 1

    return count


def _download_all_artifacts(local_dir: str) -> int:
    from modal.volume import FileEntryType

    vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in vol.listdir("/", recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = entry.path.lstrip("/")
        local_path = local_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = list(vol.read_file(entry.path))
        local_path.write_bytes(b"".join(chunks))
        print(f"  ↓ {rel}")
        count += 1

    return count


# ── Entrypoint local ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    download: bool = True,
    only_download: bool = False,
    only_upload_dataset: bool = False,
    build_cache: bool = False,
    interp_method: str = "nearest",
    local_dir: str = "artifacts/lazy_modal",
    force_dataset_upload: bool = False,
    cluster_merge: str = "",
    stratify_seasons: bool = True,
    n_neighbor_clusters: int = 1,
    eval_window: str = "monthly",
    exp_name: str = "",
    feature_groups: str = "",
    ablation_group: str = "",
    spatial_year: str = "2023",
    spatial_smoothing: str = "gaussian",
    skip_spatial: bool = False,
    only_clusters: str = "",
) -> None:
    """
    Flags
    -----
    --no-download          Roda mas não baixa artefatos
    --only-download        Só baixa; não roda
    --only-upload-dataset  Só sincroniza dataset/raw+shp pro volume; não roda
                           nem baixa (volume compartilhado por lazy/mlp/lstm)
    --build-cache          Gera/regenera o cache do merge ERA5 completo no
                           volume (era5_merged_cache.nc) e sai. Lazy/MLP/
                           LSTM passam a usar esse cache automaticamente
                           (features novas ficam fora dele, sempre relidas).
    --interp-method        Com --build-cache: nearest (default, cache atual)
                           ou bilinear (LSTM v2 → era5_merged_cache_bilinear.nc)
    --only-clusters       Retoma só os clusters listados (ex: "2,5,9,13"),
                           em vez de rodar todos de novo — usa o MESMO
                           --exp-name de uma run anterior interrompida
                           (ex: por timeout), sem reprocessar os que já têm
                           resultado salvo no volume.
    --local-dir            Destino local (default: artifacts/lazy_modal)
    --force-dataset-upload Re-envia dataset mesmo se já no volume
    --cluster-merge        Agrega clusters, ex: "1-2-3,5-6"
    --exp-name             Nome do experimento; senão autoincrementa exp{n}
    --feature-groups       Grupos separados por vírgula: original, era5_basin,
                           new_features, new_features_static,
                           new_features_dynamic (ou 'all'; default: original).
                           Com qualquer new_features*, só treinam clusters
                           100% cobertos pelas features pedidas
    --ablation-group       Tag opcional p/ agrupar experimentos de ablation
    --spatial-year         Ano para gerar mapa corrigido (default: 2023)
    --spatial-smoothing    Suavização: gaussian | none (default: gaussian)
    --skip-spatial         Pular geração de mapas espaciais
    """
    if only_download:
        print(f"\nBaixando todos os artefatos para '{local_dir}'...")
        n = _download_all_artifacts(local_dir)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
        return

    if only_upload_dataset:
        _ensure_dataset(force=force_dataset_upload)
        return

    if build_cache:
        _ensure_dataset(force=force_dataset_upload)
        print("\nGerando cache do merge ERA5 completo no Modal (via spawn — "
              "desacoplado da conexão local; uma queda de rede aqui não "
              "cancela mais a task remota)...")
        call = build_dataset_cache.spawn(interp_method=interp_method)
        print(f"Function call id: {call.object_id}")
        print("Se a conexão cair durante o .get(), retome com:\n"
              "  python3 -c \"import modal; "
              f"print(modal.FunctionCall.from_id('{call.object_id}').get())\"")
        cache_path = call.get()
        print(f"\nCache pronto: {cache_path}")
        return

    _ensure_dataset(force=force_dataset_upload)

    cm = cluster_merge or None

    # ── 1. Descobre clusters + resolve nome do experimento ────────────────
    print("\nDescobrindo clusters...")
    resolved_exp, clusters = discover_clusters.remote(
        cluster_merge=cm,
        stratify_seasons=stratify_seasons,
        n_neighbor_clusters=n_neighbor_clusters,
        eval_window=eval_window,
        exp_name=exp_name or None,
        feature_groups=feature_groups or None,
    )
    print(f"Experimento: {resolved_exp} | {len(clusters)} cluster(s)")

    if only_clusters:
        wanted = {c.strip() for c in only_clusters.split(",") if c.strip()}
        clusters = [c for c in clusters if str(c) in wanted]
        print(f"--only-clusters: retomando só {clusters}")

    # ── 2. Fan-out: 1 container por cluster, em paralelo ──────────────────
    print("\nProcessando clusters em paralelo (1 container cada)...")
    args = [
        (cid, resolved_exp, cm, stratify_seasons, n_neighbor_clusters, eval_window,
         feature_groups or None)
        for cid in clusters
    ]
    results = list(process_cluster.starmap(args))
    ok = [c for c, good in results if good]
    failed = [c for c, good in results if not good]
    print(f"\n{len(ok)}/{len(clusters)} clusters OK.")
    if failed:
        print(f"Falharam: {failed}")

    # ── 3. Agrega o que terminou ──────────────────────────────────────────
    print("\nAgregando resultados...")
    created_files = aggregate_clusters.remote(
        exp_name=resolved_exp, cluster_merge=cm,
        stratify_seasons=stratify_seasons,
        n_neighbor_clusters=n_neighbor_clusters,
        eval_window=eval_window,
        feature_groups=feature_groups or None,
        ablation_group=ablation_group or None,
    )
    print(
        f"\nAgregação completa — {len(created_files)} artefato(s) "
        f"no volume '{ARTIFACT_VOLUME_NAME}'."
    )

    # ── 4. Inferência Espacial + Mapas ────────────────────────────────────
    if not skip_spatial:
        print(f"\nGerando mapas espaciais ({spatial_year}, {spatial_smoothing})...")
        spatial_files = generate_spatial_maps.remote(
            exp_name=resolved_exp,
            year=spatial_year,
            smoothing=spatial_smoothing,
        )
        created_files.extend(spatial_files)
        print(f"Mapas: {len(spatial_files)} artefato(s) espaciais gerados.")
    else:
        print("\nMapas espaciais pulados (--skip-spatial).")

    print(
        f"\nRun completa — {len(created_files)} artefato(s) totais "
        f"no volume '{ARTIFACT_VOLUME_NAME}'."
    )

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts(local_dir, paths=created_files)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
