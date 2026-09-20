"""
IRC Vendaval — Modal deployment (GAN condicional tabular por cluster)
======================================================================
Executa a pipeline cluster_gan (WGAN-GP condicional + EVT/GPD +
nearest-neighbor) na nuvem (Modal) para gerar dados sintéticos de rajadas
extremas, e baixa o synthetic_augment.csv resultante.

Arquitetura do Modelo e Metodologia ExGAN
-----------------------------------------
1. Teoria dos Valores Extremos (EVT / GPD): As rajadas acima do percentil extremo
   (P90 padrão) são ajustadas a uma Generalized Pareto Distribution (GPD).
2. Distribution Shifting: k rodadas de corte e geração auxiliar deslocam a
   distribuição do dataset de treino para a cauda extrema, evitando que o
   gerador sofra com o desbalanceamento onde 90% dos dados têm condição nula.
3. Conditional WGAN-GP Alvo-Only:
   - Condição: One-hot de cluster + Season opcional + Excedência normalizada GPD.
   - Gerador: Dense(128, ELU) -> Dense(128, ELU) -> Dense(1) [prediz rajada escalar].
   - Crítico: Dense(128, ELU) -> Dense(64, ELU) -> Dense(1) [Wasserstein score].
   - Perda combinada: WGAN Wasserstein + Gradient Penalty (λ=10) + L_ext (λ=5, erro relativo do extremo).
4. Pareamento de Features por Nearest-Neighbor: O alvo sintético y_synth gerado
   recebe o vetor de covariáveis meteorológicas do evento real mais próximo,
   preservando correlações físicas e atmosféricas perfeitamente plausíveis.

Uso
---
# Rodar e baixar o CSV sintético (default)
modal run src/modal/cluster_gan.py --exp-name gan_v1

# Parâmetros do GAN
modal run src/modal/cluster_gan.py \\
    --epochs 300 --extreme-percentile 90 --n-per-cluster-ratio 0.3

# Só baixar artefatos de uma run já concluída
modal run src/modal/cluster_gan.py --only-download \\
    --local-dir artifacts/gan_modal
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── Constantes ───────────────────────────────────────────────────────────────

APP_NAME = "irc-vendaval-gan-clusters"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

DATASET_SENTINELS = [
    "raw/INMET_Stratified.nc",
    "raw/ERA5_Stratified.nc",
    "raw/ERA5_Features_Basin_2000_2026.nc",
]

_local_root = Path(__file__).parent.parent.parent

# ── Imagem ───────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libhdf5-dev",
        "libnetcdf-dev",
        "gdal-bin",
        "libgdal-dev",
        "libproj-dev",
    )
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "tensorflow>=2.15.0",
        "xarray>=2024.1.0",
        "netCDF4>=1.6.0",
        "geopandas>=0.14.0",
        "pyshp>=2.3.0",
    )
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
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
# Compartilha os mesmos volumes das outras pipelines — o dataset já enviado
# por qualquer uma delas serve aqui também (mesmo sentinel de verificação).

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


# ── Função de execução ───────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        REMOTE_ARTIFACTS_DIR: artifact_volume,
        REMOTE_DATASET_DIR: dataset_volume,
    },
    timeout=10800,
    memory=16384,
    gpu="T4",
)
def run_cluster_gan(
    epochs: int = 300,
    extreme_percentile: float = 90.0,
    n_per_cluster: int | None = None,
    n_per_cluster_ratio: float | None = 0.3,
    include_season: bool = False,
    cluster_merge: str | None = None,
    exp_name: str | None = None,
) -> list[str]:
    """Executa main.py cluster_gan dentro do container Modal (com GPU)."""
    import subprocess

    raw_dir = f"{REMOTE_DATASET_DIR}/raw"
    shp_dir = f"{REMOTE_DATASET_DIR}/shp"
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/gan_clusters"

    print(f"[modal] epochs={epochs}  extreme_percentile={extreme_percentile}")
    print(f"[modal] raw: {raw_dir}")
    print(f"[modal] output: {output_dir}")

    artifacts_root = Path(REMOTE_ARTIFACTS_DIR)
    before: set[str] = (
        {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
        if artifacts_root.exists()
        else set()
    )

    cmd = [
        sys.executable,
        f"{REMOTE_APP_DIR}/main.py",
        "cluster_gan",
        "--raw-dir", raw_dir,
        "--shp-dir", shp_dir,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--extreme-percentile", str(extreme_percentile),
    ]
    if n_per_cluster is not None:
        cmd += ["--n-per-cluster", str(n_per_cluster)]
    if n_per_cluster_ratio is not None:
        cmd += ["--n-per-cluster-ratio", str(n_per_cluster_ratio)]
    if include_season:
        cmd += ["--include-season"]
    if cluster_merge:
        cmd += ["--cluster-merge", cluster_merge]
    if exp_name:
        cmd += ["--exp-name", exp_name]

    # Streaming linha-a-linha — antes usava capture_output=True, que só
    # imprime tudo de uma vez quando o processo INTEIRO termina (até 3h);
    # não dava pra saber se estava progredindo ou travado.
    proc = subprocess.Popen(
        cmd, cwd=REMOTE_APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"cluster_gan falhou (exit code {proc.returncode}).")

    artifact_volume.commit()

    after = {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
    created = sorted(
        str(Path(p).relative_to(REMOTE_ARTIFACTS_DIR))
        for p in (after - before)
    )
    print(f"[modal] {len(created)} artefato(s) novos.")
    return created


# ── Download de artefatos ────────────────────────────────────────────────────

def _download_artifacts_dir(local_dir: str, remote_subdir: str) -> int:
    """Baixa todos os arquivos de um subdiretório específico do volume."""
    from modal.volume import FileEntryType

    vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in vol.listdir(f"/{remote_subdir}", recursive=True):
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
    local_dir: str = "artifacts/gan_modal",
    force_dataset_upload: bool = False,
    epochs: int = 300,
    extreme_percentile: float = 90.0,
    n_per_cluster: int = 0,
    n_per_cluster_ratio: float = 0.3,
    include_season: bool = False,
    cluster_merge: str = "",
    exp_name: str = "",
) -> None:
    """
    Flags
    -----
    --no-download           Roda mas não baixa artefatos
    --only-download         Só baixa; não roda
    --only-upload-dataset   Só sincroniza dataset/raw+shp pro volume; não roda
                            nem baixa (volume compartilhado por lazy/mlp/lstm/gan)
    --local-dir             Destino local (default: artifacts/gan_modal)
    --force-dataset-upload  Re-envia dataset mesmo se já no volume
    --epochs                Épocas de treino do GAN (default: 300)
    --extreme-percentile    Percentil 0-100 que define "extremo" (default: 90)
    --n-per-cluster         Nº fixo de sintéticos por cluster (sobrepõe ratio)
    --n-per-cluster-ratio   Fração do treino do cluster (default: 0.3)
    --include-season        Condiciona também em estação climática
    --cluster-merge         Agrega clusters, ex: "1-2-3,5-6"
    --exp-name              Nome do experimento; senão autoincrementa exp{n}
    """
    if only_download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts_dir(local_dir, remote_subdir="gan_clusters")
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
        return

    if only_upload_dataset:
        _ensure_dataset(force=force_dataset_upload)
        return

    _ensure_dataset(force=force_dataset_upload)

    print("\nSubmetendo pipeline cluster_gan...")
    created_files = run_cluster_gan.remote(
        epochs=epochs,
        extreme_percentile=extreme_percentile,
        n_per_cluster=n_per_cluster or None,
        n_per_cluster_ratio=n_per_cluster_ratio,
        include_season=include_season,
        cluster_merge=cluster_merge or None,
        exp_name=exp_name or None,
    )

    print(f"\nRun completa — {len(created_files)} artefato(s) totais:")
    for f in sorted(created_files):
        print(f"  {REMOTE_ARTIFACTS_DIR}/{f}")

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts_dir(local_dir, remote_subdir="gan_clusters")
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
