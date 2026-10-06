"""
IRC Vendaval — Modal deployment (MLPRegressor por cluster)
==========================================================
Executa a pipeline MLP com extreme-weighting por cluster espacial na nuvem
(Modal) e baixa os artefatos gerados para o diretório local.

Arquitetura do Modelo e Por Quê
-------------------------------
O modelo base é um `MLPRegressor` (Scikit-Learn) treinado individualmente
para cada cluster espacial com as seguintes particularidades de desenho:
- Camadas Ocultas: Configuração padrão `(128, 64)` com ativação ReLU e otimizador Adam.
- Regularização: Penalidade L2 (`alpha=0.001`), `early_stopping=True` com fração de validação de 10%, paciência de 30 épocas sem melhora (`n_iter_no_change=30`) e limite máximo de `max_iter=500` iterações do solver.
- Target Formulation (Fator de Correção): Ao invés de prever a rajada absoluta diretamente, o MLP aprende a razão `y_true / era5_wind_mag_max`. Durante a inferência, a predição final é reconstruída como `y_pred = ratio_pred * era5_wind_mag_max`.
- Extreme Weighting (Oversampling Ponderado): As amostras recebem pesos `w = (y / y_max)^extreme_power`. O conjunto de treino sofre oversampling proporcional aos pesos antes do ajuste.

O porquê desta arquitetura: Redes densas clássicas sofrem com desbalanceamento 
quando a distribuição do vento é fortemente assimétrica (cauda longa à direita). 
Modelar a razão (`ratio`) permite que a rede atue diretamente como um operador de 
correção de viés multiplicativo sobre a física do ERA5. Além disso, o 
extreme weighting foca a capacidade representacional da rede neural nos 
eventos mais severos e danosos sem distorcer o domínio físico dos dados.

Uso
---
# Rodar e baixar artefatos (default)
modal run src/modal/cluster_mlp.py

# Parâmetros do modelo
modal run src/modal/cluster_mlp.py \\
    --hidden-layers 256,128,64 \\
    --alpha 0.0005 \\
    --extreme-power 3.0

# Rodar sem baixar
modal run src/modal/cluster_mlp.py --no-download

# Só baixar artefatos de uma run já concluída
modal run src/modal/cluster_mlp.py --only-download \\
    --local-dir artifacts/mlp_modal

# Forçar re-upload dos dados
modal run src/modal/cluster_mlp.py --force-dataset-upload
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── Constantes ───────────────────────────────────────────────────────────────

APP_NAME = "irc-vendaval-mlp-clusters"
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
    # v4 — stations_metadata + predictions_by_station + remove lat/lon features
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
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "xarray>=2024.1.0",
        "netCDF4>=1.6.0",
        "geopandas>=0.14.0",
        "pyshp>=2.3.0",
        "shap>=0.46.0",
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
    # 21600s (6h): 1 container processa os 14 clusters sequencialmente (MLP +
    # SHAP KernelExplainer + permutation importance + ~6 plots/cluster) —
    # 5400s (90s/cluster) já se mostrou curto pra esse volume de trabalho.
    timeout=21600,
    memory=16384,
    nonpreemptible=True,  # sem checkpoint por cluster — preempção reinicia
                           # os 14 clusters do zero, então evita preempção.
)
def run_mlp_clusters(
    hidden_layers: str = "128,64",
    alpha: float = 0.001,
    extreme_power: float = 2.0,
    max_iter: int = 500,
    cluster_merge: str | None = None,
    exp_name: str | None = None,
    feature_groups: str | None = None,
    ablation_group: str | None = None,
) -> list[str]:
    """
    Executa main_mlp_clusters.py dentro do container Modal.
    Retorna lista de caminhos relativos dos artefatos criados.
    """
    import subprocess

    raw_dir = f"{REMOTE_DATASET_DIR}/raw"
    shp_dir = f"{REMOTE_DATASET_DIR}/shp"
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/mlp_clusters"

    print(f"[modal] hidden_layers={hidden_layers}")
    print(f"[modal] alpha={alpha}  extreme_power={extreme_power}  max_iter={max_iter}")
    print(f"[modal] raw: {raw_dir}")
    print(f"[modal] output: {output_dir}")
    if cluster_merge:
        print(f"[modal] cluster_merge: {cluster_merge}")


    # Cada run cria seu próprio exp{n} dentro de output_dir — não limpamos
    # o diretório para preservar os experimentos anteriores e o índice.

    artifacts_root = Path(REMOTE_ARTIFACTS_DIR)
    before: set[str] = (
        {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
        if artifacts_root.exists()
        else set()
    )

    cmd = [
        sys.executable,
        f"{REMOTE_APP_DIR}/main.py",
        "cluster_mlp",
        "--raw-dir", raw_dir,
        "--shp-dir", shp_dir,
        "--output-dir", output_dir,
        "--hidden-layers", hidden_layers,
        "--alpha", str(alpha),
        "--extreme-power", str(extreme_power),
        "--max-iter", str(max_iter),
    ]
    if cluster_merge:
        cmd += ["--cluster-merge", cluster_merge]
    if exp_name:
        cmd += ["--exp-name", exp_name]
    if feature_groups:
        cmd += ["--feature-groups", feature_groups]
    if ablation_group:
        cmd += ["--ablation-group", ablation_group]

    # Streaming linha-a-linha — antes usava capture_output=True, que só
    # imprime tudo de uma vez quando o processo INTEIRO termina (até 6h);
    # não dava pra saber se estava progredindo ou travado.
    proc = subprocess.Popen(
        cmd, cwd=REMOTE_APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"main_mlp_clusters.py falhou (exit code {proc.returncode})."
        )

    artifact_volume.commit()

    after = {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
    created = sorted(
        str(Path(p).relative_to(REMOTE_ARTIFACTS_DIR))
        for p in (after - before)
    )
    print(f"[modal] {len(created)} artefato(s) novos.")
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


@app.function(
    image=image,
    volumes={
        REMOTE_ARTIFACTS_DIR: artifact_volume,
        REMOTE_DATASET_DIR: dataset_volume,
    },
    timeout=7200,
    memory=16384,
)
def generate_spatial_maps(
    exp_dir_path: str,
    year: str = "2023",
    smoothing: str = "gaussian",
) -> list[str]:
    """Usa os modelos MLP para gerar mapa corrigido.

    Saída salva em:
      <exp_dir>/spatial_maps/era5_corrigido_YYYY_<smoothing>.nc
    """
    import subprocess

    artifact_volume.reload()

    exp_dir = Path(exp_dir_path)
    models_dir = exp_dir / "fitted_models"
    maps_dir = exp_dir / "spatial_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists() or not list(models_dir.glob("*.joblib")):
        print(f"[modal] ⚠ Nenhum modelo salvo em {models_dir} — pulando mapas.")
        return []

    out_nc = maps_dir / f"era5_corrigido_{year}_{smoothing}.nc"

    print(f"[modal] Gerando mapa corrigido MLP: {year} ({smoothing})...")
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
        print(f"[modal] ✗ Inferência espacial MLP FALHOU:\n{res.stderr}")
    else:
        print(f"[modal] ✓ NetCDF salvo: {out_nc}")

    artifact_volume.commit()

    created = sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR))
        for p in maps_dir.rglob("*") if p.is_file()
    )
    print(f"[modal] {len(created)} artefato(s) espaciais MLP.")
    return created


# ── Entrypoint local ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    download: bool = True,
    only_download: bool = False,
    only_upload_dataset: bool = False,
    local_dir: str = "artifacts/mlp_modal",
    force_dataset_upload: bool = False,
    hidden_layers: str = "128,64",
    alpha: float = 0.001,
    extreme_power: float = 2.0,
    max_iter: int = 500,
    cluster_merge: str = "",
    exp_name: str = "",
    feature_groups: str = "",
    ablation_group: str = "",
    spatial_year: str = "2023",
    spatial_smoothing: str = "gaussian",
    skip_spatial: bool = False,
) -> None:
    """
    Flags
    -----
    --no-download          Roda mas não baixa artefatos
    --only-download        Só baixa; não roda
    --only-upload-dataset  Só sincroniza dataset/raw+shp pro volume; não roda
                           nem baixa (volume compartilhado por lazy/mlp/lstm)
    --local-dir            Destino local (default: artifacts/mlp_modal)
    --force-dataset-upload Re-envia dataset mesmo se já no volume
    --hidden-layers        Camadas ocultas (default: "128,64")
    --alpha                Regularização L2 (default: 0.001)
    --extreme-power        Expoente do sample_weight (default: 2.0)
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

    _ensure_dataset(force=force_dataset_upload)

    print("\nSubmetendo pipeline MLP × Cluster...")
    created_files = run_mlp_clusters.remote(
        hidden_layers=hidden_layers,
        alpha=alpha,
        extreme_power=extreme_power,
        max_iter=max_iter,
        cluster_merge=cluster_merge or None,
        exp_name=exp_name or None,
        feature_groups=feature_groups or None,
        ablation_group=ablation_group or None,
    )

    if created_files:
        first_file = Path(created_files[0])
        exp_dir_rel = Path(first_file.parts[0]) / first_file.parts[1]
        exp_dir_path = f"{REMOTE_ARTIFACTS_DIR}/{exp_dir_rel}"
        
        if not skip_spatial:
            print(f"\nGerando mapas espaciais ({spatial_year}, {spatial_smoothing})...")
            spatial_files = generate_spatial_maps.remote(
                exp_dir_path=exp_dir_path,
                year=spatial_year,
                smoothing=spatial_smoothing,
            )
            created_files.extend(spatial_files)
            print(f"Mapas: {len(spatial_files)} artefato(s) espaciais gerados.")
        else:
            print("\nMapas espaciais pulados (--skip-spatial).")

    print(f"\nRun completa — {len(created_files)} artefato(s) totais:")
    for f in sorted(created_files):
        print(f"  {REMOTE_ARTIFACTS_DIR}/{f}")

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts_dir(local_dir, remote_subdir="mlp_clusters")
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
