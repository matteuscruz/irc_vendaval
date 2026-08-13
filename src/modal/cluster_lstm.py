"""
IRC Vendaval — Modal deployment (pipeline de clusters)
======================================================
Executa o pipeline dual-especialidade por cluster na nuvem (Modal) e baixa
os artefatos gerados para o diretório local.

Arquitetura do Modelo e Por Quê
-------------------------------
O modelo base é um LSTM em configuração Multi-Task Learning com um
backbone compartilhado e duas saídas paralelas (dual-especialidade):
- Backbone: `LSTM(units=64)` -> `BatchNormalization()` -> `Dense(16, elu)` -> `Dropout()`
- Saída 1 (`head_normal`): `Dense(1)` otimizada com Huber loss. Foca em prever o regime padrão e estável das rajadas.
- Saída 2 (`head_extreme`): `Dense(1)` otimizada com `robust_extreme_loss`. Foca unicamente em aprender a dinâmica de eventos extremos e raros.

O porquê desta arquitetura: Redes neurais tradicionais tendem a suavizar (subestimar) 
picos extremos ao tentar minimizar o erro médio. Ao separar a aprendizagem 
em duas cabeças isoladas, o modelo captura perfeitamente a variabilidade do 
clima regular, mas também possui uma "especialidade" em extremos. Durante 
a inferência, se a previsão da cabeça normal indicar uma rajada além do limiar, 
o modelo chaveia a resposta e substitui a saída final pela previsão da cabeça extrema.

Gestão de dados
---------------
Os datasets NetCDF e shapefiles são armazenados no volume
"irc-vendaval-dataset". Na primeira execução (ou com
--force-dataset-upload) o upload é feito automaticamente de
dataset/raw/ e dataset/shp/. Execuções seguintes reutilizam
o volume sem re-upload.

Uso
---
# Rodar experimento + baixar artefatos
modal run src/modal/cluster_lstm.py

# Rodar sem baixar artefatos
modal run src/modal/cluster_lstm.py --no-download

# Só baixar artefatos de uma run já concluída
modal run src/modal/cluster_lstm.py --only-download --local-dir artifacts/modal

# Forçar re-upload do dataset
modal run src/modal/cluster_lstm.py --force-dataset-upload

# Config diferente
modal run src/modal/cluster_lstm.py --config experiment_cluster_lstm_modal.yaml

# Com augmentação ExGAN
modal run src/modal/cluster_lstm.py --augmentation-method extreme_gan
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── Constantes ─────────────────────────────────────────────────────────────

APP_NAME = "irc-vendaval-clusters"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

# Sentinelas: se todos existirem no volume, o dataset já foi enviado
DATASET_SENTINELS = [
    "raw/INMET_Stratified.nc",
    "raw/ERA5_Stratified.nc",
    "raw/dados_era5_parana_18utc",
    "raw/dados_temperatura_brilho_BT55",
    "raw/ERA5_Features_Basin_2000_2026.nc",
]

_local_root = Path(__file__).parent.parent.parent

# ── Imagem ──────────────────────────────────────────────────────────────────
# Camadas da mais estável para a menos estável (melhor cache hit):
#   1. Base OS + CUDA  (tensorflow:2.16.1-gpu)
#   2. pip install (separado em dois blocos para cache granular)
#   3. src/, config/, main_clusters.py

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    .pip_install(
        # Dependências estáveis do projeto base
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.13.3",
        "pyyaml>=6.0.3",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
    )
    .pip_install(
        # Dependências específicas do pipeline de clusters
        "xarray>=2024.1.0",
        "netCDF4>=1.6.0",
        "geopandas>=0.14.0",
        "pyshp>=2.3.0",
        "scipy>=1.12.0",    # GPD / EVT do ExGAN
        # BT55: leitura de parquets mensais
        "pyarrow>=14.0.0",
        # ERA5-18UTC: xarray usa dask internamente
        "dask>=2024.1.0",
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "FORCE_CPU": "0",
            "TF_DETERMINISTIC": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(_local_root / "src"),
        remote_path=f"{REMOTE_APP_DIR}/src",
        copy=True,
    )
    .add_local_dir(
        str(_local_root / "config"),
        remote_path=f"{REMOTE_APP_DIR}/config",
        copy=True,
    )
    .add_local_file(
        str(_local_root / "main.py"),
        remote_path=f"{REMOTE_APP_DIR}/main.py",
        copy=True,
    )
)

# ── Volumes ─────────────────────────────────────────────────────────────────

artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME, create_if_missing=True
)
dataset_volume = modal.Volume.from_name(
    DATASET_VOLUME_NAME, create_if_missing=True
)

# ── App ─────────────────────────────────────────────────────────────────────

app = modal.App(APP_NAME)


# ── Gestão do dataset ───────────────────────────────────────────────────────

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


def _ensure_dataset(force: bool = False) -> None:
    """
    Garante que raw/ e shp/ estão no volume Modal.
    Upload somente se o sentinela não existir (ou force=True).
    """
    local_raw = _local_root / "dataset" / "raw"
    local_shp = _local_root / "dataset" / "shp"

    for d in (local_raw, local_shp):
        if not d.exists():
            raise FileNotFoundError(
                f"Diretório local '{d}' não encontrado. "
                "Extraia o zip antes de rodar: "
                "python -c \"import zipfile; "
                "zipfile.ZipFile('dataset/drive-download-*.zip')"
                ".extractall('dataset/')\""
            )

    if not force and _dataset_exists_in_volume():
        print(
            f"Dataset já disponível no volume "
            f"'{DATASET_VOLUME_NAME}' — pulando upload."
        )
        return

    action = "Re-enviando" if force else "Enviando"
    nc_size = (
        sum(f.stat().st_size for f in local_raw.glob("*.nc")) / 1024**2
    )
    print(
        f"{action} dataset para volume '{DATASET_VOLUME_NAME}' "
        f"({nc_size:.0f} MB de NetCDF)..."
    )

    with dataset_volume.batch_upload(force=force) as upload:
        upload.put_directory(str(local_raw), "/raw")
        upload.put_directory(str(local_shp), "/shp")

    print("Upload concluído.")


# ── Função de treinamento ────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        REMOTE_ARTIFACTS_DIR: artifact_volume,
        REMOTE_DATASET_DIR: dataset_volume,
    },
    gpu="A10G",      # A10G para runs com augmentação (GPU mais rápida)
    timeout=10800,   # 3 horas — margem extra: 271 estações/14 clusters
                     # aumentam tanto o load_extended() quanto o treino
    memory=32768,    # 32 GB RAM (NetCDF + xarray são mais exigentes)
    # nonpreemptible=True não suportado em workloads de GPU (Modal) — sem
    # checkpoint por cluster, uma preempção aqui reinicia os 14 do zero.
)
def run_experiment(
    config_name: str = "experiment_cluster_lstm_modal.yaml",
    augmentation_method: str | None = None,
    synthetic_csv: str | None = None,
    feature_groups: str | None = None,
    restrict_coverage: bool = False,
    ablation_group: str | None = None,
    exp_name: str | None = None,
) -> list[str]:
    """
    Executa main_clusters.py dentro do container Modal.
    Retorna lista de caminhos relativos dos artefatos criados.

    Parâmetros
    ----------
    config_name : str
        Nome do YAML em config/.
    augmentation_method : str | None
        Se definido, sobrescreve augmentation.method do YAML.
        Opções: 'extreme_gan', 'extreme_diffusion', 'none'.
    """
    import subprocess

    config_path = f"{REMOTE_APP_DIR}/config/{config_name}"
    print(f"[modal] Dataset: {REMOTE_DATASET_DIR}")
    print(f"[modal] Config: {config_path}")
    if augmentation_method:
        print(f"[modal] Augmentação: {augmentation_method}")

    artifacts_root = Path(REMOTE_ARTIFACTS_DIR)
    before: set[str] = (
        {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
        if artifacts_root.exists()
        else set()
    )

    cmd = [
        sys.executable,
        f"{REMOTE_APP_DIR}/main.py",
        "cluster_lstm",
        "--config", config_path,
    ]
    if augmentation_method:
        cmd += ["--augmentation-method", augmentation_method]
    if synthetic_csv:
        cmd += ["--synthetic-csv", synthetic_csv]
    if feature_groups:
        cmd += ["--feature-groups", feature_groups]
    if restrict_coverage:
        cmd += ["--restrict-coverage"]
    if ablation_group:
        cmd += ["--ablation-group", ablation_group]
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
        raise RuntimeError(f"cluster_lstm falhou (exit {proc.returncode}).")

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
    """Usa os melhores modelos DL para gerar mapa corrigido.

    Saída salva em:
      <exp_dir>/spatial_maps/era5_corrigido_YYYY_<smoothing>.nc
    """
    import subprocess

    artifact_volume.reload()

    exp_dir = Path(exp_dir_path)
    models_dir = exp_dir / "fitted_models"
    maps_dir = exp_dir / "spatial_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists() or not list(models_dir.glob("*.keras")):
        print(f"[modal] ⚠ Nenhum modelo salvo em {models_dir} — pulando mapas.")
        return []

    out_nc = maps_dir / f"era5_corrigido_{year}_{smoothing}.nc"

    print(f"[modal] Gerando mapa corrigido DL: {year} ({smoothing})...")
    res = subprocess.run(
        [
            sys.executable,
            "-m", "src.inference.spatial_correction_dl",
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
        print(f"[modal] ✗ Inferência espacial DL FALHOU:\n{res.stderr}")
    else:
        print(f"[modal] ✓ NetCDF salvo: {out_nc}")

    artifact_volume.commit()

    created = sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR))
        for p in maps_dir.rglob("*") if p.is_file()
    )
    print(f"[modal] {len(created)} artefato(s) espaciais DL.")
    return created


# ── Entrypoint local ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    config: str = "experiment_cluster_lstm_modal.yaml",
    augmentation_method: str | None = None,
    synthetic_csv: str | None = None,
    feature_groups: str | None = None,
    restrict_coverage: bool = False,
    ablation_group: str | None = None,
    exp_name: str | None = None,
    download: bool = True,
    only_download: bool = False,
    only_upload_dataset: bool = False,
    local_dir: str = "artifacts",
    force_dataset_upload: bool = False,
    spatial_year: str = "2023",
    spatial_smoothing: str = "gaussian",
    skip_spatial: bool = False,
) -> None:
    """
    Flags
    -----
    --config               YAML em config/
    --augmentation-method  Override augmentation.method
                           (extreme_gan | extreme_diffusion | none)
    --synthetic-csv        Path no volume para CSV de sintéticos do GAN
                           (ex: /artifacts/gan/synthetic_extremes.csv)
    --feature-groups       Grupos separados por vírgula: original, era5_18z,
                           bt55 (ou 'all'); sobrepõe data.feature_groups do YAML
    --restrict-coverage    Restringe às estações com cobertura REAL de
                           era5_18z/bt55 (~46-57/243) em vez de imputar NaN
                           nas demais — recomendado com --feature-groups
                           incluindo era5_18z/bt55 (ex: braço "newfeatures")
    --ablation-group       Tag opcional p/ agrupar experimentos de ablation
    --exp-name             Sobrepõe experiment.name do YAML
    --no-download          Roda o treino mas não baixa artefatos
    --only-download        Só baixa; não treina
    --only-upload-dataset  Só sincroniza dataset/raw+shp pro volume; não treina
                           nem baixa (volume 'irc-vendaval-dataset' é
                           compartilhado por lazy/mlp/lstm/gan — subir por
                           qualquer um dos wrappers basta)
    --local-dir            Destino local (default: artifacts/modal)
    --force-dataset-upload Re-envia dataset mesmo se já no volume
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

    remote_synthetic: str | None = None
    if synthetic_csv:
        local_p = Path(synthetic_csv)
        if not local_p.exists():
            raise FileNotFoundError(f"synthetic_csv não encontrado: {local_p}")
        remote_path = f"/synthetic/{local_p.name}"
        print(f"Enviando {local_p.name} → volume:{remote_path} ...")
        with artifact_volume.batch_upload(force=True) as upload:
            upload.put_file(str(local_p), remote_path)
        remote_synthetic = f"{REMOTE_ARTIFACTS_DIR}{remote_path}"
        print(f"Upload concluído: {remote_synthetic}")

    print(f"\nSubmetendo experimento: {config}")
    created_files = run_experiment.remote(
        config, augmentation_method, remote_synthetic,
        feature_groups, restrict_coverage, ablation_group, exp_name,
    )

    # Resolve exp_dir from the created files
    if created_files:
        first_file = Path(created_files[0])
        # artifacts are typically "cluster_lstm/exp_name/..."
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

    print(
        f"\nRun completa — {len(created_files)} artefato(s) totais "
        f"no volume '{ARTIFACT_VOLUME_NAME}':"
    )
    for f in sorted(created_files):
        print(f"  {REMOTE_ARTIFACTS_DIR}/{f}")

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts(local_dir, paths=created_files)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
