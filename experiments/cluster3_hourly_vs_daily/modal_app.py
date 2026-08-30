"""Modal wrapper isolado do experimento Cluster 3 diário-vs-horário.

Não modifica `src/modal/cluster_lstm.py` — só importa `image`/volumes de lá
(reaproveitando a imagem/GPU/volumes já configurados em produção) e adiciona
o diretório `experiments/` numa imagem DERIVADA própria deste experimento.

Uso
---
    # primeira vez: sincroniza o arquivo horário novo pro volume Modal
    # (upload direcionado, não resincroniza os 7,2 GB de dataset/raw inteiro)
    modal run experiments/cluster3_hourly_vs_daily/modal_app.py \\
        --sync-hourly-file --only daily

    # depois, roda os dois braços com 100% dos dados
    modal run experiments/cluster3_hourly_vs_daily/modal_app.py --only both

    # validação rápida/barata antes do run de verdade
    modal run experiments/cluster3_hourly_vs_daily/modal_app.py --only both --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

# Modal reimporta este arquivo dentro do container (a partir de /root/) pra
# localizar a função antes de rodar o corpo de qualquer @app.function — o
# `sys.path.insert` que estava só dentro de train_daily_arm_remote/
# train_hourly_arm_remote roda tarde demais pra este import no nível do
# módulo. Localmente é inofensivo (/app não existe, só é ignorado).
sys.path.insert(0, "/app")

from src.modal.cluster_lstm import (
    ARTIFACT_VOLUME_NAME,
    REMOTE_ARTIFACTS_DIR,
    REMOTE_DATASET_DIR,
    _download_artifacts,
    _ensure_dataset,
    artifact_volume,
    dataset_volume,
)
from src.modal.cluster_lstm import image as _base_image

REMOTE_APP_DIR = "/app"
HOURLY_FILENAME = "test_cluster_3_hourly.nc"

_local_root = Path(__file__).parent.parent.parent

# Imagem derivada (não muta _base_image) — acrescenta experiments/ e
# lazypredict (_base_image não traz, só a imagem CPU-only de
# src/modal/cluster_lazy.py tem isso; mais simples reaproveitar uma imagem
# só pros 4 braços do experimento do que manter duas).
image = _base_image.pip_install("lazypredict[boost]>=0.3.0").add_local_dir(
    str(_local_root / "experiments"),
    remote_path=f"{REMOTE_APP_DIR}/experiments",
    copy=True,
)

app = modal.App("irc-vendaval-cluster3-hourly-experiment")


def _sync_hourly_file() -> None:
    """Upload direcionado só do arquivo horário novo. `_ensure_dataset` de
    produção não o inclui (não está em DATASET_SENTINELS) e forçar um
    re-upload completo resincronizaria ~7,2 GB à toa por causa de um arquivo
    de 221 MB."""
    local_path = _local_root / "dataset" / "raw" / HOURLY_FILENAME
    if not local_path.exists():
        raise FileNotFoundError(f"Não encontrado: {local_path}")
    size_mb = local_path.stat().st_size / 1024**2
    print(f"Enviando {HOURLY_FILENAME} ({size_mb:.0f} MB) para o volume...")
    with dataset_volume.batch_upload(force=True) as upload:
        upload.put_file(str(local_path), f"/raw/{HOURLY_FILENAME}")
    print("Upload concluído.")


def _artifacts_created(output_dir: str) -> list[str]:
    return sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR))
        for p in Path(output_dir).rglob("*")
        if p.is_file()
    )


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
    timeout=600,
    memory=8192,
)
def diag_time_ranges() -> None:
    """Diagnóstico barato (CPU, sem GPU) — confirma se o volume Modal tem a
    mesma cobertura temporal (2008-2018) nas 4 fontes que load_extended()
    mescla, antes de gastar GPU tentando treinar o braço diário."""
    import sys
    sys.path.insert(0, REMOTE_APP_DIR)
    import numpy as np
    import pandas as pd
    import xarray as xr
    from src.data.netcdf_loader import NetCDFLoader

    raw = f"{REMOTE_DATASET_DIR}/raw"

    def rng(times):
        if len(times) == 0:
            return "VAZIO"
        return f"{str(np.min(times))[:10]} .. {str(np.max(times))[:10]} (n={len(times)})"

    print("=== INMET ===", flush=True)
    ds_inmet = xr.open_dataset(f"{raw}/INMET_Stratified.nc")
    print("inmet.time:", rng(ds_inmet.time.values), flush=True)

    print("=== ERA5 base ===", flush=True)
    _, ds_era5_base = NetCDFLoader(raw).load()
    print("era5_base.time:", rng(ds_era5_base.time.values), flush=True)

    print("=== ERA5-18UTC ===", flush=True)
    try:
        from src.data.era5_18utc_loader import ERA518UTCLoader
        ds_18 = ERA518UTCLoader(raw).load(ds_inmet)
        ds_18f = ds_18.assign_coords(time=pd.DatetimeIndex(ds_18.time.values).floor("D"))
        common = np.intersect1d(ds_era5_base.time.values, ds_18f.time.values)
        print("18z.time:", rng(ds_18f.time.values), " common(base,18z):", rng(common), flush=True)
    except Exception as e:
        print("18z falhou:", repr(e), flush=True)

    print("=== BT55 ===", flush=True)
    try:
        from src.data.bt55_loader import BT55Loader
        ds_bt55 = BT55Loader(raw).load(ds_inmet)
        common = np.intersect1d(ds_era5_base.time.values, ds_bt55.time.values)
        print("bt55.time:", rng(ds_bt55.time.values), " common(base,bt55):", rng(common), flush=True)
    except Exception as e:
        print("bt55 falhou:", repr(e), flush=True)

    print("=== ERA5-Basin (loader completo, com interpolação) ===", flush=True)
    try:
        from src.data.era5_basin_loader import ERA5BasinLoader
        ds_basin = ERA5BasinLoader(raw).load(ds_inmet)
        common = np.intersect1d(ds_era5_base.time.values, ds_basin.time.values)
        print("basin.time:", rng(ds_basin.time.values), " common(base,basin):", rng(common), flush=True)
    except Exception as e:
        print("basin falhou:", repr(e), flush=True)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
    gpu="A10G",
    timeout=21600,
    memory=32768,
)
def train_daily_arm_remote(smoke: bool = False) -> list[str]:
    import sys
    sys.path.insert(0, REMOTE_APP_DIR)
    from experiments.cluster3_hourly_vs_daily.run_experiment import run_daily_arm

    artifact_volume.reload()
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/experiments/cluster3_hourly_vs_daily/daily"
    run_daily_arm(
        output_dir, smoke=smoke,
        raw_dir=f"{REMOTE_DATASET_DIR}/raw", shp_dir=f"{REMOTE_DATASET_DIR}/shp",
    )
    artifact_volume.commit()
    return _artifacts_created(output_dir)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
    gpu="A10G",
    timeout=21600,
    memory=32768,
)
def train_hourly_arm_remote(smoke: bool = False) -> list[str]:
    import sys
    sys.path.insert(0, REMOTE_APP_DIR)
    from experiments.cluster3_hourly_vs_daily.run_experiment import run_hourly_arm

    artifact_volume.reload()
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/experiments/cluster3_hourly_vs_daily/hourly"
    run_hourly_arm(
        output_dir, smoke=smoke,
        raw_dir=f"{REMOTE_DATASET_DIR}/raw", shp_dir=f"{REMOTE_DATASET_DIR}/shp",
    )
    artifact_volume.commit()
    return _artifacts_created(output_dir)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
    timeout=21600,
    memory=32768,
)
def train_daily_lazy_arm_remote(smoke: bool = False) -> list[str]:
    import sys
    sys.path.insert(0, REMOTE_APP_DIR)
    from experiments.cluster3_hourly_vs_daily.run_experiment import run_daily_lazy_arm

    artifact_volume.reload()
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/experiments/cluster3_hourly_vs_daily/daily_lazy"
    run_daily_lazy_arm(
        output_dir, smoke=smoke,
        raw_dir=f"{REMOTE_DATASET_DIR}/raw", shp_dir=f"{REMOTE_DATASET_DIR}/shp",
    )
    artifact_volume.commit()
    return _artifacts_created(output_dir)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume},
    timeout=3600,
    memory=8192,
)
def train_hourly_lazy_arm_remote(smoke: bool = False) -> list[str]:
    import sys
    sys.path.insert(0, REMOTE_APP_DIR)
    from experiments.cluster3_hourly_vs_daily.run_experiment import run_hourly_lazy_arm

    artifact_volume.reload()
    output_dir = f"{REMOTE_ARTIFACTS_DIR}/experiments/cluster3_hourly_vs_daily/hourly_lazy"
    run_hourly_lazy_arm(
        output_dir, smoke=smoke,
        raw_dir=f"{REMOTE_DATASET_DIR}/raw", shp_dir=f"{REMOTE_DATASET_DIR}/shp",
    )
    artifact_volume.commit()
    return _artifacts_created(output_dir)


@app.local_entrypoint()
def main(
    only: str = "both",
    sync_hourly_file: bool = False,
    smoke: bool = False,
    download: bool = True,
    local_dir: str = "artifacts/experiments/cluster3_hourly_vs_daily",
    diag: bool = False,
    force_dataset_upload: bool = False,
) -> None:
    """
    Flags
    -----
    --only               daily | hourly | daily_lazy | hourly_lazy | both | all
                          (default: both — só os 2 braços LSTM)
    --sync-hourly-file    Envia dataset/raw/test_cluster_3_hourly.nc pro
                          volume Modal (upload direcionado só desse
                          arquivo) — necessário rodar uma vez antes do
                          braço horário
    --smoke               Config minúscula p/ validar rápido/barato antes
                          do run de verdade (100% dos dados)
    --no-download          Roda mas não baixa os artefatos ao final
    --local-dir            Destino local (default:
                          artifacts/experiments/cluster3_hourly_vs_daily)
    --diag                 Só roda o diagnóstico de cobertura temporal
                          (CPU, barato) e sai — não treina nada
    --force-dataset-upload Re-sincroniza dataset/raw+shp inteiro pro volume
                          (~7,2GB) via _ensure_dataset(force=True) — usar
                          quando o volume estiver desatualizado (confirmado
                          via --diag)
    """
    if force_dataset_upload:
        _ensure_dataset(force=True)

    if diag:
        diag_time_ranges.remote()
        return

    if sync_hourly_file:
        _sync_hourly_file()

    created: list[str] = []
    if only in ("daily", "both", "all"):
        print("\nSubmetendo braço diário LSTM (Cluster 3, arm basin)...")
        created += train_daily_arm_remote.remote(smoke=smoke)
    if only in ("hourly", "both", "all"):
        print("\nSubmetendo braço horário LSTM (Cluster 3, dataset novo)...")
        created += train_hourly_arm_remote.remote(smoke=smoke)
    if only in ("daily_lazy", "all"):
        print("\nSubmetendo braço diário LazyPredict (Cluster 3, arm basin)...")
        created += train_daily_lazy_arm_remote.remote(smoke=smoke)
    if only in ("hourly_lazy", "all"):
        print("\nSubmetendo braço horário LazyPredict (Cluster 3, dataset novo)...")
        created += train_hourly_lazy_arm_remote.remote(smoke=smoke)

    print(f"\n{len(created)} artefato(s) no volume '{ARTIFACT_VOLUME_NAME}':")
    for f in created:
        print(f"  {REMOTE_ARTIFACTS_DIR}/{f}")

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts(local_dir, paths=created)
        print(f"\n{n} arquivo(s) -> {Path(local_dir).resolve()}")
