"""
IRC Vendaval — Modal deployment (grade corrigida / corrected_grid)
======================================================================
Executa `main.py corrected_grid` no Modal (32GB RAM) em vez de local —
o processo local morreu por falta de memória (16GB, sem swap) ao recarregar
o dataset completo + reconstruir o DataFrame achatado por combinação
(pipeline, arm) vencedora, uma vez por combo (~7 combos).

Uso
---
modal run src/modal/corrected_grid.py \\
    --start-date 2020-01-08 --end-date 2024-12-31 --out-dir v1.1

# Só baixar uma run já concluída
modal run src/modal/corrected_grid.py --only-download --out-dir v1.1 \\
    --local-dir artifacts/corrected_grid/v1.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── Constantes ───────────────────────────────────────────────────────────────

APP_NAME = "irc-vendaval-corrected-grid"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

_local_root = Path(__file__).parent.parent.parent

# ── Imagem ───────────────────────────────────────────────────────────────────
# Mesmas libs de cluster_lazy.py (pandas/sklearn/xarray/...) + as libs de
# regressão usadas pelos modelos salvos (joblib.load precisa da lib original
# instalada pra desserializar CatBoost/XGBoost/LightGBM) + tensorflow (só o
# combo LSTM vencedor usa, inferência em CPU já é suficiente aqui).

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
        "scikit-learn>=1.6.0",
        "xarray>=2024.1.0",
        "netCDF4>=1.6.0",
        "geopandas>=0.14.0",
        "pyshp>=2.3.0",
        "scipy>=1.10.0",
        "shapely>=2.0.0",
        "pyarrow>=14.0.0",
        "dask>=2024.1.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "tensorflow>=2.15.0",
    )
    .env({
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        # "NetCDF: HDF error" ao abrir o MESMO .nc mais de uma vez no mesmo
        # processo (aqui: INMET_Stratified.nc via NetCDFLoader e de novo via
        # _load_direction_stations) — o locking padrão do HDF5 não convive
        # bem com o volume do Modal (FUSE, filesystem de rede).
        "HDF5_USE_FILE_LOCKING": "FALSE",
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

app = modal.App(APP_NAME)


# ── Função de execução ───────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        REMOTE_ARTIFACTS_DIR: artifact_volume,
        REMOTE_DATASET_DIR: dataset_volume,
    },
    timeout=10800,
    memory=32768,
    # sem checkpoint por combo — preempção reinicia os ~7 combos do zero.
    nonpreemptible=True,
)
def run_corrected_grid(
    start_date: str,
    end_date: str,
    out_dir: str,
    metric: str = "R2",
    interp_mode: str = "basin",
    idw_neighbors: int = 15,
    idw_power: float = 2.0,
    smoothing: str = "none",
    consolidated: bool = False,
    clusters: str | None = None,
) -> list[str]:
    """Executa main.py corrected_grid dentro do container Modal."""
    import subprocess

    raw_dir = f"{REMOTE_DATASET_DIR}/raw"
    shp_dir = f"{REMOTE_DATASET_DIR}/shp"
    artifacts_root = Path(REMOTE_ARTIFACTS_DIR)
    remote_out_dir = f"{REMOTE_ARTIFACTS_DIR}/corrected_grid/{out_dir}"

    # best_model_selector.py espera artifacts_root/{lazy_modal/lazy_clusters,
    # mlp_modal/mlp_clusters, modal/experiments} — convenção dos diretórios
    # LOCAIS pós-download (--local-dir artifacts/lazy_modal etc.), diferente
    # da estrutura crua do volume remoto (lazy_clusters/, mlp_clusters/,
    # modal/experiments direto na raiz). Symlinks fazem a ponte sem tocar em
    # best_model_selector.py (usado tanto local quanto aqui).
    (artifacts_root / "lazy_modal").mkdir(parents=True, exist_ok=True)
    (artifacts_root / "mlp_modal").mkdir(parents=True, exist_ok=True)
    lazy_link = artifacts_root / "lazy_modal" / "lazy_clusters"
    mlp_link = artifacts_root / "mlp_modal" / "mlp_clusters"
    if not lazy_link.exists():
        lazy_link.symlink_to(artifacts_root / "lazy_clusters")
    if not mlp_link.exists():
        mlp_link.symlink_to(artifacts_root / "mlp_clusters")

    print(f"[modal] Período: {start_date} .. {end_date}")
    print(f"[modal] Saída: {remote_out_dir}")

    before: set[str] = (
        {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
        if artifacts_root.exists()
        else set()
    )

    cmd = [
        sys.executable,
        f"{REMOTE_APP_DIR}/main.py",
        "corrected_grid",
        "--raw-dir", raw_dir,
        "--shp-dir", shp_dir,
        "--artifacts-root", str(artifacts_root),
        "--out-dir", remote_out_dir,
        "--start-date", start_date,
        "--end-date", end_date,
        "--metric", metric,
        "--interp-mode", interp_mode,
        "--idw-neighbors", str(idw_neighbors),
        "--idw-power", str(idw_power),
        "--smoothing", smoothing,
    ]
    if consolidated:
        cmd += ["--consolidated"]
    if clusters:
        cmd += ["--clusters", clusters]

    # Streaming linha-a-linha — combos podem levar bastante tempo (recarrega
    # dataset + reconstrói DataFrame achatado por combo); sem isso, nada
    # aparece até o processo inteiro terminar.
    proc = subprocess.Popen(
        cmd, cwd=REMOTE_APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"corrected_grid falhou (exit code {proc.returncode}).")

    artifact_volume.commit()

    after = {str(p) for p in artifacts_root.rglob("*") if p.is_file()}
    created = sorted(
        str(Path(p).relative_to(REMOTE_ARTIFACTS_DIR))
        for p in (after - before)
        if "lazy_modal" not in str(p) and "mlp_modal" not in str(p)
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
        local_path = local_root / Path(rel).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = list(vol.read_file(entry.path))
        local_path.write_bytes(b"".join(chunks))
        print(f"  ↓ {rel}")
        count += 1

    return count


# ── Entrypoint local ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    start_date: str = "2020-01-08",
    end_date: str = "2024-12-31",
    out_dir: str = "v1.1",
    metric: str = "R2",
    interp_mode: str = "basin",
    idw_neighbors: int = 15,
    idw_power: float = 2.0,
    smoothing: str = "none",
    consolidated: bool = False,
    clusters: str = "",
    download: bool = True,
    only_download: bool = False,
    local_dir: str = "",
) -> None:
    """
    Flags
    -----
    --start-date/--end-date  Período da grade (default: 2020-01-08..2024-12-31,
                             mesmo período do v1 — único intervalo com dado
                             real nas 4 fontes mescladas simultaneamente)
    --out-dir               Nome da subpasta em artifacts/corrected_grid/
                             (default: v1.1)
    --only-download         Só baixa uma run já concluída; não roda
    --local-dir             Destino local (default: artifacts/corrected_grid/<out-dir>)
    """
    local_dir = local_dir or f"artifacts/corrected_grid/{out_dir}"
    remote_subdir = f"corrected_grid/{out_dir}"

    if only_download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts_dir(local_dir, remote_subdir=remote_subdir)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
        return

    print(f"\nGerando grade corrigida '{out_dir}' no Modal "
          f"(via spawn — desacoplado da conexão local)...")
    call = run_corrected_grid.spawn(
        start_date=start_date,
        end_date=end_date,
        out_dir=out_dir,
        metric=metric,
        interp_mode=interp_mode,
        idw_neighbors=idw_neighbors,
        idw_power=idw_power,
        smoothing=smoothing,
        consolidated=consolidated,
        clusters=clusters or None,
    )
    print(f"Function call id: {call.object_id}")
    print("Se a conexão cair durante o .get(), retome com:\n"
          "  python3 -c \"import modal; "
          f"print(modal.FunctionCall.from_id('{call.object_id}').get())\"")
    created_files = call.get()

    print(f"\nRun completa — {len(created_files)} artefato(s) totais:")
    for f in sorted(created_files):
        print(f"  {REMOTE_ARTIFACTS_DIR}/{f}")

    if download:
        print(f"\nBaixando artefatos para '{local_dir}'...")
        n = _download_artifacts_dir(local_dir, remote_subdir=remote_subdir)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
