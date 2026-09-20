"""Modal: relatório de XAI (SHAP + ablação) sobre um experimento já treinado.

Separado das pipelines de treino de propósito: não treina, não escreve modelo
e não altera dado. Lê os artefatos de um experimento no volume, reconstrói a
matriz de avaliação (é por isso que precisa rodar aqui — a reconstrução exige
o merge completo do ERA5, inviável na máquina local) e grava em
`<exp>/xai/` no volume, baixando ao final.

`--exp-dir` é relativo à RAIZ DO VOLUME, não ao diretório local: no volume os
experimentos do lazy ficam em `lazy_clusters/<exp>`, enquanto localmente eles
são espelhados em `artifacts/lazy_modal/lazy_clusters/<exp>` (o `lazy_modal/`
é criado pelo download, não existe no volume).

Uso:
    modal run src/modal/xai.py --exp-dir lazy_clusters/lazy_c3_static_mb
    modal run src/modal/xai.py --exp-dir lazy_clusters/<exp> --clusters 3 --strategy shuffle
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "irc-vendaval-xai"
ARTIFACT_VOLUME_NAME = "irc-vendaval-artifacts"
DATASET_VOLUME_NAME = "irc-vendaval-dataset"
REMOTE_APP_DIR = "/app"
REMOTE_ARTIFACTS_DIR = "/artifacts"
REMOTE_DATASET_DIR = "/dataset"

_local_root = Path(__file__).parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libhdf5-dev", "libnetcdf-dev", "gdal-bin", "libgdal-dev", "libproj-dev")
    .pip_install(
        "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0", "xarray>=2024.1.0", "netCDF4>=1.6.0",
        "geopandas>=0.14.0", "pyshp>=2.3.0", "scipy>=1.10.0", "shapely>=2.0.0",
    )
    # Os artefatos do lazy são Pipelines do LazyPredict com estimadores
    # CatBoost/LGBM/XGB: sem estes pacotes o joblib.load falha ao desserializar.
    .pip_install("lazypredict[boost]>=0.3.0", "xgboost>=2.0.0", "lightgbm>=4.0.0", "shap>=0.46.0")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(_local_root / "main.py"), remote_path=f"{REMOTE_APP_DIR}/main.py", copy=True)
    .add_local_dir(str(_local_root / "src"), remote_path=f"{REMOTE_APP_DIR}/src", copy=True)
    .add_local_dir(str(_local_root / "scripts"), remote_path=f"{REMOTE_APP_DIR}/scripts", copy=True)
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME)
_VOLUMES = {REMOTE_ARTIFACTS_DIR: artifact_volume, REMOTE_DATASET_DIR: dataset_volume}


@app.function(image=image, volumes=_VOLUMES, timeout=10800, memory=32768)
def explain(
    exp_dir: str,
    clusters: str | None = None,
    seasons: str | None = None,
    split: str = "test",
    strategy: str = "mean",
    n_repeats: int = 5,
    max_shap_samples: int = 2000,
    skip_shap: bool = False,
    skip_ablation: bool = False,
) -> list[str]:
    """Roda o relatório e devolve os caminhos criados (relativos ao volume)."""
    import subprocess

    remote_exp = f"{REMOTE_ARTIFACTS_DIR}/{exp_dir.lstrip('/')}"
    if not Path(remote_exp).is_dir():
        # Confusão natural: o caminho LOCAL carrega o prefixo do espelho de
        # download (ex. `lazy_modal/`), que não existe no volume. Em vez de
        # exigir a tradução, resolve pelo nome do experimento quando ele é
        # inequívoco — e só falha se for ambíguo ou inexistente.
        disponiveis = sorted(
            str(d.relative_to(REMOTE_ARTIFACTS_DIR))
            for d in Path(REMOTE_ARTIFACTS_DIR).glob("*/*")
            if (d / "fitted_models").is_dir()
        )
        alvo = exp_dir.strip("/").split("/")[-1]
        candidatos = [d for d in disponiveis if d.split("/")[-1] == alvo]
        if len(candidatos) == 1:
            print(f"[xai] '{exp_dir}' não existe no volume; usando '{candidatos[0]}' "
                  "(mesmo nome de experimento).")
            remote_exp = f"{REMOTE_ARTIFACTS_DIR}/{candidatos[0]}"
        else:
            detalhe = (f"'{alvo}' é ambíguo: {candidatos}" if candidatos
                       else f"experimentos com modelos salvos: {disponiveis or '(nenhum)'}")
            raise FileNotFoundError(
                f"{remote_exp} não existe no volume '{ARTIFACT_VOLUME_NAME}'.\n"
                f"--exp-dir é relativo à raiz do volume (sem o prefixo do espelho local).\n"
                f"{detalhe}"
            )

    cmd = [
        sys.executable, "-u", f"{REMOTE_APP_DIR}/scripts/run_xai.py",
        "--exp-dir", remote_exp,
        "--raw-dir", f"{REMOTE_DATASET_DIR}/raw",
        "--shp-dir", f"{REMOTE_DATASET_DIR}/shp",
        "--split", split, "--strategy", strategy,
        "--n-repeats", str(n_repeats), "--max-shap-samples", str(max_shap_samples),
    ]
    if clusters:
        cmd += ["--clusters", clusters]
    if seasons:
        cmd += ["--seasons", seasons]
    if skip_shap:
        cmd += ["--skip-shap"]
    if skip_ablation:
        cmd += ["--skip-ablation"]

    # stderr capturado (e reemitido) para que a exceção carregue a CAUSA: sem
    # isso o erro que chega ao terminal local é só "exit 1", e o traceback real
    # fica perdido no meio do log do container.
    proc = subprocess.run(cmd, cwd=REMOTE_APP_DIR, text=True, stderr=subprocess.PIPE)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        cauda = "\n".join(proc.stderr.strip().splitlines()[-25:]) if proc.stderr else "(sem stderr)"
        raise RuntimeError(f"run_xai.py falhou (exit {proc.returncode}):\n{cauda}")

    artifact_volume.commit()
    out = Path(remote_exp) / "xai"
    return sorted(
        str(p.relative_to(REMOTE_ARTIFACTS_DIR)) for p in out.rglob("*") if p.is_file()
    )


def _download(paths: list[str], local_dir: str) -> int:
    vol = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)
    root = Path(local_dir)
    count = 0
    for rel in paths:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"".join(vol.read_file(rel)))
        print(f"  ↓ {rel}")
        count += 1
    return count


@app.local_entrypoint()
def main(
    exp_dir: str,
    clusters: str = "",
    seasons: str = "",
    split: str = "test",
    strategy: str = "mean",
    n_repeats: int = 5,
    max_shap_samples: int = 2000,
    skip_shap: bool = False,
    skip_ablation: bool = False,
    download: bool = True,
    local_dir: str = "artifacts/lazy_modal",
) -> None:
    """
    Flags
    -----
    --exp-dir            Experimento RELATIVO À RAIZ DO VOLUME, ex.:
                         "lazy_clusters/lazy_c3_static_mb" (sem o prefixo
                         "lazy_modal/", que só existe no espelho local)
    --clusters           Subconjunto, ex.: "3" ou "3,4" (default: todos)
    --seasons            Ex.: "ALL,DJF" — ALL é o modelo agrupado
    --split              train | val | test (default: test)
    --strategy           Ablação: mean (indisponibilidade) | shuffle (permutação)
    --n-repeats          Repetições do shuffle (default: 5)
    --max-shap-samples   Teto de linhas no SHAP (default: 2000)
    --skip-shap          Só ablação
    --skip-ablation      Só SHAP
    --no-download        Não baixa os resultados
    --local-dir          Espelho local da raiz do volume (default:
                         artifacts/lazy_modal, mesma convenção do wrapper do
                         lazy — assim o xai/ cai dentro do experimento já
                         baixado). Use outro destino para mlp/lstm.
    """
    print(f"Explicando {exp_dir} (split={split}, ablação={strategy})...")
    created = explain.remote(
        exp_dir=exp_dir,
        clusters=clusters or None,
        seasons=seasons or None,
        split=split,
        strategy=strategy,
        n_repeats=n_repeats,
        max_shap_samples=max_shap_samples,
        skip_shap=skip_shap,
        skip_ablation=skip_ablation,
    )
    print(f"\n{len(created)} arquivo(s) gerados no volume.")
    if download and created:
        n = _download(created, local_dir)
        print(f"\n{n} arquivo(s) → {Path(local_dir).resolve()}")
