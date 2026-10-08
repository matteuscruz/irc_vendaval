"""Copia `resultados_lfs/feature_study/*` (baixado com `git lfs pull`) para `artifacts/feature_study/`,
onde os notebooks leem. Não sobrescreve arquivo existente, a menos que `--force`."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEM, DESTINO = RAIZ / "resultados_lfs/feature_study", RAIZ / "artifacts/feature_study"


def restaurar(force: bool = False) -> int:
    n = 0
    for f in ORIGEM.rglob("*"):
        if f.is_file():
            d = DESTINO / f.relative_to(ORIGEM)
            if d.exists() and not force:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d)
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    print(f"{restaurar(ap.parse_args().force)} arquivos copiados para {DESTINO}")
