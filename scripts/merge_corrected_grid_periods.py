"""Consolida os NetCDFs por-ano de `corrected_grid` de dois (ou mais)
diretórios de saída num único diretório final — Fase 5 do plano de correção
ERA5 2000-2024.

Não recalcula nada: `grid_generator.py` já escreve um `grid_corrected_<ano>.nc`
por ano, então "juntar" é organizacional — copia os arquivos de cada
diretório de origem pro destino (por padrão como symlink, mais barato que
copiar ~29MB por ano; use --copy se preferir arquivos reais) e concatena os
`winners.csv` de cada origem num `winners_full.csv` único, marcado com uma
coluna `period` (o nome do diretório de origem) pra rastreabilidade.

Uso
---
python3 scripts/merge_corrected_grid_periods.py \\
    --sources artifacts/corrected_grid/v1.1_2000_2019 artifacts/corrected_grid/v1.1 \\
    --dest artifacts/corrected_grid/final_2000_2024
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge(sources: list[Path], dest: Path, copy: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    winner_frames = []
    n_years = 0
    for src in sources:
        if not src.is_dir():
            raise FileNotFoundError(f"Diretório de origem não existe: {src}")

        year_files = sorted(src.glob("grid_corrected_*.nc"))
        if not year_files:
            print(f"[AVISO] Nenhum grid_corrected_*.nc em {src} — pulando.")
        for f in year_files:
            target = dest / f.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"{target} já existe — anos sobrepostos entre origens? "
                    "Resolva manualmente antes de re-rodar o merge."
                )
            if copy:
                shutil.copy2(f, target)
            else:
                target.symlink_to(f.resolve())
            n_years += 1
        print(f"  {src}: {len(year_files)} ano(s) → {dest}")

        winners_csv = src / "winners.csv"
        if winners_csv.exists():
            df = pd.read_csv(winners_csv)
            df["period"] = src.name
            winner_frames.append(df)
        else:
            print(f"  [AVISO] Sem winners.csv em {src} — não entra no winners_full.csv")

    if winner_frames:
        combined = pd.concat(winner_frames, ignore_index=True)
        combined.to_csv(dest / "winners_full.csv", index=False)
        print(f"  winners_full.csv: {len(combined)} linha(s) ({len(winner_frames)} origem(ns))")

    print(f"\nOK — {n_years} arquivo(s) de ano consolidado(s) em {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, type=Path,
                         help="Diretórios de origem (ex.: v1.1_2000_2019 v1.1), em qualquer ordem")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--copy", action="store_true",
                         help="Copia os arquivos em vez de symlink (mais espaço em disco)")
    args = parser.parse_args()
    merge(args.sources, args.dest, args.copy)


if __name__ == "__main__":
    main()
