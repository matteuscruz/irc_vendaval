#!/usr/bin/env python3
"""Sincroniza o grid corrigido (artifacts/corrected_grid/v1/) pro dashboard de
produção (repo separado irc_vendaval_dashboard) — mesma ideia de
sync_ablation_to_dashboard.py, mas sem conversão (os .nc já são compactos o
bastante e o dashboard lê via xarray/open_mfdataset diretamente).

Uso:
    python3 scripts/sync_corrected_grid_to_dashboard.py
    python3 scripts/sync_corrected_grid_to_dashboard.py --source-dir artifacts/corrected_grid/v2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", default=str(ROOT / "artifacts" / "corrected_grid" / "v1"),
    )
    parser.add_argument(
        "--dashboard-dir", default=str(ROOT.parent / "irc_vendaval_dashboard"),
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dashboard_dir = Path(args.dashboard_dir)
    if not source_dir.exists():
        parser.error(f"--source-dir não encontrado: {source_dir}")
    if not dashboard_dir.exists():
        parser.error(f"--dashboard-dir não encontrado: {dashboard_dir}")

    dest_dir = dashboard_dir / "artifacts" / "corrected_grid" / source_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src in sorted(source_dir.glob("grid_corrected_*.nc")):
        shutil.copy2(src, dest_dir / src.name)
        print(f"[sync] {src.name} -> {dest_dir}")
        n += 1

    winners_src = source_dir / "winners.csv"
    if winners_src.exists():
        shutil.copy2(winners_src, dest_dir / "winners.csv")
        print(f"[sync] winners.csv -> {dest_dir}")

    print(f"\n[sync] {n} arquivo(s) de grid sincronizado(s) para {dest_dir}")


if __name__ == "__main__":
    main()
