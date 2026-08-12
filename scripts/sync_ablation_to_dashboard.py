#!/usr/bin/env python3
"""Sincroniza os resultados da matriz de ablation pro dashboard de produção
(repo separado irc_vendaval_dashboard), convertendo results.csv/predictions.csv
pra Parquet (mais leve e rápido de carregar no Streamlit).

Também copia (sem conversão) os arquivos de suporte legados que as abas MLP
Explorer / LazyPredict Screening do dashboard já sabem ler (mlp_cluster_results.csv,
feature_importance.csv, stations_metadata.csv, predictions_by_station.csv,
lazy_cluster_results.csv) — permite que essas abas usem os braços da matriz de
ablation como sua fonte de experimento, no lugar dos antigos exp1-3/exp1-5.

Chamado logo após cada pipeline (lazy/mlp/lstm) terminar de agregar seus 4
braços — não espera a matriz inteira terminar (run_ablation_modal_all.sh já
chama isso automaticamente após cada agregação).

Uso:
    python3 scripts/sync_ablation_to_dashboard.py --pipeline mlp \\
        --source-dir artifacts/mlp_modal/mlp_clusters
    python3 scripts/sync_ablation_to_dashboard.py --all \\
        --lazy-dir artifacts/lazy_modal/lazy_clusters \\
        --mlp-dir artifacts/mlp_modal/mlp_clusters \\
        --lstm-dir artifacts/modal/experiments
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts._ablation_common import ABLATION_MATRIX

ARM_NAMES = [c["name"] for c in ABLATION_MATRIX]


# Arquivos legados (sem conversão) que o dashboard já sabe ler diretamente,
# por pipeline: (caminho relativo ao dir do experimento, nome no destino).
_LEGACY_FILES = {
    "mlp": [
        ("_partial/csv/mlp_cluster_results.csv", "mlp_cluster_results.csv"),
        ("_partial/csv/feature_importance.csv", "feature_importance.csv"),
        ("_partial/csv/stations_metadata.csv", "stations_metadata.csv"),
        ("predictions/predictions_by_station.csv", "predictions_by_station.csv"),
    ],
    "lazy": [
        ("lazy_cluster_results.csv", "lazy_cluster_results.csv"),
        ("predictions/predictions_by_station.csv", "predictions_by_station.csv"),
    ],
    "lstm": [
        ("predictions/predictions_by_station.csv", "predictions_by_station.csv"),
    ],
}


def _sync_arm(source_dir: Path, pipeline: str, arm: str, dashboard_dir: Path) -> bool:
    exp_dir = source_dir / arm
    results_csv = exp_dir / "results.csv"
    if not results_csv.exists():
        print(f"[sync] AVISO: {pipeline}/{arm} — results.csv não encontrado em {results_csv}, pulando.")
        return False

    dest_dir = dashboard_dir / "artifacts" / "ablation" / pipeline / arm
    dest_dir.mkdir(parents=True, exist_ok=True)

    pd.read_csv(results_csv).to_parquet(dest_dir / "results.parquet", index=False)

    predictions_csv = exp_dir / "predictions" / "predictions.csv"
    if predictions_csv.exists():
        pd.read_csv(predictions_csv).to_parquet(dest_dir / "predictions.parquet", index=False)

    meta_src = exp_dir / "run_meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, dest_dir / "run_meta.json")

    plots_src = exp_dir / "plots"
    if plots_src.exists():
        plots_dst = dest_dir / "plots"
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_src, plots_dst)

    # histories.json (LSTM): {cluster_id: {season: {loss: [...], val_loss: [...]}}}
    # — curva de convergência de treino/validação, usada pelo diagnóstico do LSTM.
    histories_src = exp_dir / "histories.json"
    if histories_src.exists():
        shutil.copy2(histories_src, dest_dir / "histories.json")

    for rel_src, name_dst in _LEGACY_FILES.get(pipeline, []):
        legacy_src = exp_dir / rel_src
        if legacy_src.exists():
            shutil.copy2(legacy_src, dest_dir / name_dst)

    print(f"[sync] {pipeline}/{arm} -> {dest_dir}")
    return True


def sync_pipeline(pipeline: str, source_dir: Path, dashboard_dir: Path) -> int:
    n = 0
    for arm in ARM_NAMES:
        if _sync_arm(source_dir, pipeline, arm, dashboard_dir):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["lazy", "mlp", "lstm"], default=None,
                        help="Sincroniza só essa pipeline (usa --source-dir)")
    parser.add_argument("--source-dir", default=None,
                        help="Diretório com os 4 braços (ex: artifacts/mlp_modal/mlp_clusters) — obrigatório com --pipeline")
    parser.add_argument("--all", action="store_true",
                        help="Sincroniza as 3 pipelines de uma vez (usa --lazy-dir/--mlp-dir/--lstm-dir)")
    parser.add_argument("--lazy-dir", default=str(ROOT / "artifacts" / "lazy_modal" / "lazy_clusters"))
    parser.add_argument("--mlp-dir", default=str(ROOT / "artifacts" / "mlp_modal" / "mlp_clusters"))
    parser.add_argument("--lstm-dir", default=str(ROOT / "artifacts" / "modal" / "experiments"))
    parser.add_argument("--dashboard-dir", default=str(ROOT.parent / "irc_vendaval_dashboard"))
    args = parser.parse_args()

    dashboard_dir = Path(args.dashboard_dir)
    if not dashboard_dir.exists():
        parser.error(f"--dashboard-dir não encontrado: {dashboard_dir}")

    total = 0
    if args.all:
        total += sync_pipeline("lazy", Path(args.lazy_dir), dashboard_dir)
        total += sync_pipeline("mlp", Path(args.mlp_dir), dashboard_dir)
        total += sync_pipeline("lstm", Path(args.lstm_dir), dashboard_dir)
    elif args.pipeline:
        if not args.source_dir:
            parser.error("--source-dir é obrigatório junto com --pipeline")
        total += sync_pipeline(args.pipeline, Path(args.source_dir), dashboard_dir)
    else:
        parser.error("Informe --pipeline <nome> --source-dir <dir> ou --all")

    print(f"\n[sync] {total} braço(s) sincronizado(s) para {dashboard_dir}/artifacts/ablation/")


if __name__ == "__main__":
    main()
