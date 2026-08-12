#!/usr/bin/env python3
"""Mostra quais experimentos da matriz de ablation já foram rodados.

Varre os diretórios locais das 3 pipelines (+ o do GAN) procurando por
results.csv/synthetic_augment.csv já baixados, e lê o run_meta.json de cada
um pra mostrar com que config (feature_groups, ablation_group, synthetic_csv,
timestamp) cada experimento foi gerado — útil para conferir antes de rodar
run_ablation_modal_all.sh de novo, e não repetir braços que já estão prontos.

Uso:
    python3 scripts/check_ablation_status.py
    python3 scripts/check_ablation_status.py --lstm-config config/experiment_cluster_lstm_modal.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._ablation_common import ABLATION_MATRIX, exp_name_for

ARM_NAMES = [c["name"] for c in ABLATION_MATRIX]


def _read_meta(exp_dir: Path) -> dict | None:
    meta_path = exp_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _status_line(label: str, exp_dir: Path) -> str:
    results_path = exp_dir / "results.csv"
    if not results_path.exists():
        return f"  {label:<28} FALTA       ({exp_dir})"

    meta = _read_meta(exp_dir)
    if meta is None:
        return f"  {label:<28} PRONTO      (sem run_meta.json — {exp_dir})"

    ts = meta.get("timestamp", "?")
    fg = meta.get("feature_groups", "?")
    synth = meta.get("synthetic_csv")
    synth_str = Path(synth).name if synth else "—"
    return f"  {label:<28} PRONTO      ts={ts}  feature_groups={fg}  synthetic={synth_str}"


def main():
    parser = argparse.ArgumentParser(description="Status da matriz de ablation (o que já foi rodado)")
    parser.add_argument("--lazy-dir", default=str(ROOT / "artifacts" / "lazy_modal" / "lazy_clusters"))
    parser.add_argument("--mlp-dir", default=str(ROOT / "artifacts" / "mlp_modal" / "mlp_clusters"))
    parser.add_argument("--lstm-config", default=None,
                        help="YAML da LSTM — usado só para resolver onde os resultados ficam localmente")
    parser.add_argument("--lstm-dir", default=None,
                        help="Alternativa: caminho local direto (se não quiser passar --lstm-config)")
    parser.add_argument("--gan-dir", default=str(ROOT / "artifacts" / "gan_modal" / "gan_clusters"))
    parser.add_argument("--exp-prefix", default="")
    args = parser.parse_args()

    print("=" * 70)
    print("  STATUS DA MATRIZ DE ABLATION")
    print("=" * 70)

    # ── GAN (dados sintéticos) ──
    gan_dir = Path(args.gan_dir)
    print("\n[GAN — dados sintéticos]")
    if gan_dir.exists() and any(gan_dir.iterdir()):
        for exp_dir in sorted(gan_dir.iterdir()):
            csv_path = exp_dir / "synthetic_augment.csv"
            if csv_path.exists():
                meta = _read_meta(exp_dir)
                ts = meta.get("timestamp", "?") if meta else "?"
                print(f"  {exp_dir.name:<28} PRONTO      ts={ts}  ({csv_path})")
    else:
        print(f"  Nenhum experimento encontrado em {gan_dir}")

    # ── LSTM: resolve diretório local a partir do --config, se dado ──
    lstm_dir = None
    if args.lstm_dir:
        lstm_dir = Path(args.lstm_dir)
    elif args.lstm_config:
        import yaml
        cfg = yaml.safe_load(open(args.lstm_config))
        remote = cfg["experiment"]["output_dir"]
        lstm_dir = Path("artifacts") / Path(remote.lstrip("/")).relative_to("artifacts")

    pipelines = {
        "LAZY": Path(args.lazy_dir),
        "MLP": Path(args.mlp_dir),
    }
    if lstm_dir:
        pipelines["LSTM"] = lstm_dir

    for name, base_dir in pipelines.items():
        print(f"\n[{name} — {base_dir}]")
        for arm in ARM_NAMES:
            exp_dir = base_dir / exp_name_for(args.exp_prefix, arm)
            print(_status_line(arm, exp_dir))

    if "LSTM" not in pipelines:
        print(
            "\n[LSTM] Não verificado — informe --lstm-config <yaml> ou --lstm-dir <caminho local> "
            "para checar."
        )

    print("\n" + "=" * 70)
    print("  'FALTA' = precisa rodar. 'PRONTO' = já existe, run_ablation_modal_all.sh")
    print("  pula automaticamente (a menos que --force seja usado).")
    print("=" * 70)


if __name__ == "__main__":
    main()
