#!/usr/bin/env python3
"""Junta a LSTM e os modelos tabulares (árvores, lineares, MLP) numa só análise.

Lê as unidades já gravadas (`units/<tag>` dos tabulares e `units/<tag>_lstm` da LSTM) e escreve, em
`<estudo>/todos_os_modelos/`: erro por modelo × arm × trimestre (com o ERA5 ao lado), o resumo por modelo e arm,
e o efeito `full` contra `base` por modelo, com bootstrap pareado em blocos. Não retreina nada.

    python scripts/analise_todos_modelos.py                       # estudo do Modal, base e full
    python scripts/analise_todos_modelos.py --study cluster3_groups --lstm-tags full_lstm --tree-tags full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.feature_study.diagnostics import all_models as am  # noqa: E402

TREE_TAGS = "full,full_s43,full_s44,full_s45,full_s46"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="cluster3_groups_modal")
    ap.add_argument("--tree-tags", default=TREE_TAGS, help="uma pasta por seed dos modelos tabulares")
    ap.add_argument("--lstm-tags", default="full_lstm", help="uma pasta por seed da LSTM")
    ap.add_argument("--arms", default="base,full")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    r = am.run(RAIZ / "artifacts/feature_study" / a.study, [t for t in a.tree_tags.split(",") if t],
               [t for t in a.lstm_tags.split(",") if t], n_boot=a.n_boot, arms=tuple(a.arms.split(",")))
    print(f"escrito em {r['out']}\n")
    print("== RMSE médio (trimestres em que o modelo existe) ==")
    print(r["resumo"].pivot_table(index=["model", "familia"], columns="arm", values="RMSE").round(3).to_string())
    print("\n== efeito full contra base, por modelo (positivo = o full é melhor) ==")
    print(r["efeitos"][["model", "familia", "efeito_full_vs_base", "ci_lo", "ci_hi", "veredito"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
