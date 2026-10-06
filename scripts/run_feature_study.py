#!/usr/bin/env python3
"""Estudo de informação de features novas (cluster 3) — CLI dos estágios.

Não há flag de cluster nem de seed: ambos são fixos por decisão
(`src/feature_study/config.py`). Em produção isto roda no Modal
(`src/modal/feature_study.py`); a CLI serve para depuração e testes.

    python scripts/run_feature_study.py prepare   --raw-dir R --shp-dir S --out-dir O
    python scripts/run_feature_study.py fit       --out-dir O --season DJF \
                                                  --arms base,full --models reference
    python scripts/run_feature_study.py aggregate --out-dir O --tags full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_study.analysis import load_arms, run_aggregate  # noqa: E402
from src.feature_study.arms import DEFAULT_ARM_SETS  # noqa: E402
from src.feature_study.config import MAIN_TAGS, MODEL_SEED, SEASONS, seed_tag  # noqa: E402
from src.feature_study.prepare import prepare  # noqa: E402
from src.feature_study.worker import run_unit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--raw-dir", default="dataset/raw")
    p.add_argument("--shp-dir", default="dataset/shp")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm-sets", default=",".join(DEFAULT_ARM_SETS))

    f = sub.add_parser("fit")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--tag", default="full", help="amostra de treino (padrão: full = completo)")
    f.add_argument("--seed", type=int, default=MODEL_SEED, help="seed do modelo (uma das MODEL_SEEDS)")
    f.add_argument("--season", required=True, choices=SEASONS)
    f.add_argument("--arms", required=True, help="nomes separados por vírgula")
    # `--models` aceita também uma lista de nomes separados por vírgula (é
    # como a triagem top-5 entra quando rodada localmente), então não pode ser
    # um `choices` fechado.
    f.add_argument("--models", default="all",
                   help="all (39) | reference (7) | fast3 (3) | lista de nomes separados por vírgula")
    f.add_argument("--no-skip-existing", action="store_true")

    a = sub.add_parser("aggregate")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--tags", default=",".join(MAIN_TAGS), help="réplicas (uma pasta por seed)")
    a.add_argument("--label", default="main")
    a.add_argument("--n-boot", type=int, default=2000)

    args = ap.parse_args()
    out = Path(args.out_dir)

    if args.stage == "prepare":
        prepare(args.raw_dir, args.shp_dir, out,
                arm_sets=tuple(s for s in args.arm_sets.split(",") if s))
    elif args.stage == "fit":
        wanted = [n for n in args.arms.split(",") if n]
        by_name = {x.name: x for x in load_arms(out / "data")}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise SystemExit(f"arms desconhecidos: {unknown}. Disponíveis: {sorted(by_name)}")
        # Lista de nomes vira `list[str]`; os modos continuam string. É o que
        # faz `select_regressors` filtrar o pool pelos modelos da triagem.
        modelos = args.models
        if modelos not in ("all", "reference", "fast3"):
            modelos = [m for m in modelos.split(",") if m]
        run_unit(out / "data", out, args.tag, args.season, [by_name[n] for n in wanted],
                 models=modelos, skip_existing=not args.no_skip_existing,
                 seed=args.seed, out_tag=seed_tag(args.seed))
    else:
        res = run_aggregate(out, out / "data", [t for t in args.tags.split(",") if t],
                            label=args.label, n_boot=args.n_boot)
        print(res["ranking_features"].to_string(index=False))


if __name__ == "__main__":
    main()
