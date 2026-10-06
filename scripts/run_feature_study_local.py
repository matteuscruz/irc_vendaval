"""Executa o estudo de features (estrutura nova, por grupo) NA MÁQUINA LOCAL.

É o mesmo caminho do Modal — `prepare`, `fit`, `screen`, `aggregate` —, só que
sequencial e com cuidado com os recursos: a máquina tem ~4 GB de RAM livres e já
travou em testes pesados, então aqui há teto de threads, prioridade baixa e uma
guarda de memória que espera (ou aborta) em vez de deixar o sistema congelar.

Tudo é IDEMPOTENTE: uma unidade já pronta é pulada, então Ctrl-C, queda de luz
ou falta de memória não custam nada — rode o mesmo comando de novo.

  python scripts/run_feature_study_local.py plan
  python scripts/run_feature_study_local.py run --stage prepare
  python scripts/run_feature_study_local.py run --stage triage --seeds 42
  python scripts/run_feature_study_local.py run --stage all --seeds 42
  python scripts/run_feature_study_local.py status

Ordem dos estágios (cada um é um portão para o seguinte):
  prepare    monta a população (segundos)
  triage     os 39 modelos só no arm `base`, por trimestre e seed
  screen     congela os 5 melhores por trimestre em summary/top_models.json
  study      os 5 modelos em TODOS os arms
  aggregate  efeitos pareados, bootstrap em blocos, ranking
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

STAGES = ("prepare", "triage", "screen", "study", "aggregate")
ARM_SETS = ("anchors", "groups", "controls", "era5_groups")
SEASONS = ("DJF", "MAM", "JJA", "SON")
DEFAULT_OUT = "artifacts/feature_study/cluster3_groups"


def _limita_threads(n: int) -> None:
    """Antes de importar numpy/lightgbm/catboost: depois, o teto não pega.
    CatBoost e XGBoost ignoram o cgroup e abririam uma thread por CPU."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(n))
    try:
        os.nice(10)        # o VS Code e o resto da máquina continuam responsivos
    except (OSError, AttributeError):
        pass


def _espera_memoria(minimo_gb: float, tentativas: int = 20) -> None:
    import psutil

    for _ in range(tentativas):
        livre = psutil.virtual_memory().available / 1e9
        if livre >= minimo_gb:
            return
        print(f"   [memória] só {livre:.1f} GB livres (mínimo {minimo_gb}) — aguardando 30 s…", flush=True)
        time.sleep(30)
    raise MemoryError(
        f"menos de {minimo_gb} GB livres por {tentativas * 30 // 60} min. Feche programas "
        "(o VS Code é o maior) e rode o mesmo comando — o que já ficou pronto é pulado."
    )


def _formata(seg: float) -> str:
    seg = int(seg)
    return f"{seg // 3600}h{seg % 3600 // 60:02d}" if seg >= 3600 else f"{seg // 60}min{seg % 60:02d}s"


class Plano:
    def __init__(self, args):
        from src.feature_study.core.config import MODEL_SEEDS, seed_tag, triage_tag
        self.out = RAIZ / args.out_dir
        self.data = self.out / "data"
        self.seeds = [int(s) for s in args.seeds.split(",") if s] or [MODEL_SEEDS[0]]
        self.tags = [seed_tag(s) for s in self.seeds]                # estudo
        self.triage_tags = [triage_tag(s) for s in self.seeds]       # triagem, pasta própria
        self.triage_tag = triage_tag
        self.seasons = tuple(x for x in args.seasons.split(",") if x) or SEASONS
        invalidos = set(self.seasons) - set(SEASONS)
        if invalidos:
            raise SystemExit(f"trimestres inválidos: {sorted(invalidos)} (válidos: {list(SEASONS)})")
        self.args = args
        self.seed_tag = seed_tag

    def arms(self):
        from src.feature_study.core.analysis import load_arms
        return load_arms(self.data)

    def top5(self):
        import json
        caminho = self.out / "summary" / "top_models.json"
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não existe — rode `--stage screen` (depois do triage)")
        return json.loads(caminho.read_text())["by_season"]


def _roda_unidades(plano: Plano, unidades, rotulo: str) -> None:
    """`unidades`: lista de (descricao, callable). Sequencial, com ETA e guarda de memória."""
    total, t0, feitos = len(unidades), time.time(), 0
    for i, (desc, fn) in enumerate(unidades, start=1):
        _espera_memoria(plano.args.min_free_gb)
        ti = time.time()
        print(f"[{rotulo} {i}/{total}] {desc}", flush=True)
        fn()
        feitos += 1
        dt = time.time() - ti
        medio = (time.time() - t0) / feitos
        print(f"   ✓ {_formata(dt)} | decorrido {_formata(time.time() - t0)} | "
              f"faltam ≈ {_formata(medio * (total - i))}", flush=True)


def estagio_prepare(plano: Plano) -> None:
    from src.feature_study.core.prepare import prepare

    if (plano.data / "meta.json").exists() and not plano.args.force:
        print(f"[prepare] já existe em {plano.data} (use --force para refazer)")
        return
    prepare(str(RAIZ / plano.args.raw_dir), plano.out, arm_sets=ARM_SETS)


def estagio_triage(plano: Plano) -> None:
    from src.feature_study.core.worker import run_unit

    arms = [a for a in plano.arms() if a.name == "base"]
    unidades = [
        (f"{s} seed {seed} — base, 39 modelos",
         lambda s=s, seed=seed: run_unit(plano.data, plano.out, "full", s, arms, models="all",
                                         seed=seed, out_tag=plano.triage_tag(seed)))
        for seed in plano.seeds for s in plano.seasons
    ]
    _roda_unidades(plano, unidades, "triage")


def estagio_screen(plano: Plano) -> None:
    import json

    from src.feature_study.core.analysis import top_models_by_season, top_models_payload

    top = top_models_by_season(plano.out, plano.triage_tags, arm="base", k=plano.args.k)
    if top.empty:
        raise ValueError("nenhuma métrica do arm base — rode `--stage triage` antes")
    destino = plano.out / "summary"
    destino.mkdir(parents=True, exist_ok=True)
    top.to_parquet(destino / "top_models.parquet", index=False)
    payload = top_models_payload(top, arm="base", k=plano.args.k, tags=plano.triage_tags)
    (destino / "top_models.json").write_text(json.dumps(payload, indent=2))
    for s, modelos in payload["by_season"].items():
        print(f"[screen] {s}: {', '.join(modelos)}")
    instavel = top[top["sd_RMSE_seeds"] > 0.05]
    if len(plano.triage_tags) < 2:
        print("[screen] AVISO: com uma só seed não há como medir a estabilidade da escolha")
    elif not instavel.empty:
        print(f"[screen] AVISO: {len(instavel)} escolha(s) com sd do RMSE entre seeds > 0.05")


def estagio_study(plano: Plano) -> None:
    from src.feature_study.core.worker import run_unit

    por_trimestre = plano.top5()
    arms = plano.arms()
    unidades = []
    for seed in plano.seeds:
        for s in plano.seasons:
            modelos = por_trimestre[s]
            unidades.append((
                f"{s} seed {seed} — {len(arms)} arms × {len(modelos)} modelos ({modelos[0]} lidera)",
                lambda s=s, seed=seed, m=modelos: run_unit(
                    plano.data, plano.out, "full", s, arms, models=m, seed=seed,
                    out_tag=plano.seed_tag(seed), champion=m[0]),
            ))
    _roda_unidades(plano, unidades, "study")


def estagio_aggregate(plano: Plano) -> None:
    from src.feature_study.core.analysis import run_aggregate

    res = run_aggregate(plano.out, plano.data, plano.tags, label="main", n_boot=plano.args.n_boot)
    print(res["ranking_groups"].to_string(index=False))


ESTAGIO_FN = {"prepare": estagio_prepare, "triage": estagio_triage, "screen": estagio_screen,
              "study": estagio_study, "aggregate": estagio_aggregate}


def mostra_plano(plano: Plano) -> None:
    n_arms = len(plano.arms()) if (plano.data / "arms.json").exists() else None
    n_s, n_seed = len(plano.seasons), len(plano.seeds)
    print(f"saída: {plano.out}")
    print(f"seeds: {plano.seeds} | trimestres: {list(plano.seasons)}\n")
    print(f"triage : {n_s * n_seed} unidades (base × 39 modelos)")
    print("screen : instantâneo")
    if n_arms:
        print(f"study  : {n_s * n_seed} unidades × {n_arms} arms × {plano.args.k} modelos "
              f"= {n_s * n_seed * n_arms} ajustes de arm")
    else:
        print("study  : depende do prepare (número de arms ainda desconhecido)")
    print("aggregate: bootstrap, alguns minutos\n")
    fits = sorted((plano.out / "units").glob("*_triage/metrics__*__base.parquet")) if (plano.out / "units").exists() else []
    if fits:
        import pandas as pd
        seg = pd.concat([pd.read_parquet(f, columns=["fit_seconds"]) for f in fits]).fit_seconds.mean()
        print(f"medido aqui: {seg:.0f} s por ajuste de arm com 39 modelos")
    else:
        print("sem medida local ainda — rode o triage de um trimestre para calibrar o tempo")


def mostra_status(plano: Plano) -> None:
    units = plano.out / "units"
    print(f"{plano.out}\n")
    print("prepare :", "ok" if (plano.data / "meta.json").exists() else "pendente")
    for tag in (*plano.triage_tags, *plano.tags):
        pasta = units / tag
        n_m = len(list(pasta.glob("metrics__*.parquet"))) if pasta.exists() else 0
        n_base = len(list(pasta.glob("metrics__*__base.parquet"))) if pasta.exists() else 0
        print(f"{tag:10s}: {n_m} arquivos de métricas ({n_base}/4 do arm base)")
    print("screen  :", "ok" if (plano.out / "summary/top_models.json").exists() else "pendente")
    print("aggregate:", "ok" if (plano.out / "summary/main/effects.csv").exists() else "pendente")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=("plan", "run", "status"))
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--seeds", default="42", help="seeds separadas por vírgula (padrão: só a 42)")
    ap.add_argument("--seasons", default="", help="restringe os trimestres (ex.: DJF); vazio = todos. "
                    "Útil para calibrar o tempo com uma unidade")
    ap.add_argument("--threads", type=int, default=4, help="teto de threads (padrão 4 de 8)")
    ap.add_argument("--k", type=int, default=5, help="modelos por trimestre na triagem")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-free-gb", type=float, default=1.0, help="RAM livre mínima antes de cada unidade")
    ap.add_argument("--force", action="store_true", help="refaz o prepare")
    args = ap.parse_args()

    _limita_threads(args.threads)
    plano = Plano(args)
    if args.comando == "plan":
        return mostra_plano(plano)
    if args.comando == "status":
        return mostra_status(plano)

    estagios = STAGES if args.stage == "all" else (args.stage,)
    try:
        for e in estagios:
            print(f"\n══ {e} ══", flush=True)
            ESTAGIO_FN[e](plano)
    except KeyboardInterrupt:
        print("\ninterrompido — o que já ficou pronto é pulado na próxima execução")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
