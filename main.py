"""Dispatcher único para todos os pipelines IRC Vendaval."""
from __future__ import annotations

import argparse


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw-dir", default="dataset/raw")
    p.add_argument("--shp-dir", default="dataset/shp")
    p.add_argument(
        "--exp-name", default=None,
        help="Nome do experimento (subpasta em --output-dir)",
    )


def _add_spatial_validation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--validation-mode", choices=["temporal", "spatial-kfold", "loocv"],
        default="temporal",
        help="'temporal' (padrão, split atual) | 'spatial-kfold' (station-holdout "
             "em k folds geograficamente coesos) | 'loocv' (leave-one-station-out). "
             "Modos espaciais são ADITIVOS: geram linhas split=spatial_holdout em "
             "results.csv sem alterar as linhas train/val/test existentes.",
    )
    p.add_argument("--spatial-n-folds", type=int, default=5,
                    help="Nº de folds espaciais quando --validation-mode=spatial-kfold (default: 5)")
    p.add_argument("--spatial-seed", type=int, default=42,
                    help="Seed do KMeans espacial (default: 42)")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="main.py",
        description="IRC Vendaval — pipelines de correção de rajadas ERA5→INMET",
    )
    sub = root.add_subparsers(
        dest="pipeline", required=True, metavar="PIPELINE"
    )

    # ── cluster_lstm ──────────────────────────────────────────────────────────
    p1 = sub.add_parser("cluster_lstm", help="LSTM dual-head por cluster")
    p1.add_argument("--config", required=True)
    p1.add_argument("--augmentation-method", default=None)
    p1.add_argument("--synthetic-csv", default=None)
    p1.add_argument("--feature-groups", default=None,
                    help="Grupos separados por vírgula: original, era5_18z, bt55 (ou 'all'); "
                         "sobrepõe data.feature_groups do YAML")
    p1.add_argument("--restrict-coverage", action="store_true",
                    help="Restringe às estações com cobertura REAL de era5_18z/bt55 "
                         "(~46-57/243), em vez de treinar com NaN/imputação nas demais. "
                         "Sem efeito se --feature-groups não incluir era5_18z/bt55.")
    p1.add_argument("--ablation-group", default=None,
                    help="Tag opcional para agrupar experimentos de ablation no dashboard")
    p1.add_argument("--exp-name", default=None,
                    help="Sobrepõe experiment.name do YAML (útil para rodar a mesma config "
                         "em vários experimentos isolados)")

    # ── cluster_mlp ───────────────────────────────────────────────────────────
    p2 = sub.add_parser("cluster_mlp", help="MLPRegressor com extreme-weighting")
    _add_common(p2)
    p2.add_argument("--output-dir", default="artifacts/mlp_clusters")
    p2.add_argument("--hidden-layers", default="128,64")
    p2.add_argument("--alpha", type=float, default=0.001)
    p2.add_argument("--extreme-power", type=float, default=2.0)
    p2.add_argument("--max-iter", type=int, default=500)
    p2.add_argument("--cluster-merge", default=None)
    p2.add_argument("--synthetic-csv", default=None)
    p2.add_argument("--feature-groups", default="original,era5_18z,bt55",
                    help="Grupos separados por vírgula: original, era5_18z, bt55 (ou 'all')")
    p2.add_argument("--restrict-coverage", action="store_true",
                    help="Restringe às estações com cobertura REAL de era5_18z/bt55 "
                         "(~46-57/243), em vez de treinar com NaN/imputação nas demais. "
                         "Sem efeito se --feature-groups não incluir era5_18z/bt55.")
    p2.add_argument("--ablation-group", default=None,
                    help="Tag opcional para agrupar experimentos de ablation no dashboard")
    _add_spatial_validation_args(p2)

    # ── cluster_lazy ──────────────────────────────────────────────────────────
    p3 = sub.add_parser("cluster_lazy", help="LazyPredict benchmark por cluster")
    _add_common(p3)
    p3.add_argument("--output-dir", default="artifacts/lazy_clusters")
    p3.add_argument("--cluster-merge", default=None)
    p3.add_argument("--synthetic-csv", default=None)
    p3.add_argument("--synth-n-above", type=int, default=None,
                    help="Sintéticos > P90 por cluster (default: todos)")
    p3.add_argument("--synth-n-below", type=int, default=0,
                    help="Sintéticos < P90 por cluster (default: 0)")
    p3.add_argument("--extreme-percentile", type=float, default=0.90,
                    help="Percentil extremo/normal (default: 0.90)")
    p3.add_argument("--no-stratify-seasons", action="store_false", dest="stratify_seasons",
                    help="Desabilitar treino por cluster × trimestre (padrão: ativado)")
    p3.add_argument("--n-neighbor-clusters", type=int, default=1,
                    help="Nº de clusters vizinhos cujos dados entram no treino (default: 1)")
    p3.add_argument("--eval-window", choices=["monthly", "biweekly"], default="monthly",
                    help="Janela de avaliação de deploy: monthly ou biweekly (default: monthly)")
    p3.add_argument("--cluster-id", default=None)
    p3.add_argument("--aggregate-only", action="store_true")
    p3.add_argument("--list-clusters", action="store_true")
    p3.add_argument("--feature-groups", default="original,era5_18z,bt55",
                    help="Grupos separados por vírgula: original, era5_18z, bt55 (ou 'all')")
    p3.add_argument("--restrict-coverage", action="store_true",
                    help="Restringe às estações com cobertura REAL de era5_18z/bt55 "
                         "(~46-57/243), em vez de treinar com NaN/imputação nas demais. "
                         "Sem efeito se --feature-groups não incluir era5_18z/bt55.")
    p3.add_argument("--ablation-group", default=None,
                    help="Tag opcional para agrupar experimentos de ablation no dashboard")
    _add_spatial_validation_args(p3)

    # ── cluster_gan ───────────────────────────────────────────────────────────
    p4 = sub.add_parser("cluster_gan", help="cWGAN-GP condicional (alvo) + GPD + nearest-neighbor")
    _add_common(p4)
    p4.add_argument("--output-dir", default="artifacts/gan_clusters")
    p4.add_argument("--epochs", type=int, default=300)
    p4.add_argument("--extreme-percentile", type=float, default=90.0,
                    help="Percentil 0-100 (convenção do GAN/EVT — difere da fração 0-1 do cluster_lazy)")
    p4.add_argument("--n-per-cluster", type=int, default=None)
    p4.add_argument("--n-per-cluster-ratio", type=float, default=0.3)
    p4.add_argument("--cluster-merge", default=None)
    p4.add_argument("--include-season", action="store_true")

    # ── corrected_grid ────────────────────────────────────────────────────────
    p5 = sub.add_parser(
        "corrected_grid",
        help="Grade NetCDF corrigida (magnitude + direção) usando o MELHOR "
             "modelo já treinado por (cluster, trimestre climático)",
    )
    _add_common(p5)
    p5.add_argument("--artifacts-root", default="artifacts")
    p5.add_argument("--out-dir", default="artifacts/corrected_grid")
    p5.add_argument("--start-date", default="2000-01-08",
                    help="INMET começa em 2000-01-01; os 7 primeiros dias "
                         "ficam sem lags/rolling — default pula essa semana")
    p5.add_argument("--end-date", default="2025-12-31")
    p5.add_argument("--metric", default="R2",
                    choices=["R2", "RMSE", "Bias", "Bias_P90", "RMSE_P90"],
                    help="Métrica de seleção do melhor (pipeline, arm) por cluster × trimestre")
    p5.add_argument("--prediction-source", choices=["reload", "saved"], default="reload")
    p5.add_argument("--interp-mode", choices=["basin", "per-cluster"], default="basin")
    p5.add_argument("--idw-neighbors", type=int, default=15)
    p5.add_argument("--idw-power", type=float, default=2.0)
    p5.add_argument("--smoothing", choices=["none", "gaussian"], default="none")
    p5.add_argument("--consolidated", action="store_true",
                    help="Um único NetCDF em vez de um por ano (default: um por ano)")
    p5.add_argument("--clusters", default=None,
                    help="Subconjunto de clusters pra smoke test, ex: '1,2'")
    p5.add_argument("--winners-only", action="store_true",
                    help="Só imprime/salva a tabela de vencedores e sai (sem gerar a grade)")

    return root


def _dispatch(args: argparse.Namespace) -> None:
    if args.pipeline == "cluster_lstm":
        from src.pipelines.cluster_lstm import run
        run(
            config=args.config,
            augmentation_method=args.augmentation_method,
            synthetic_csv=args.synthetic_csv,
            feature_groups=args.feature_groups,
            restrict_coverage=args.restrict_coverage,
            ablation_group=args.ablation_group,
            exp_name_override=args.exp_name,
        )

    elif args.pipeline == "cluster_mlp":
        from src.pipelines.cluster_mlp import run
        run(
            raw_dir=args.raw_dir,
            shp_dir=args.shp_dir,
            output_dir=args.output_dir,
            hidden_layers=tuple(int(x) for x in args.hidden_layers.split(",")),
            alpha=args.alpha,
            extreme_power=args.extreme_power,
            max_iter=args.max_iter,
            cluster_merge=args.cluster_merge,
            synthetic_csv=args.synthetic_csv,
            exp_name=args.exp_name,
            feature_groups=args.feature_groups,
            restrict_coverage=args.restrict_coverage,
            ablation_group=args.ablation_group,
            validation_mode=args.validation_mode,
            spatial_n_folds=args.spatial_n_folds,
            spatial_seed=args.spatial_seed,
        )

    elif args.pipeline == "cluster_lazy":
        from src.pipelines.cluster_lazy import run
        run(
            raw_dir=args.raw_dir,
            shp_dir=args.shp_dir,
            output_dir=args.output_dir,
            cluster_merge=args.cluster_merge,
            synthetic_csv=args.synthetic_csv,
            synth_n_above=args.synth_n_above,
            synth_n_below=args.synth_n_below,
            extreme_percentile=args.extreme_percentile,
            stratify_seasons=args.stratify_seasons,
            n_neighbor_clusters=args.n_neighbor_clusters,
            eval_window=args.eval_window,
            exp_name=args.exp_name,
            cluster_id=args.cluster_id,
            aggregate_only=args.aggregate_only,
            list_clusters=args.list_clusters,
            feature_groups=args.feature_groups,
            restrict_coverage=args.restrict_coverage,
            ablation_group=args.ablation_group,
            validation_mode=args.validation_mode,
            spatial_n_folds=args.spatial_n_folds,
            spatial_seed=args.spatial_seed,
        )

    elif args.pipeline == "cluster_gan":
        from src.pipelines.cluster_gan import run
        run(
            raw_dir=args.raw_dir,
            shp_dir=args.shp_dir,
            output_dir=args.output_dir,
            exp_name=args.exp_name,
            epochs=args.epochs,
            extreme_percentile=args.extreme_percentile,
            n_per_cluster=args.n_per_cluster,
            n_per_cluster_ratio=args.n_per_cluster_ratio,
            cluster_merge=args.cluster_merge,
            include_season=args.include_season,
        )

    elif args.pipeline == "corrected_grid":
        from src.dataset.creation.grid_generator import run
        run(
            raw_dir=args.raw_dir,
            shp_dir=args.shp_dir,
            artifacts_root=args.artifacts_root,
            out_dir=args.out_dir,
            exp_name=args.exp_name,
            start_date=args.start_date,
            end_date=args.end_date,
            metric=args.metric,
            prediction_source=args.prediction_source,
            interp_mode=args.interp_mode,
            idw_neighbors=args.idw_neighbors,
            idw_power=args.idw_power,
            smoothing=args.smoothing,
            per_year=not args.consolidated,
            clusters=args.clusters,
            winners_only=args.winners_only,
        )


if __name__ == "__main__":
    _dispatch(build_parser().parse_args())
