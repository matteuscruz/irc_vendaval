"""Pipeline de geração de dados sintéticos (GAN condicional tabular) por cluster.

Gera rajadas sintéticas (foco em extremos, via EVT/GPD + WGAN-GP condicional,
TabularExtremeGANAugmenter) e atribui a cada uma um vetor de features REAL por
nearest-neighbor (mesmo alvo real mais próximo), produzindo um CSV completo
compatível com o `--synthetic-csv` já consumido por cluster_lazy/cluster_mlp.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data.netcdf_loader import NetCDFLoader
from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_climatology
from src.pipelines.common import (
    BASE_FEATURES, TARGET_VAR, ERA5_GUST_PROXY, TRAIN_SLICE,
    build_flat_dataframe, make_split, month_to_season,
    parse_cluster_merge, apply_cluster_merge,
)
from src.pipeline.augmentation.tabular_gan_augmenter import TabularExtremeGANAugmenter
from src.utils.artifact_manager import ArtifactManager

warnings.filterwarnings("ignore")


def _nearest_neighbor_assign(
    y_real: np.ndarray, x_real: pd.DataFrame, y_synth: np.ndarray,
) -> pd.DataFrame:
    """Atribui a cada y_synth o vetor de features do y_real mais próximo.

    Preserva a relação real feature↔alvo (ex: wind_mag_max coerente com a
    rajada), em vez de gerar features independentes que poderiam produzir
    combinações fisicamente implausíveis.
    """
    order = np.argsort(y_real)
    y_sorted = y_real[order]
    idx = np.searchsorted(y_sorted, y_synth)
    idx = np.clip(idx, 0, len(y_real) - 1)
    return x_real.iloc[order[idx]].reset_index(drop=True)


def _plot_synthetic_vs_real(y_real: np.ndarray, y_synth: np.ndarray, manager, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(float(y_real.max()), float(y_synth.max()), 1.0), 40)
    ax.hist(y_real, bins=bins, density=True, alpha=0.6, color="#1f77b4", label=f"Real (n={len(y_real)})")
    ax.hist(y_synth, bins=bins, density=True, alpha=0.6, color="#d62728", label=f"Sintético (n={len(y_synth)})")
    for p in (90, 95, 99):
        ax.axvline(np.percentile(y_real, p), color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Distribuição do alvo — real vs sintético (pool de todos os clusters)")
    ax.set_xlabel("Rajada máx. (m/s)")
    ax.set_ylabel("Densidade")
    ax.legend()
    path = manager.get_plot_path("summary", "synthetic_vs_real_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[cluster_gan] Distribuição salva: {path}")


def fit_gan(
    raw_dir: str, shp_dir: str, cluster_merge: str | None,
    extreme_percentile: float, epochs: int, include_season: bool,
) -> tuple[TabularExtremeGANAugmenter, dict, np.ndarray]:
    """Carrega os dados, monta o pool de treino por cluster e ajusta o GAN
    condicional (EVT/GPD + WGAN-GP). Extraído de `run()` sem mudança de
    lógica — `TabularExtremeGANAugmenter` já tinha fit/generate separados
    (`src/pipeline/augmentation/tabular_gan_augmenter.py`), só a orquestração
    ao redor estava tudo dentro de `run()`."""
    print("[cluster_gan] Carregando NetCDF...")
    ds_inmet, ds_era5 = NetCDFLoader(raw_dir).load_extended()

    print("[cluster_gan] Atribuindo clusters...")
    station_clusters = assign_station_clusters(ds_inmet, shp_dir)

    print("[cluster_gan] Calculando climatologia ERA5...")
    ds_clim = get_climatology(ds_era5, ERA5_GUST_PROXY, slice(*TRAIN_SLICE))

    print("[cluster_gan] Construindo DataFrame de features...")
    df = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim)

    merge_groups = parse_cluster_merge(cluster_merge)
    if merge_groups:
        df = apply_cluster_merge(df, merge_groups)
        labels = sorted("-".join(str(m) for m in sorted(g)) for g in merge_groups)
        print(f"[cluster_gan] Agregando clusters: {', '.join(labels)}")

    df_tr_by_cluster: dict = {}
    pooled_y, pooled_cid, pooled_season = [], [], []
    for cid, group in df.groupby("cluster_id"):
        df_tr = make_split(group, TRAIN_SLICE).dropna(subset=[TARGET_VAR]).reset_index(drop=True)
        if df_tr.empty:
            continue
        df_tr_by_cluster[cid] = df_tr
        pooled_y.append(df_tr[TARGET_VAR].to_numpy(float))
        pooled_cid.append(np.full(len(df_tr), cid, dtype=object))
        if include_season:
            pooled_season.append(month_to_season(df_tr["time"].dt.month).to_numpy())

    if not df_tr_by_cluster:
        raise RuntimeError("Nenhum cluster com dados de treino — abortando.")

    y_train_all = np.concatenate(pooled_y)
    cluster_ids_all = np.concatenate(pooled_cid)
    season_labels_all = np.concatenate(pooled_season) if include_season else None

    print(f"[cluster_gan] Treinando GAN em {len(y_train_all)} amostras "
          f"({len(df_tr_by_cluster)} clusters)...")
    augmenter = TabularExtremeGANAugmenter(
        extreme_percentile=extreme_percentile,
        epochs=epochs,
        include_season=include_season,
    )
    augmenter.fit(y_train_all, cluster_ids_all, season_labels_all)

    return augmenter, df_tr_by_cluster, y_train_all


def generate_and_save(
    augmenter: TabularExtremeGANAugmenter, df_tr_by_cluster: dict, y_train_all: np.ndarray,
    *, n_per_cluster: int | None, n_per_cluster_ratio: float | None, include_season: bool,
    cluster_merge: str | None, epochs: int, extreme_percentile: float, manager, plt,
) -> pd.DataFrame:
    """Gera as amostras sintéticas do GAN já ajustado, atribui features reais
    por nearest-neighbor, salva o CSV + relatório de diagnóstico. Extraído de
    `run()` sem mudança de lógica."""
    n_gen_by_cluster = {}
    for cid, df_tr in df_tr_by_cluster.items():
        n_gen_by_cluster[cid] = (
            n_per_cluster if n_per_cluster is not None
            else max(1, round((n_per_cluster_ratio or 0.3) * len(df_tr)))
        )

    print(f"[cluster_gan] Gerando sintéticos: {n_gen_by_cluster}")
    synth_targets = augmenter.generate(n_gen_by_cluster)

    rows = []
    for cid, df_tr in df_tr_by_cluster.items():
        cluster_synth = synth_targets[synth_targets["cluster_id"] == cid]
        if cluster_synth.empty:
            continue
        y_real = df_tr[TARGET_VAR].to_numpy(float)
        x_real = df_tr.reindex(columns=BASE_FEATURES)
        y_synth = cluster_synth["y_synth"].to_numpy(float)

        x_assigned = _nearest_neighbor_assign(y_real, x_real, y_synth)
        x_assigned[TARGET_VAR] = y_synth
        x_assigned["cluster_id"] = cid
        x_assigned["is_extreme"] = cluster_synth["is_extreme"].to_numpy()
        if include_season:
            x_assigned["season"] = cluster_synth["season"].to_numpy()
        rows.append(x_assigned)

    synthetic_df = pd.concat(rows, ignore_index=True)
    csv_path = manager.get_root_path("synthetic_augment.csv")
    synthetic_df.to_csv(csv_path, index=False)
    print(f"[cluster_gan] Sintéticos salvos: {csv_path} ({len(synthetic_df)} linhas)")

    percentiles = [50, 90, 95, 99]
    real_pct = {p: float(np.percentile(y_train_all, p)) for p in percentiles}
    synth_pct = {p: float(np.percentile(synthetic_df[TARGET_VAR], p)) for p in percentiles}
    real_extreme = y_train_all[y_train_all >= augmenter._threshold]
    print("[cluster_gan] Percentis — real (pool completo) vs sintético:")
    for p in percentiles:
        print(f"   P{p}: real={real_pct[p]:.2f}  synth={synth_pct[p]:.2f}")
    print(f"   max: real={y_train_all.max():.2f}  synth={synthetic_df[TARGET_VAR].max():.2f}")
    print(
        "[cluster_gan] AVISO: o GAN gera deliberadamente amostras extremas "
        "(is_extreme=True sempre) — não espere synth≈real nos percentis do "
        "pool completo. A comparação correta é contra a CAUDA real "
        f"(y >= threshold={augmenter._threshold:.2f}, n={len(real_extreme)}): "
        f"real_extreme[min,mean,max]=[{real_extreme.min():.2f}, {real_extreme.mean():.2f}, {real_extreme.max():.2f}] "
        f"vs synth[min,mean,max]=[{synthetic_df[TARGET_VAR].min():.2f}, {synthetic_df[TARGET_VAR].mean():.2f}, {synthetic_df[TARGET_VAR].max():.2f}]"
    )

    _plot_synthetic_vs_real(y_train_all, synthetic_df[TARGET_VAR].to_numpy(), manager, plt)

    manager.write_run_meta({
        "epochs": epochs,
        "extreme_percentile": extreme_percentile,
        "n_per_cluster": n_per_cluster,
        "n_per_cluster_ratio": n_per_cluster_ratio,
        "include_season": include_season,
        "cluster_merge": cluster_merge,
        "n_synthetic_rows": len(synthetic_df),
        "gpd_shape": augmenter._gpd_shape,
        "gpd_scale": augmenter._gpd_scale,
        "gpd_threshold": augmenter._threshold,
        "real_percentiles": real_pct,
        "synthetic_percentiles": synth_pct,
        "real_extreme_stats": {
            "n": int(len(real_extreme)),
            "min": float(real_extreme.min()),
            "mean": float(real_extreme.mean()),
            "max": float(real_extreme.max()),
        },
        "synthetic_stats": {
            "min": float(synthetic_df[TARGET_VAR].min()),
            "mean": float(synthetic_df[TARGET_VAR].mean()),
            "max": float(synthetic_df[TARGET_VAR].max()),
        },
    })

    return synthetic_df


def run(
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    output_dir: str = "artifacts/gan_clusters",
    exp_name: str | None = None,
    epochs: int = 300,
    extreme_percentile: float = 90.0,
    n_per_cluster: int | None = None,
    n_per_cluster_ratio: float | None = 0.3,
    include_season: bool = False,
    cluster_merge: str | None = None,
    **_,
) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manager = ArtifactManager(output_dir, exp_name)
    print(f"[cluster_gan] Experimento: {manager.root}")

    augmenter, df_tr_by_cluster, y_train_all = fit_gan(
        raw_dir, shp_dir, cluster_merge, extreme_percentile, epochs, include_season,
    )

    return generate_and_save(
        augmenter, df_tr_by_cluster, y_train_all,
        n_per_cluster=n_per_cluster, n_per_cluster_ratio=n_per_cluster_ratio,
        include_season=include_season, cluster_merge=cluster_merge,
        epochs=epochs, extreme_percentile=extreme_percentile,
        manager=manager, plt=plt,
    )
