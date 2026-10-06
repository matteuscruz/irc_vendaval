from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

from src.pipeline.data.cluster_preprocessor import SEASONS
from src.pipeline.validation.cluster_metrics import get_cluster_season_arrays
from src.pipeline.training.cluster_trainer import cluster_mask


def _predict_all(data_batch, result, trainer):
    return trainer.predict(
        x_dict=data_batch.x_test,
        models=result.models,
        scaler_y=data_batch.scaler_y,
        feature_names=data_batch.feature_names,
        cluster_ids=data_batch.cluster_ids,
    )


def _draw_season_scatter(ax, y_true, y_pred, season):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        ax.set_title(f"{season}  (Sem dados)")
        ax.axis("off")
        return

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = (
        np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else float("nan")
    )
    sns.regplot(
        x=y_true, y=y_pred, ax=ax,
        scatter_kws={"alpha": 0.4, "color": "steelblue", "s": 20},
        line_kws={"color": "red"},
    )
    extreme_mask = y_true > 12.0
    if np.any(extreme_mask):
        ax.scatter(
            y_true[extreme_mask], y_pred[extreme_mask],
            color="orange", s=30, label=">12 m/s", zorder=5,
        )
        ax.legend(fontsize=8)
    ax.set_title(f"{season}  RMSE={rmse:.2f}  Corr={corr:.2f}")
    ax.set_xlabel("Observado [m/s]")
    ax.set_ylabel("Predito [m/s]")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_cluster_diagnostics(
    data_batch,
    result,
    trainer,
    save_dir: str | None = None,
) -> None:
    """
    Scatter (obs × pred) + histograma de erros por cluster × estação do ano.
    Salva em save_dir/diagnostic_cluster_{id}.png se save_dir for fornecido.
    """
    preds = _predict_all(data_batch, result, trainer)

    for cluster_id in data_batch.cluster_ids:
        available = []
        for s in SEASONS:
            mask = cluster_mask(data_batch.x_test.get(s), data_batch.feature_names, cluster_id)
            if mask is not None and mask.any():
                available.append(s)
        if not available:
            continue

        fig, axes = plt.subplots(
            len(available), 2,
            figsize=(14, 5 * len(available)),
            squeeze=False,
        )
        fig.suptitle(
            f"Diagnóstico — Cluster {cluster_id}", fontsize=16, y=1.01
        )

        for i, season in enumerate(available):
            ax_sc, ax_hist = axes[i, 0], axes[i, 1]
            arrays = get_cluster_season_arrays(data_batch, preds, season, cluster_id)
            if arrays is None:
                ax_sc.set_title(f"{season}  (Sem dados)")
                ax_sc.axis("off")
                ax_hist.axis("off")
                continue
            y_true, y_pred = arrays
            _draw_season_scatter(ax_sc, y_true, y_pred, season)

            errors = y_pred - y_true
            sns.histplot(
                errors, kde=True, ax=ax_hist, color="mediumpurple", bins=25
            )
            ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
            ax_hist.set_title(f"Distribuição de Erro — {season}")
            ax_hist.set_xlabel("Erro [m/s]")

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"diagnostic_cluster_{cluster_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  Salvo: {path}")

        plt.close(fig)


def _plot_season_curves(ax, hist: dict, te_loss: float | None, season: str):
    if "loss" in hist:
        ax.plot(hist["loss"], label="Huber (treino)", color="steelblue")
    if "val_loss" in hist:
        ax.plot(
            hist["val_loss"], label="Huber (val)",
            color="steelblue", linestyle="--",
        )
    if te_loss is not None:
        ax.axhline(
            te_loss, color="seagreen", linestyle=":", linewidth=1.8,
            label=f"Huber (teste) = {te_loss:.3f}",
        )

    ax.set_title(season, fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_learning_curves(
    histories: dict[int, dict[str, dict]],
    test_losses: dict[int, dict[str, float]] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Curvas de convergência (Huber de treino/val por época) por cluster.

    O modelo é um só por cluster ("global"); as losses de teste por estação
    do ano entram como linhas horizontais no mesmo painel.
    Salva em save_dir/convergence_cluster_{id}.png se save_dir for fornecido.
    """
    test_losses = test_losses or {}

    for cluster_id, season_dict in histories.items():
        available = list(season_dict.keys())
        if not available:
            continue

        fig, axes = plt.subplots(
            len(available), 1, figsize=(10, 4 * len(available)), squeeze=False
        )
        fig.suptitle(
            f"Convergência LSTM — Cluster {cluster_id}",
            fontsize=14, y=1.01,
        )

        cluster_test = test_losses.get(cluster_id, {})
        for i, key in enumerate(available):
            ax = axes[i, 0]
            _plot_season_curves(ax, season_dict[key], cluster_test.get(key), key)
            if key == "global":
                colors = plt.cm.tab10.colors
                for j, (season, te) in enumerate(cluster_test.items()):
                    ax.axhline(
                        te, color=colors[(j + 2) % len(colors)], linestyle=":",
                        linewidth=1.2, label=f"teste {season} = {te:.3f}",
                    )
                ax.legend(fontsize=8)

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"convergence_cluster_{cluster_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  Salvo: {path}")

        plt.close(fig)
