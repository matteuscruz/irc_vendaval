from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

from src.pipeline.data.cluster_preprocessor import SEASONS


def _cluster_test_data(data_batch, season, col_idx):
    """Retorna (y_true, y_pred_abs) para um cluster/estação no teste, ou None."""
    x_all = data_batch.x_test[season].astype("float32")
    y_all = data_batch.y_test[season].astype("float32")
    era5_all = np.array(data_batch.era5_test[season], dtype="float32")
    n = min(len(x_all), len(y_all), len(era5_all))
    mask = x_all[:n, -1, col_idx] == 1
    era5_safe = np.clip(era5_all[:n], 0.1, None)
    y_true = (
        data_batch.scaler_y.inverse_transform(y_all[:n]).flatten()
        * era5_safe
    )[mask]
    return y_true, mask, n


def _predict_all(data_batch, result, trainer):
    from src.pipeline.training.cluster_tr_trainer import ClusterTRTrainer

    if isinstance(trainer, ClusterTRTrainer):
        return trainer.predict(
            x_dict=data_batch.x_test,
            xs_dict=data_batch.x_static_test,
            models=result.models,
            era5_dict=data_batch.era5_test,
            scaler_y=data_batch.scaler_y,
            feature_names=data_batch.feature_names,
            cluster_ids=data_batch.cluster_ids,
        )
    return trainer.predict(
        x_dict=data_batch.x_test,
        models=result.models,
        era5_dict=data_batch.era5_test,
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
    Scatter (obs × pred) + histograma de erros por cluster × estação climática.
    Salva em save_dir/diagnostic_cluster_{id}.png se save_dir for fornecido.
    """
    preds = _predict_all(data_batch, result, trainer)

    for cluster_id in data_batch.cluster_ids:
        col = f"cluster_{cluster_id}"
        if col not in data_batch.feature_names:
            continue
        col_idx = data_batch.feature_names.index(col)

        available = [
            s for s in SEASONS
            if s in data_batch.x_test
            and np.any(
                data_batch.x_test[s].astype("float32")[:, -1, col_idx] == 1
            )
        ]
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
            y_true, mask, n = _cluster_test_data(
                data_batch, season, col_idx
            )
            y_pred = preds.get(season, np.array([]))[:n][mask]

            if len(y_true) == 0:
                continue

            ax_sc, ax_hist = axes[i, 0], axes[i, 1]
            _draw_season_scatter(ax_sc, y_true, y_pred, season)

            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            if np.any(valid):
                errors = y_pred[valid] - y_true[valid]
                sns.histplot(
                    errors, kde=True, ax=ax_hist, color="mediumpurple", bins=25
                )
                ax_hist.axvline(0, color="black", linestyle="--", linewidth=1)
                ax_hist.set_title(f"Distribuição de Erro — {season}")
                ax_hist.set_xlabel("Erro [m/s]")
            else:
                ax_hist.set_title(f"Distribuição de Erro — {season} (Sem dados)")
                ax_hist.axis("off")

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"diagnostic_cluster_{cluster_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  Salvo: {path}")

        plt.close(fig)


def _resolve_loss_key(hist: dict, base: str, val: bool = False) -> str | None:
    """Resolve chave de loss independente do separador do Keras."""
    prefix = "val_" if val else ""
    for sep in ("_", "/"):
        k = f"{prefix}{base.replace('_', sep, 1)}"
        if k in hist:
            return k
    k_exact = f"{prefix}{base}"
    return k_exact if k_exact in hist else None


def _plot_season_curves(ax, hist: dict, te_loss: float | None, season: str):
    norm_key     = _resolve_loss_key(hist, "head_normal_loss")
    val_norm_key = _resolve_loss_key(hist, "head_normal_loss", val=True)
    ext_key      = _resolve_loss_key(hist, "head_extreme_loss")
    val_ext_key  = _resolve_loss_key(hist, "head_extreme_loss", val=True)
    loss_key     = _resolve_loss_key(hist, "loss")
    val_loss_key = _resolve_loss_key(hist, "loss", val=True)

    if norm_key:
        ax.plot(hist[norm_key], label="Normal (treino)", color="steelblue")
    if val_norm_key:
        ax.plot(
            hist[val_norm_key], label="Normal (val)",
            color="steelblue", linestyle="--",
        )
    if ext_key:
        ax.plot(hist[ext_key], label="Extreme (treino)", color="orangered")
    if val_ext_key:
        ax.plot(
            hist[val_ext_key], label="Extreme (val)",
            color="orangered", linestyle="--",
        )

    if not norm_key and not ext_key:
        if loss_key:
            ax.plot(hist[loss_key], label="Loss (treino)", color="steelblue")
        if val_loss_key:
            ax.plot(
                hist[val_loss_key], label="Loss (val)",
                color="steelblue", linestyle="--",
            )

    if te_loss is not None:
        ax.axhline(
            te_loss, color="seagreen", linestyle=":", linewidth=1.8,
            label=f"Loss (teste) = {te_loss:.3f}",
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
    Curvas de convergência (loss por cabeça) por cluster × estação climática.

    Mostra treino e validação por época + loss de teste como linha horizontal.
    Salva em save_dir/convergence_cluster_{id}.png se save_dir for fornecido.
    """
    test_losses = test_losses or {}

    for cluster_id, season_dict in histories.items():
        # Chaves de season_dict: "global" (treino pooled, 1 modelo/cluster) ou,
        # em artefatos antigos, os 4 códigos de estação — list(keys()) cobre
        # os dois formatos sem hardcode.
        available = list(season_dict.keys())
        if not available:
            continue

        n_rows = len(available)
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False
        )
        fig.suptitle(
            f"Convergência Dual-Head — Cluster {cluster_id}",
            fontsize=14, y=1.01,
        )

        cluster_test = test_losses.get(cluster_id, {})
        for i, season in enumerate(available):
            _plot_season_curves(
                axes[i, 0],
                season_dict[season],
                cluster_test.get(season),
                season,
            )

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"convergence_cluster_{cluster_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  Salvo: {path}")

        plt.close(fig)
