from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from src.config.schema import VisualizationConfig


class Plotter:
    @staticmethod
    def plot_all(
        histories: dict[str, dict],
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        cfg: VisualizationConfig,
        run_dir: str,
    ) -> None:
        save_dir = os.path.join(run_dir, cfg.save_dir.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)

        dispatch = {
            "convergence": lambda: Plotter.plot_convergence(histories, save_dir),
            "scatter": lambda: Plotter.plot_scatter(y_true, predictions, save_dir),
            "violin": lambda: Plotter.plot_violin(y_true, predictions, save_dir),
            "bar_r2": lambda: Plotter.plot_bar_r2(y_true, predictions, save_dir),
            "bar_bias": lambda: Plotter.plot_bar_bias(y_true, predictions, save_dir),
            "tradeoff": lambda: Plotter.plot_tradeoff(y_true, predictions, save_dir),
        }

        for name in cfg.plots:
            if name in dispatch:
                dispatch[name]()

    @staticmethod
    def plot_convergence(histories: dict[str, dict], save_dir: str) -> None:
        n = len(histories)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, (name, hist) in zip(axes[0], histories.items()):
            ax.plot(hist.get("loss", []), label="train")
            ax.plot(hist.get("val_loss", []), label="val")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "convergence.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_scatter(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_dir: str,
        p90_thresh: float | None = None,
        p95_thresh: float | None = None,
    ) -> None:
        p90 = p90_thresh or float(np.quantile(y_true, 0.90))
        p95 = p95_thresh or float(np.quantile(y_true, 0.95))

        n = len(predictions)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        for ax, (name, y_pred) in zip(axes[0], predictions.items()):
            colors = np.where(
                y_true >= p95, "red", np.where(y_true >= p90, "orange", "steelblue")
            )
            ax.scatter(y_true, y_pred, c=colors, alpha=0.4, s=6)
            lim = max(y_true.max(), y_pred.max())
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("y_true (m/s)")
            ax.set_ylabel("y_pred (m/s)")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "scatter.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_violin(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_dir: str,
    ) -> None:
        p90 = float(np.quantile(y_true, 0.90))
        mask_ext = y_true >= p90
        mask_norm = ~mask_ext

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mask, title in [
            (axes[0], mask_norm, "Normal (< P90)"),
            (axes[1], mask_ext, "Extreme (≥ P90)"),
        ]:
            data = [y_pred[mask] - y_true[mask] for y_pred in predictions.values()]
            parts = ax.violinplot(data, showmedians=True)
            ax.set_title(title)
            ax.set_xticks(range(1, len(predictions) + 1))
            ax.set_xticklabels(list(predictions.keys()), rotation=30, ha="right", fontsize=7)
            ax.axhline(0, color="red", lw=1, ls="--")
            ax.set_ylabel("Residual (pred − true)")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "violin.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_bar_r2(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_dir: str,
    ) -> None:
        from sklearn.metrics import r2_score

        names = list(predictions)
        r2s = [r2_score(y_true, predictions[n]) for n in names]
        order = np.argsort(r2s)[::-1]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([names[i] for i in order], [r2s[i] for i in order])
        ax.set_ylabel("R²")
        ax.set_title("R² por função de loss")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "bar_r2.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_bar_bias(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_dir: str,
        percentile: float = 0.90,
    ) -> None:
        thresh = float(np.quantile(y_true, percentile))
        mask = y_true >= thresh

        names = list(predictions)
        biases = [float(np.mean(predictions[n][mask] - y_true[mask])) for n in names]
        colors = ["green" if b >= 0 else "red" for b in biases]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, biases, color=colors)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel(f"Bias P{int(percentile*100)} (m/s)")
        ax.set_title(f"Bias em extremos (≥ P{int(percentile*100)})")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "bar_bias.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_tradeoff(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
        save_dir: str,
        percentile: float = 0.90,
    ) -> None:
        from sklearn.metrics import r2_score

        thresh = float(np.quantile(y_true, percentile))
        mask = y_true >= thresh

        names = list(predictions)
        r2s = [r2_score(y_true, predictions[n]) for n in names]
        biases = [float(np.mean(predictions[n][mask] - y_true[mask])) for n in names]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(r2s, biases, s=60)
        for name, x, y in zip(names, r2s, biases):
            ax.annotate(name, (x, y), fontsize=7, textcoords="offset points", xytext=(5, 3))
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_xlabel("R² geral")
        ax.set_ylabel(f"Bias P{int(percentile*100)} (m/s)")
        ax.set_title("Trade-off: acurácia global vs. calibração em extremos")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "tradeoff.png"), dpi=150)
        plt.close(fig)
