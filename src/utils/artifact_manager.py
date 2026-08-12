from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ArtifactManager:
    """Gerenciador centralizado de caminhos e artefatos de experimentos.
    
    Adota a Rota Passiva: fornece os objetos Path() já garantindo que a árvore
    de pastas exista (mkdir), mas não se acopla às bibliotecas de I/O (Pandas, Joblib, etc).
    """

    def __init__(self, base_dir: str | Path, exp_name: str | None = None):
        """Inicializa o gerenciador. Se exp_name for None, autoincrementa (exp1, exp2...)."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = self._resolve_exp_dir(exp_name)

    def _resolve_exp_dir(self, exp_name: str | None) -> Path:
        if exp_name:
            exp_dir = self.base_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir

        existing = [
            int(m.group(1))
            for p in self.base_dir.iterdir()
            if p.is_dir() and (m := re.fullmatch(r"exp(\d+)", p.name))
        ]
        n = max(existing) + 1 if existing else 1
        exp_dir = self.base_dir / f"exp{n}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    @property
    def root(self) -> Path:
        """Retorna a raiz do experimento (ex: artifacts/cluster_lazy/exp1)."""
        return self.exp_dir

    def get_plot_dir(self, plot_type: str = "") -> Path:
        """Retorna a subpasta para um tipo específico de gráfico.
        
        Se plot_type for vazio, retorna a pasta raiz de plots.
        Ex: manager.get_plot_dir("rolling_r2") -> expX/plots/rolling_r2/
        """
        d = self.exp_dir / "plots"
        if plot_type:
            d = d / plot_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_plot_path(self, plot_type: str, filename: str) -> Path:
        """Retorna o caminho completo para salvar um gráfico."""
        return self.get_plot_dir(plot_type) / filename

    def get_partial_dir(self, partial_type: str) -> Path:
        """Retorna a subpasta para artefatos parciais (ex: 'clusters', 'meta')."""
        d = self.exp_dir / "_partial" / partial_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_partial_path(self, partial_type: str, filename: str) -> Path:
        """Retorna o caminho completo para salvar um arquivo parcial."""
        return self.get_partial_dir(partial_type) / filename

    def get_model_dir(self) -> Path:
        """Retorna a pasta para salvar os modelos serializados (fitted_models)."""
        d = self.exp_dir / "fitted_models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_model_path(self, filename: str) -> Path:
        """Retorna o caminho completo para salvar um modelo."""
        return self.get_model_dir() / filename

    def get_prediction_dir(self) -> Path:
        """Retorna a pasta para previsões de validação/teste."""
        d = self.exp_dir / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_prediction_path(self, filename: str) -> Path:
        """Retorna o caminho completo para salvar um arquivo de previsões."""
        return self.get_prediction_dir() / filename

    def get_root_path(self, filename: str) -> Path:
        """Retorna um caminho na raiz do experimento (ex: resultados agregados)."""
        return self.exp_dir / filename

    def write_run_meta(self, extra: dict, filename: str = "run_meta.json") -> Path:
        """Mescla {experiment, timestamp} com `extra` e grava JSON na raiz."""
        meta = {
            "experiment": self.exp_dir.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        path = self.get_root_path(filename)
        path.write_text(json.dumps(meta, indent=2, default=str))
        return path

    def append_experiments_index(
        self, rows: "pd.DataFrame", filename: str = "experiments_index.csv"
    ) -> Path:
        """Adiciona `rows` ao índice de experimentos em base_dir (união de colunas).

        Diferente de um append puro (mode="a"), relê o índice existente e
        reescreve com pd.concat(sort=False) para não corromper o CSV quando
        uma run tem colunas extras diferentes de runs anteriores.
        """
        rows = rows.copy()
        if "experiment" not in rows.columns:
            rows.insert(0, "experiment", self.exp_dir.name)
        if "timestamp" not in rows.columns:
            rows.insert(1, "timestamp", datetime.now(timezone.utc).isoformat())

        index_path = self.base_dir / filename
        if index_path.exists():
            existing = pd.read_csv(index_path)
            combined = pd.concat([existing, rows], ignore_index=True, sort=False)
        else:
            combined = rows
        combined.to_csv(index_path, index=False)
        return index_path

    def write_results(self, df: "pd.DataFrame", filename: str = "results.csv") -> Path:
        """Grava o results.csv (schema comum) na raiz do experimento."""
        path = self.get_root_path(filename)
        df.to_csv(path, index=False)
        return path
