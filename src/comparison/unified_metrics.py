"""Ponte entre as métricas de IA (`irc_vendaval`) e as métricas do projeto de
interpolação clássica (`interpolation comparisson/`), no domínio comum onde a
comparação faz sentido: EXTREMOS ANUAIS (P95/P99 dos máximos anuais por
estação) — não no domínio diário, onde os métodos clássicos não operam (são
interpoladores espaciais de campos anuais, não modelos de série temporal).

`derive_annual_percentiles` replica EXATAMENTE a lógica de
`interpolation comparisson/streamlit_app/data_io.py::load_station_annual_percentiles`
(groupby estação×ano → máximo anual → quantil EMPÍRICO 0.95/0.99 sobre a
série de máximos anuais, sem exigir um nº mínimo de anos) — qualquer desvio
dessa definição (ex.: quantil direto sobre a série diária) invalida a
comparação com os métodos clássicos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CLASSICAL_STREAMLIT_DIR = str(
    Path(__file__).resolve().parents[2] / "interpolation comparisson" / "streamlit_app"
)
if _CLASSICAL_STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, _CLASSICAL_STREAMLIT_DIR)

from validation import compute_validation_metrics  # noqa: E402


def derive_annual_percentiles(
    pred_by_station_df: pd.DataFrame,
    value_col: str,
    station_col: str = "estacao",
    time_col: str = "time",
) -> pd.DataFrame:
    """P95/P99 empíricos dos máximos anuais de `value_col`, por estação.

    Réplica de `load_station_annual_percentiles`: NÃO exige um nº mínimo de
    anos (esse filtro só existe no projeto clássico para estabilizar o ajuste
    paramétrico GEV, usado exclusivamente pelo MSP Método 2 — irrelevante
    aqui). Retorna DataFrame [estacao, p95, p99, n_obs] (n_obs = nº de anos
    com pelo menos uma observação válida).
    """
    df = pred_by_station_df[[station_col, time_col, value_col]].dropna(subset=[value_col]).copy()
    df["year"] = pd.to_datetime(df[time_col]).dt.year
    annual_max = df.groupby([station_col, "year"])[value_col].max().unstack("year")

    rows = []
    for code, row in annual_max.iterrows():
        xi_data = row.to_numpy(dtype=float)
        xi_data = xi_data[~np.isnan(xi_data)]
        if len(xi_data) == 0:
            continue
        rows.append({
            station_col: code,
            "p95": np.quantile(xi_data, 0.95),
            "p99": np.quantile(xi_data, 0.99),
            "n_obs": len(xi_data),
        })
    return pd.DataFrame(rows)


def load_ai_daily_results(results_paths: dict[str, str | Path]) -> pd.DataFrame:
    """Lê results.csv (schema `src/pipelines/metrics_schema.py`) de um ou mais
    experimentos de IA — domínio diário, R²/RMSE/Bias/Bias_P90/RMSE_P90,
    inalterado (não comparável aos métodos clássicos, ver docstring do módulo)."""
    frames = []
    for pipeline, path in results_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"[unified_metrics] [AVISO] {path} não existe — pulando '{pipeline}'.")
            continue
        df = pd.read_csv(path)
        df["family"] = "ai"
        df["pipeline"] = pipeline
        df["domain"] = "daily"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ai_annual_extremes(
    pred_paths: dict[str, str | Path],
    value_col: str = "y_pred",
    truth_col: str = "y_true",
    split_filter: str | None = None,
) -> pd.DataFrame:
    """Deriva P95/P99 de extremos anuais a partir das predições por estação
    de cada pipeline de IA (`predictions_by_station.csv` ou
    `spatial_holdout_predictions.csv`) e calcula bias/rmse/mae/corr contra o
    observado — MESMA função (`compute_validation_metrics`) usada pelo
    projeto clássico, para manter a régua estatística idêntica.

    `split_filter`: se dado (ex.: "test"), restringe a esse split antes de
    derivar os percentis anuais (só relevante para `predictions_by_station.csv`,
    que tem uma coluna `split`; `spatial_holdout_predictions.csv` já é
    implicitamente só holdout).
    """
    rows = []
    for pipeline, path in pred_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"[unified_metrics] [AVISO] {path} não existe — pulando '{pipeline}'.")
            continue
        df = pd.read_csv(path, parse_dates=["time"])
        if split_filter is not None and "split" in df.columns:
            df = df[df["split"] == split_filter]
        if df.empty:
            continue

        obs_pct = derive_annual_percentiles(df, value_col=truth_col)
        pred_pct = derive_annual_percentiles(df, value_col=value_col)
        merged = obs_pct.merge(pred_pct, on="estacao", suffixes=("_obs", "_pred"))
        if merged.empty:
            continue

        for pct in ("p95", "p99"):
            m = compute_validation_metrics(merged[f"{pct}_pred"], merged[f"{pct}_obs"])
            rows.append({
                "family": "ai", "pipeline": pipeline, "method": pipeline,
                "domain": "annual_extremes", "pct": pct, **m,
            })
    return pd.DataFrame(rows)


def load_classical_annual_extremes(loocv_metrics_csv: str | Path) -> pd.DataFrame:
    """Lê `loocv_consolidated_metrics.csv` (já produzido por
    `interpolation comparisson/loocv/run_loocv_comparison.py`) sem recomputar
    nada — só padroniza colunas para o schema unificado."""
    path = Path(loocv_metrics_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe — rode "
            "'interpolation comparisson/loocv/run_loocv_comparison.py' primeiro."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"method": "method"})
    df["family"] = "classical"
    df["pipeline"] = df["method"]
    df["domain"] = "annual_extremes"
    return df


def build_unified_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatena os frames de IA (derivado) e clássico (nativo) no domínio de
    extremos anuais — mesmas colunas (family, pipeline/method, domain, pct,
    bias, rmse, mae, corr, n)."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["family", "pipeline", "method", "domain", "pct", "bias", "rmse", "mae", "corr", "n"])
    cols = ["family", "pipeline", "method", "domain", "pct", "bias", "rmse", "mae", "corr", "n"]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for c in cols:
        if c not in combined.columns:
            combined[c] = np.nan
    return combined[cols + [c for c in combined.columns if c not in cols]]
