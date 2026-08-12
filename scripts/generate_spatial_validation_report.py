"""Relatório de validação cruzada espacial (station-holdout) das pipelines de
IA — análogo a `interpolation comparisson/loocv/generate_loocv_report.py`.

Lê o que `run_spatial_validation.py` produziu:
  - artifacts/spatial_validation/<tag>/consolidated_results.csv
  - artifacts/spatial_validation/<tag>/consolidated_predictions.csv

Duas tabelas:
  - Station-Holdout por pipeline × cluster: RMSE/Bias/R²/Bias_P90/RMSE_P90
    calculados sobre as predições POOLED de todos os folds (não a média das
    métricas por fold — mais estável estatisticamente).
  - "Gap de otimismo": RMSE do split temporal (test, 2024, todas as estações
    já vistas em treino) vs. RMSE do station-holdout (estações nunca vistas)
    lado a lado — quantifica diretamente o quanto o split temporal atual
    superestima a generalização espacial.

Uso:
    python scripts/generate_spatial_validation_report.py --tag holdout_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.common import compute_metrics

CONSOLIDATED_DIR = Path("artifacts/spatial_validation")


def _holdout_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipeline, cid), grp in predictions_df.groupby(["pipeline", "cluster_id"]):
        m = compute_metrics(grp["y_true"].to_numpy(float), grp["y_pred"].to_numpy(float))
        rows.append({
            "pipeline": pipeline,
            "cluster_id": cid,
            "n_folds": grp["fold_id"].nunique(),
            "n_stations_held_out": grp["estacao"].nunique(),
            "n_samples": len(grp),
            **m,
        })
    return pd.DataFrame(rows)


def _optimism_gap_table(results_df: pd.DataFrame, holdout_table: pd.DataFrame) -> pd.DataFrame:
    test_rows = results_df[
        (results_df["split"] == "test") & (results_df["season"].astype(str).isin(["ALL", "nan", ""]) | results_df["season"].isna())
    ]
    test_rmse = (
        test_rows.groupby(["pipeline", "cluster_id"])["RMSE"]
        .first()
        .reset_index()
        .rename(columns={"RMSE": "RMSE_temporal_test"})
    )
    merged = holdout_table.merge(test_rmse, on=["pipeline", "cluster_id"], how="left")
    merged = merged.rename(columns={"RMSE": "RMSE_spatial_holdout"})
    merged["gap_pct"] = (
        (merged["RMSE_spatial_holdout"] - merged["RMSE_temporal_test"])
        / merged["RMSE_temporal_test"].replace(0, np.nan) * 100
    )
    return merged[["pipeline", "cluster_id", "RMSE_temporal_test", "RMSE_spatial_holdout", "gap_pct", "n_stations_held_out"]]


def _row_html_holdout(r):
    return (
        f"<tr><td>{r['pipeline']}</td><td>{r['cluster_id']}</td>"
        f"<td>{r['R2']:.3f}</td><td>{r['RMSE']:.2f}</td><td>{r['Bias']:+.2f}</td>"
        f"<td>{r['Bias_P90']:+.2f}</td><td>{r['RMSE_P90']:.2f}</td>"
        f"<td>{int(r['n_folds'])}</td><td>{int(r['n_stations_held_out'])}</td><td>{int(r['n_samples'])}</td></tr>"
    )


def _row_html_gap(r):
    gap = r["gap_pct"]
    gap_str = f"{gap:+.1f}%" if pd.notna(gap) else "—"
    cls = ' class="worse"' if pd.notna(gap) and gap > 0 else ""
    rmse_test = f"{r['RMSE_temporal_test']:.2f}" if pd.notna(r["RMSE_temporal_test"]) else "—"
    return (
        f"<tr{cls}><td>{r['pipeline']}</td><td>{r['cluster_id']}</td>"
        f"<td>{rmse_test}</td><td>{r['RMSE_spatial_holdout']:.2f}</td>"
        f"<td>{gap_str}</td><td>{int(r['n_stations_held_out'])}</td></tr>"
    )


def _table_html(rows_html, caption, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f"""
    <table>
      <caption>{caption}</caption>
      <thead><tr>{head}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>"""


def build_html(holdout_table: pd.DataFrame, gap_table: pd.DataFrame, tag: str, validation_mode: str) -> str:
    holdout_rows = [_row_html_holdout(r) for _, r in holdout_table.sort_values(["pipeline", "RMSE"]).iterrows()]
    gap_rows = [_row_html_gap(r) for _, r in gap_table.sort_values(["pipeline", "cluster_id"]).iterrows()]

    holdout_html = _table_html(
        holdout_rows, f"Station-Holdout ({validation_mode}) por pipeline × cluster",
        ["Pipeline", "Cluster", "R²", "RMSE", "Bias", "Bias@P90", "RMSE@P90", "Folds", "Estações held-out", "N amostras"],
    )
    gap_html = _table_html(
        gap_rows, "Gap de Otimismo — RMSE(split temporal) vs. RMSE(station-holdout)",
        ["Pipeline", "Cluster", "RMSE temporal (test)", "RMSE spatial-holdout", "Δ (%)", "Estações held-out"],
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório de Validação Espacial — {tag}</title>
<style>
  :root {{
    --bg: #07111c; --panel: #0d1e2e; --border: #1a2f40;
    --accent: #4da6d6; --text: #c8dde8; --muted: #6090a8; --heading: #d0e8f4;
    --worse: #331414; --worse-border: #7a2f2f;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.6; }}
  header {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 32px 40px 24px; }}
  header h1 {{ color: var(--heading); font-size: 1.5rem; margin-bottom: 8px; }}
  header p {{ color: var(--muted); max-width: 900px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 40px 80px; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ color: var(--heading); font-size: 1.15rem; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); margin-bottom: 8px; }}
  caption {{ text-align: left; color: var(--muted); font-size: 0.85rem; padding: 8px 4px; caption-side: top; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
  tr.worse {{ background: var(--worse); }}
  tr.worse td:first-child {{ border-left: 3px solid var(--worse-border); }}
  .note {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px 20px; font-size: 0.88rem; color: var(--muted); margin-bottom: 20px; }}
</style>
</head>
<body>
<header>
  <h1>Validação Espacial (Station-Holdout) — {tag}</h1>
  <p>A IA generaliza para estações nunca vistas? Cada estação é excluída do treino e avaliada
     isoladamente, com climatologia populacional (cluster-mean, sem a própria estação) para
     evitar vazamento — análogo ao LOOCV do projeto de interpolação clássica.</p>
</header>
<main>
  <div class="note">
    <strong>Metodologia:</strong> modo <code>{validation_mode}</code>. O feature
    <code>era5_clim_wind</code> (climatologia por dia-do-ano) é recalculada por fold usando
    só as estações de treino (cluster-mean), nunca a estação held-out. O
    <code>cluster_lazy</code> reusa o estimador campeão já escolhido pelo split temporal em vez
    de repetir o screening completo por fold (não re-seleciona modelo, só valida o já escolhido).
    Linha destacada em vermelho na Tabela 2 = RMSE piora sob station-holdout (esperado —
    quantifica o quanto o split temporal atual superestimava a generalização espacial).
  </div>

  <section>
    <h2>Tabela 1 — Métricas de Station-Holdout</h2>
    {holdout_html}
  </section>

  <section>
    <h2>Tabela 2 — Gap de Otimismo (Temporal vs. Spatial-Holdout)</h2>
    {gap_html}
  </section>
</main>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--validation-mode", default="spatial-kfold",
                         help="Só usado no texto do relatório (metadado, não recomputa nada)")
    args = parser.parse_args()

    run_dir = CONSOLIDATED_DIR / args.tag
    results_path = run_dir / "consolidated_results.csv"
    predictions_path = run_dir / "consolidated_predictions.csv"
    if not results_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            f"Rode scripts/run_spatial_validation.py --tag {args.tag} primeiro — "
            f"{results_path} e/ou {predictions_path} não existem."
        )

    results_df = pd.read_csv(results_path)
    predictions_df = pd.read_csv(predictions_path)

    holdout_table = _holdout_table(predictions_df)
    gap_table = _optimism_gap_table(results_df, holdout_table)

    html = build_html(holdout_table, gap_table, args.tag, args.validation_mode)
    out_path = run_dir / "spatial_validation_report.html"
    out_path.write_text(html)
    print(f"[generate_spatial_validation_report] Salvo: {out_path}")


if __name__ == "__main__":
    main()
