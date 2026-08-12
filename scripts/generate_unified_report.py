"""Relatório estático IA vs. Métodos Clássicos — Tabela C do plano de
unificação, no domínio de extremos anuais (P95/P99). Reusa o mesmo template
visual de `interpolation comparisson/loocv/generate_loocv_report.py`.

Lê `artifacts/unified_comparison/<tag>/unified_metrics.csv`
(produzido por `scripts/compare_unified.py`).

Uso:
    python scripts/generate_unified_report.py --tag v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("artifacts/unified_comparison")


def _row_html(r, best_rmse):
    is_best = np.isclose(r["rmse"], best_rmse)
    cls = ' class="best"' if is_best else ""
    family_badge = "IA" if r["family"] == "ai" else "Clássico"
    return (
        f'<tr{cls}><td>{r["pipeline"]}</td><td>{family_badge}</td>'
        f'<td>{r["bias"]:+.2f}</td><td>{r["rmse"]:.2f}</td>'
        f'<td>{r["mae"]:.2f}</td><td>{r["corr"]:.2f}</td><td>{int(r["n"])}</td></tr>'
    )


def build_table_html(df: pd.DataFrame, caption: str) -> str:
    df = df.dropna(subset=["rmse"]).sort_values("rmse").reset_index(drop=True)
    if df.empty:
        return f"<p class='muted'>Sem dados para {caption}.</p>"
    best_rmse = df["rmse"].min()
    rows = "\n".join(_row_html(r, best_rmse) for _, r in df.iterrows())
    return f"""
    <table>
      <caption>{caption}</caption>
      <thead><tr><th>Método</th><th>Família</th><th>Bias (m/s)</th><th>RMSE (m/s)</th><th>MAE (m/s)</th><th>Corr</th><th>N</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    metrics_path = OUT_DIR / args.tag / "unified_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Rode scripts/compare_unified.py --tag {args.tag} primeiro — {metrics_path} não existe."
        )
    df = pd.read_csv(metrics_path)

    table_p99 = build_table_html(df[df["pct"] == "p99"], "Tabela C — IA vs. Métodos Clássicos (P99, extremos anuais)")
    table_p95 = build_table_html(df[df["pct"] == "p95"], "Tabela C — IA vs. Métodos Clássicos (P95, extremos anuais)")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório Unificado — IA vs. Métodos Clássicos</title>
<style>
  :root {{
    --bg: #07111c; --panel: #0d1e2e; --border: #1a2f40;
    --accent: #4da6d6; --text: #c8dde8; --muted: #6090a8; --heading: #d0e8f4;
    --best: #12331f; --best-border: #2f7a4a;
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
  tr.best {{ background: var(--best); }}
  tr.best td:first-child {{ border-left: 3px solid var(--best-border); }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  .note {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px 20px; font-size: 0.88rem; color: var(--muted); margin-bottom: 20px; }}
</style>
</head>
<body>
<header>
  <h1>IA vs. Métodos Clássicos de Interpolação — Extremos Anuais</h1>
  <p>Comparação restrita ao domínio onde os dois mundos são comparáveis: P95/P99 dos máximos
     anuais por estação (quantil empírico, sem GEV). O domínio diário (razão INMET/ERA5) fica
     de fora — métodos clássicos de interpolação espacial não operam nessa escala.</p>
</header>
<main>
  <div class="note">
    <strong>Metodologia:</strong> os percentis anuais das pipelines de IA são derivados das
    predições por estação (<code>predictions_by_station.csv</code> ou
    <code>spatial_holdout_predictions.csv</code>) com a MESMA definição usada pelo projeto
    clássico (<code>load_station_annual_percentiles</code>: groupby estação×ano → máximo anual →
    quantil empírico). As métricas (Bias/RMSE/MAE/Corr) usam a MESMA função
    (<code>compute_validation_metrics</code>) nos dois mundos. Se as predições de IA vierem de
    um experimento com <code>validation_mode=temporal</code> (estações já vistas em treino), a
    comparação favorece a IA — prefira predições de station-holdout (Gap 1) quando disponíveis.
  </div>

  <section>
    <h2>Tabela C — P99</h2>
    {table_p99}
  </section>

  <section>
    <h2>Tabela C — P95</h2>
    {table_p95}
  </section>
</main>
</body>
</html>"""

    out_path = OUT_DIR / args.tag / "unified_report.html"
    out_path.write_text(html)
    print(f"[generate_unified_report] Salvo: {out_path}")


if __name__ == "__main__":
    main()
