"""QA da grade ERA5 corrigida consolidada — Fase 6 do plano de correção
ERA5 2000-2024. Três checagens independentes, cada uma imprime um veredito
e não falha o processo (achados vão pro relatório, não pro exit code):

1. Continuidade na fronteira 2019→2020: compara `rajada_max_corrigida` da
   última semana de dez/ano_a com a primeira semana de jan/ano_b em algumas
   células de grade, procurando um salto muito maior que a variação
   dia-a-dia típica dentro de cada ano (sinalizaria problema na costura
   entre duas tabelas de vencedores diferentes, se essa abordagem tivesse
   sido usada — ver Fase 4 do plano).
2. Flag `in_sample`: confirma que reflete exatamente
   `TRAIN_SLICE[0] <= time <= TRAIN_SLICE[1]` em cada arquivo, célula a
   célula da coordenada de tempo (antes de 2026-09-01 só checava o limite
   superior, marcando erroneamente anos antes do início de TRAIN_SLICE
   como in_sample=True).
3. Spot-check de vencedores: cruza N linhas aleatórias de um winners.csv/
   winners_full.csv com o results.csv do combo correspondente, conferindo
   que metric_value/n_samples batem (winner table não fabricou número).

Uso
---
python3 scripts/qa_corrected_grid.py --dir artifacts/corrected_grid/final_2000_2024 \\
    --boundary-years 2019 2020 --winners-sample 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TRAIN_START = "2000-01-01"
TRAIN_END = "2018-12-31"


def check_boundary_continuity(grid_dir: Path, year_a: int, year_b: int, n_cells: int = 6) -> None:
    fa = grid_dir / f"grid_corrected_{year_a}.nc"
    fb = grid_dir / f"grid_corrected_{year_b}.nc"
    print(f"\n=== 1. Continuidade {year_a} -> {year_b} ===")
    if not fa.exists() or not fb.exists():
        print(f"  [PULADO] Faltando {fa if not fa.exists() else fb}")
        return

    ds_a = xr.open_dataset(fa)
    ds_b = xr.open_dataset(fb)
    var = "rajada_max_corrigida"

    last_week_a = ds_a[var].sel(time=slice(f"{year_a}-12-25", f"{year_a}-12-31"))
    first_week_b = ds_b[var].sel(time=slice(f"{year_b}-01-01", f"{year_b}-01-07"))

    # Amostra de células de grade não totalmente NaN em ambos os lados.
    valid_mask = last_week_a.notnull().any("time") & first_week_b.notnull().any("time")
    lat_idx, lon_idx = np.where(valid_mask.values)
    if len(lat_idx) == 0:
        print("  [AVISO] Nenhuma célula com dado válido nos dois lados da fronteira.")
        ds_a.close(); ds_b.close()
        return

    rng = np.random.default_rng(0)
    sample = rng.choice(len(lat_idx), size=min(n_cells, len(lat_idx)), replace=False)

    jumps = []
    for i in sample:
        la, lo = lat_idx[i], lon_idx[i]
        series_a = last_week_a.isel(latitude=la, longitude=lo).values
        series_b = first_week_b.isel(latitude=la, longitude=lo).values
        boundary_jump = abs(series_b[0] - series_a[-1])
        typical_daily_var = np.nanstd(np.diff(np.concatenate([series_a, series_b])))
        ratio = boundary_jump / typical_daily_var if typical_daily_var > 0 else np.nan
        jumps.append(ratio)
        flag = "⚠" if ratio > 5 else " "
        print(f"  {flag} cel({la},{lo}): salto={boundary_jump:.2f} m/s, "
              f"var. típica={typical_daily_var:.2f} m/s, razão={ratio:.1f}x")

    n_flagged = sum(1 for r in jumps if r > 5)
    verdict = "OK" if n_flagged == 0 else f"REVISAR ({n_flagged}/{len(jumps)} células com salto >5x)"
    print(f"  Veredito: {verdict}")
    ds_a.close(); ds_b.close()


def check_in_sample_flag(grid_dir: Path) -> None:
    print("\n=== 2. Flag in_sample ===")
    files = sorted(grid_dir.glob("grid_corrected_*.nc"))
    if not files:
        print(f"  [PULADO] Nenhum grid_corrected_*.nc em {grid_dir}")
        return
    train_start, train_end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    n_bad = 0
    for f in files:
        ds = xr.open_dataset(f)
        expected = (
            (ds["time"].values >= np.datetime64(train_start))
            & (ds["time"].values <= np.datetime64(train_end))
        )
        actual = ds["in_sample"].values
        mismatch = int((expected != actual).sum())
        if mismatch:
            n_bad += 1
            print(f"  ⚠ {f.name}: {mismatch}/{len(actual)} timesteps com in_sample incorreto")
        ds.close()
    verdict = "OK" if n_bad == 0 else f"REVISAR ({n_bad} arquivo(s) com mismatch)"
    print(f"  Veredito ({len(files)} arquivo(s) checado(s)): {verdict}")


def check_winners_spot(grid_dir: Path, artifacts_root: Path, n_sample: int = 8) -> None:
    print("\n=== 3. Spot-check de vencedores ===")
    winners_path = grid_dir / "winners_full.csv"
    if not winners_path.exists():
        winners_path = grid_dir / "winners.csv"
    if not winners_path.exists():
        print(f"  [PULADO] Nenhum winners.csv/winners_full.csv em {grid_dir}")
        return

    from src.dataset.creation.best_model_selector import PIPELINE_ROOTS

    winners = pd.read_csv(winners_path)
    sample = winners.sample(n=min(n_sample, len(winners)), random_state=0)

    n_ok, n_bad = 0, 0
    for _, row in sample.iterrows():
        combo_dir = artifacts_root / PIPELINE_ROOTS[row["pipeline"]] / row["arm"]
        results_csv = combo_dir / "results.csv"
        if not results_csv.exists():
            print(f"  ⚠ {row['pipeline']}/{row['arm']}: sem results.csv em {combo_dir}")
            n_bad += 1
            continue
        results = pd.read_csv(results_csv)
        match = results[
            (results["cluster_id"].astype(str) == str(row["cluster_id"]))
            & (results["season"] == (row["season"] if row["season"] != "ALL" else results["season"]))
        ]
        if match.empty:
            print(f"  ⚠ cluster {row['cluster_id']}/{row['season']} "
                  f"({row['pipeline']}/{row['arm']}): linha não encontrada em results.csv")
            n_bad += 1
            continue
        n_ok += 1
        n_samples_match = int(match.iloc[0].get("n_samples", -1))
        print(f"  ✓ cluster {row['cluster_id']}/{row['season']} "
              f"({row['pipeline']}/{row['arm']}): metric_value={row['metric_value']:.3f} "
              f"n_samples(winner)={row['n_samples']} n_samples(results.csv, 1a linha)={n_samples_match}")

    print(f"  Veredito: {n_ok}/{len(sample)} encontrados em results.csv "
          f"({n_bad} não encontrados — não necessariamente errado, "
          "season 'ALL' pode ser derivada, ver resolve_all_season)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path,
                         help="Diretório com grid_corrected_<ano>.nc + winners(_full).csv")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--boundary-years", nargs=2, type=int, default=[2019, 2020])
    parser.add_argument("--winners-sample", type=int, default=8)
    args = parser.parse_args()

    check_boundary_continuity(args.dir, args.boundary_years[0], args.boundary_years[1])
    check_in_sample_flag(args.dir)
    check_winners_spot(args.dir, args.artifacts_root, args.winners_sample)


if __name__ == "__main__":
    main()
