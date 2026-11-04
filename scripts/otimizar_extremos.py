"""Busca de hiperparâmetros (Optuna) do campeão de cada trimestre, mirando os extremos, e nova inferência na grade.

    uv run python scripts/otimizar_extremos.py --trials 40

1. Para cada trimestre: reajusta o campeão padrão (igual ao estudo), roda o Optuna na VALIDAÇÃO com objetivo
   0,5·RMSE + 0,5·RMSE_P90 e compara padrão × otimizado em val e TESTE (o teste só é usado aqui).
2. Aplica os otimizados à grade e grava `artifacts/feature_study/inferencia/results_feature_selction_optuna.nc`,
   no mesmo formato de `results_feature_selction.nc`, para o notebook comparar os dois.
Saídas ao lado: `optuna_params.json`, `optuna_comparacao.csv`, `optuna_trials__<trim>.csv`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.inferencia_selecao_features import (  # noqa: E402
    GRADE, GRUPO4, LOCAL, MODAL, colunas_selecionadas,
)
from src.feature_study.core.worker import load_unit_frames  # noqa: E402
from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics import tuning as tn  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402
from src.pipelines.common import TARGET_VAR, preprocess_df  # noqa: E402

DIR = RAIZ / "artifacts/feature_study/inferencia"
CACHE_GRADE = DIR / "_grade_diaria_pico.parquet"


def main(trials: int, timeout: int | None, amplo: bool = False, sufixo: str = "") -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    cols = colunas_selecionadas()
    campeao = champions(MODAL / "summary")
    padrao = inf.fit_champions(LOCAL / "data", campeao, cols)

    tuned, comparacoes, params_out = {}, [], {}
    for s, f in padrao.items():
        train, val, test = load_unit_frames(LOCAL / "data", "full", s)
        x_tr, x_va = preprocess_df(train[cols], val[cols])
        _, x_te = preprocess_df(train[cols], test[cols])
        y_tr, y_va, y_te = (d[TARGET_VAR].to_numpy(float) for d in (train, val, test))
        lim95, lim99 = float(np.quantile(y_tr, 0.95)), float(np.quantile(y_tr, 0.99))

        t0 = time.time()
        study, params, peso, lim_cauda = tn.tune(f["pipe"], f["model"], x_tr, y_tr, x_va, y_va,
                                                 n_trials=trials, timeout=timeout, amplo=amplo)
        pipe = tn.fit_variant(f["pipe"], x_tr, y_tr, params, peso, lim_cauda)
        print(f"{s} {f['model']}: {len(study.trials)} tentativas em {time.time() - t0:.0f}s | "
              f"melhor objetivo val {study.best_value:.3f} | peso_cauda {peso:.2f} | {params}", flush=True)

        c = tn.compare(f["pipe"], pipe, {"val": (x_va, y_va), "teste": (x_te, y_te)}, lim95, lim99)
        comparacoes.append(c.assign(season=s, model=f["model"]))
        study.trials_dataframe().to_csv(DIR / f"optuna{sufixo}_trials__{s}.csv", index=False)
        params_out[s] = {"model": f["model"], "peso_cauda": peso, "lim_cauda_treino": lim_cauda,
                         "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                         "objetivo_val": study.best_value, "n_trials": len(study.trials)}
        tuned[s] = {"model": f["model"], "pipe": pipe, "train": f["train"]}

    comp = pd.concat(comparacoes, ignore_index=True)
    comp.to_csv(DIR / f"optuna{sufixo}_comparacao.csv", index=False)
    (DIR / f"optuna{sufixo}_params.json").write_text(json.dumps(params_out, indent=1, default=str))
    cols_show = ["season", "model", "split", "versao", "objetivo", "RMSE", "RMSE_P90", "vies_P99", "P99_pred",
                 "P99_obs", "POD_P95", "FAR_P95", "POD_P99", "FAR_P99"]
    print("\nPadrão × otimizado (o TESTE é a prova; a validação foi usada na busca):")
    print(comp[cols_show].round(3).to_string(index=False))

    if CACHE_GRADE.exists():
        grade = pd.read_parquet(CACHE_GRADE)
    else:
        print("lendo a grade (por grupo de linhas)...", flush=True)
        grade = inf.daily_peak_rows(GRADE, cols)
        grade.to_parquet(CACHE_GRADE, index=False)
    g4 = pd.read_parquet(GRUPO4, columns=["grid_lat", "grid_lon"])
    celulas = set(zip(g4.grid_lat.round(2), g4.grid_lon.round(2)))
    grade["amostra"] = inf.amostra(grade["time"], pd.Series([(round(a, 2), round(b, 2)) in celulas
                                                            for a, b in zip(grade.lat, grade.lon)]))
    grade["era5_gust_max"] = grade[inf.PEAK].astype("float32")
    grade["pred_gust_max"] = inf.predict_grid(grade, tuned, cols)
    attrs = {"titulo": "Inferência com o campeão de cada trimestre OTIMIZADO no Optuna (extremos)",
             "modelos": json.dumps({s: t["model"] for s, t in tuned.items()}),
             "optuna": json.dumps(params_out, default=str),
             "objetivo": f"{1 - tn.PESO_CAUDA}·RMSE + {tn.PESO_CAUDA}·RMSE_P90 na validação",
             "colunas": ",".join(cols), "fonte": GRADE.name}
    out = inf.write_netcdf(inf.to_dataset(grade, attrs), DIR / f"results_feature_selction_optuna{sufixo}.nc")
    print(f"\ngravado: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=900, help="segundos por trimestre")
    ap.add_argument("--amplo", action="store_true", help="espaço de busca estendido")
    ap.add_argument("--sufixo", default="", help="sufixo dos arquivos de saída (não sobrescreve outra busca)")
    a = ap.parse_args()
    main(a.trials, a.timeout, a.amplo, a.sufixo)
