"""Mapeamento de quantis (Empirical Quantile Mapping, EQM) sobre o modelo PADRÃO, sem novo treino, e nova
inferência na grade. O método e suas ressalvas: `src/feature_study/diagnostics/quantile_map.py` e
`documentation/11_mapeamento_de_quantis.md`.

    uv run python scripts/calibrar_quantis.py

Por trimestre: previsão do padrão na VALIDAÇÃO → mapa de quantis para o INMET; comparação padrão × mapeado
na validação e no TESTE; aplicação à grade. Gera `artifacts/feature_study/inferencia/results_feature_selction_qmap.nc`
e `qmap_comparacao.csv`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.inferencia_selecao_features import GRADE, GRUPO4, LOCAL, MODAL, colunas_selecionadas  # noqa: E402
from scripts.otimizar_extremos import CACHE_GRADE, DIR  # noqa: E402
from src.feature_study.core.worker import load_unit_frames  # noqa: E402
from src.feature_study.diagnostics import inference as inf  # noqa: E402
from src.feature_study.diagnostics import quantile_map as qm  # noqa: E402
from src.feature_study.diagnostics import tuning as tn  # noqa: E402
from src.feature_study.diagnostics.predictor_curves import champions  # noqa: E402
from src.pipelines.common import TARGET_VAR  # noqa: E402


def main() -> None:
    cols = colunas_selecionadas()
    fit = inf.fit_champions(LOCAL / "data", champions(MODAL / "summary"), cols)
    mapas, linhas = {}, []
    for s, f in fit.items():
        train, val, test = load_unit_frames(LOCAL / "data", "full", s)
        y_tr = train[TARGET_VAR].to_numpy(float)
        lim95, lim99 = float(np.quantile(y_tr, 0.95)), float(np.quantile(y_tr, 0.99))
        p_va = inf.predict_grid(val, {s: f}, cols)
        p_te = inf.predict_grid(test, {s: f}, cols)
        mapas[s] = qm.fit(p_va, val[TARGET_VAR])
        for split, d, p in (("val", val, p_va), ("teste", test, p_te)):
            y = d[TARGET_VAR].to_numpy(float)
            for versao, pr in (("padrão", p), ("mapeado", qm.apply(p, mapas[s]))):
                linhas.append({"season": s, "model": f["model"], "split": split, "versao": versao,
                               "objetivo": tn.objective_value(y, pr), "std_pred": float(np.std(pr)),
                               "std_obs": float(np.std(y)), "corr": float(np.corrcoef(pr, y)[0, 1]),
                               **tn.tail_metrics(y, pr, lim95, lim99)})
    comp = pd.DataFrame(linhas)
    comp.to_csv(DIR / "qmap_comparacao.csv", index=False)
    print(comp[["season", "split", "versao", "RMSE", "corr", "std_pred", "std_obs", "P99_pred", "P99_obs",
                "POD_P95", "FAR_P95", "POD_P99", "FAR_P99"]].round(3).to_string(index=False))

    grade = pd.read_parquet(CACHE_GRADE) if CACHE_GRADE.exists() else inf.daily_peak_rows(GRADE, cols)
    g4 = pd.read_parquet(GRUPO4, columns=["grid_lat", "grid_lon"])
    celulas = set(zip(g4.grid_lat.round(2), g4.grid_lon.round(2)))
    grade["amostra"] = inf.amostra(grade["time"], pd.Series([(round(a, 2), round(b, 2)) in celulas
                                                            for a, b in zip(grade.lat, grade.lon)]))
    grade["era5_gust_max"] = grade[inf.PEAK].astype("float32")
    bruto = inf.predict_grid(grade, fit, cols)
    pred = np.full(len(grade), np.nan, dtype="float32")
    for s, mapa in mapas.items():
        m = (grade["season"] == s).to_numpy()
        pred[m] = qm.apply(bruto[m], mapa)
    grade["pred_gust_max"] = pred
    attrs = {"titulo": "Modelo padrão + mapeamento de quantis para o INMET (ajustado na validação)",
             "modelos": json.dumps({s: f["model"] for s, f in fit.items()}),
             "colunas": ",".join(cols), "fonte": GRADE.name}
    out = inf.write_netcdf(inf.to_dataset(grade, attrs), DIR / "results_feature_selction_qmap.nc")
    print(f"\ngravado: {out}")


if __name__ == "__main__":
    main()
