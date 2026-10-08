"""Juntar a LSTM e os modelos tabulares (`diagnostics/all_models.py`).

Estudo sintético com resposta conhecida: o modelo A melhora com o `full`, a LSTM piora.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_study.core.arms import Arm
from src.feature_study.diagnostics import all_models as am

N_ANOS, POR_BLOCO = 20, 25


def _estudo(tmp_path):
    rng = np.random.default_rng(0)
    linhas = []
    for ano in range(2000, 2000 + N_ANOS):
        linhas.append(pd.DataFrame({"year": ano, "month": 1, "season": "DJF", "estacao": "A",
                                    "daily_wind_gust_max": rng.gamma(4, 2, POR_BLOCO)}))
    test = pd.concat(linhas, ignore_index=True)
    test["row_id"] = np.arange(len(test))
    test["era5_gust_max"] = test["daily_wind_gust_max"] + rng.normal(0, 3.0, len(test))
    data = tmp_path / "data"
    data.mkdir()
    test.to_parquet(data / "test.parquet", index=False)
    arms = [Arm("base", ("x",), "base", "anchors"), Arm("full", ("x", "y"), "full", "anchors")]
    (data / "arms.json").write_text(json.dumps([a.to_dict() for a in arms]))

    def unidade(tag, modelo, sd_base, sd_full, seed=42):
        pasta = tmp_path / "units" / tag
        pasta.mkdir(parents=True, exist_ok=True)
        shared = rng.standard_normal(len(test))
        for arm, sd in (("base", sd_base), ("full", sd_full)):
            e = sd * (0.9 * shared + 0.44 * rng.standard_normal(len(test)))
            pd.DataFrame({"row_id": test["row_id"], modelo: e.astype("float32")}).to_parquet(
                pasta / f"resid__DJF__{arm}.parquet", index=False)
            pd.DataFrame([{"arm": arm, "season": "DJF", "tag": "full", "model_seed": seed, "model": modelo, "split": "test",
                           "n": len(test), "R2": 0.5, "RMSE": float(np.sqrt((e ** 2).mean())),
                           "RMSE_P90": 5.0, "Bias": 0.0}]).to_parquet(pasta / f"metrics__DJF__{arm}.parquet", index=False)

    unidade("full", "LGBMRegressor", 2.0, 1.5)
    unidade("full_s43", "LGBMRegressor", 2.0, 1.5, seed=43)
    unidade("full_lstm", "LSTM", 2.0, 2.6)
    return tmp_path


def test_families_are_labelled_and_the_lstm_is_its_own_family():
    assert am.familia("LSTM") == "LSTM" and am.familia("CatBoostRegressor") == "boosting"
    assert am.familia("ModeloQueNaoExiste") == "outros"


def test_every_model_gets_its_error_next_to_the_era5_on_the_same_season(tmp_path):
    estudo = _estudo(tmp_path)
    ps = am.per_season(am.load_metrics(estudo, ["full", "full_s43", "full_lstm"]), am.era5_by_season(estudo / "data"))
    assert set(ps["model"]) == {"LGBMRegressor", "LSTM"} and set(ps["familia"]) == {"boosting", "LSTM"}
    assert (ps["era5_rmse"] > 2.5).all()
    assert ps.loc[ps.model == "LGBMRegressor", "n_seeds"].eq(2).all()                  # média entre as 2 seeds
    assert ps.loc[ps.model == "LSTM", "n_seeds"].eq(1).all()


def test_the_effect_of_adding_features_is_measured_per_model_with_its_own_sign(tmp_path):
    estudo = _estudo(tmp_path)
    ef = am.effects_by_model(estudo, {"tabulares": ["full", "full_s43"], "LSTM": ["full_lstm"]}, n_boot=200)
    por = ef.set_index("model")
    assert por.loc["LGBMRegressor", "efeito_full_vs_base"] > 0.3 and por.loc["LGBMRegressor", "ci_lo"] > 0
    assert por.loc["LSTM", "efeito_full_vs_base"] < -0.3 and por.loc["LSTM", "ci_hi"] < 0
    assert list(ef["efeito_full_vs_base"]) == sorted(ef["efeito_full_vs_base"])        # do pior para o melhor


def test_run_writes_the_three_tables_and_the_best_tree_is_found(tmp_path):
    estudo = _estudo(tmp_path)
    r = am.run(estudo, ["full", "full_s43"], ["full_lstm"], n_boot=100)
    for nome in ("erro_por_modelo_arm_trimestre", "resumo_por_modelo_arm", "efeito_full_vs_base_por_modelo"):
        assert (r["out"] / f"{nome}.csv").exists()
    melhor = am.best_tree_by_season(r["por_trimestre"], "base")
    assert melhor.loc["DJF", "model"] == "LGBMRegressor"
    assert r["resumo"].query("model == 'LSTM'")["n_trimestres"].eq(1).all()


def test_missing_units_fail_loudly_instead_of_returning_an_empty_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="nenhuma métrica"):
        am.load_metrics(tmp_path, ["nao_existe"])
