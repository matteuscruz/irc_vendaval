"""Ferramenta de XAI (src/xai/explain.py): SHAP e importância por ablação.

A ferramenta explica modelos já treinados — não treina nem altera dado. O que
estes testes protegem: a reconstrução da matriz de avaliação usa a partição
gravada NO ARTEFATO (não o padrão do código), a predição respeita o
`target_kind`, a ablação encontra a feature que de fato carrega o sinal, e
valores SHAP não-aditivos são recusados em vez de reportados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler

from src.pipeline.data.splits import resolve_month_block_split
from src.xai.explain import (
    EvalMatrix, ablation_importance, build_eval_matrix, load_artifact, predict,
    shap_summary, shap_values,
)

FEATS = ["wind_mag_max", "surface_pressure", "ruido"]


def _toy(n=600, seed=0, target_kind="absolute"):
    """Artefato mínimo no formato que as pipelines salvam, com um sinal
    conhecido: só `wind_mag_max` explica o alvo; `ruido` não explica nada."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        "wind_mag_max": rng.uniform(2, 18, n),
        "surface_pressure": rng.normal(101000, 400, n),
        "ruido": rng.normal(size=n),
    })
    era5 = x["wind_mag_max"].to_numpy()
    y = 1.6 * era5 + rng.normal(0, 0.4, n)

    scaler = RobustScaler().fit(x)
    target = y / np.clip(era5, 0.1, None) if target_kind == "ratio" else y
    model = ExtraTreesRegressor(n_estimators=40, random_state=seed).fit(scaler.transform(x), target)

    art = {
        "model": model, "model_name": "ExtraTreesRegressor", "scaler": scaler,
        "features": FEATS, "cluster_id": 9, "season": None,
        "target_kind": target_kind,
        "split": resolve_month_block_split(
            pd.date_range("2016-01-01", "2019-12-31", freq="D").values, seed=0,
        ).to_dict(),
        "climatology": {"method": "harmonic", "n_harmonics": 3},
    }
    data = EvalMatrix(x=x, y=y, era5=era5,
                      meta=pd.DataFrame({"estacao": "A", "time": pd.date_range("2020-01-01", periods=n)}))
    return art, data


# ── Predição ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["absolute", "ratio"])
def test_predict_reconstructs_target_in_ms(kind):
    art, data = _toy(target_kind=kind)
    p = predict(art, data.x, data.era5)

    assert p.shape == data.y.shape
    assert ((p >= 0) & (p <= 80)).all()          # clip físico da inferência
    assert np.corrcoef(p, data.y)[0, 1] > 0.9    # os dois caminhos recuperam m/s


def test_ratio_and_absolute_are_not_confused():
    """Aplicar a reconstrução errada mudaria a escala da predição."""
    art, data = _toy(target_kind="ratio")
    certo = predict(art, data.x, data.era5)
    art_errado = {**art, "target_kind": "absolute"}
    errado = predict(art_errado, data.x, data.era5)

    assert abs(certo.mean() - data.y.mean()) < 1.0
    assert abs(errado.mean() - data.y.mean()) > 5.0


# ── SHAP ────────────────────────────────────────────────────────────────────

def test_shap_ranks_the_informative_feature_first():
    art, data = _toy()
    vals, kind = shap_values(art, data, max_samples=200)
    resumo = shap_summary(vals)

    assert list(vals.columns) == FEATS
    assert "TreeExplainer" in kind
    assert resumo.index[0] == "wind_mag_max"
    assert resumo.loc["wind_mag_max", "share_pct"] > resumo.loc["ruido", "share_pct"] * 5
    assert resumo["share_pct"].sum() == pytest.approx(100.0)


def test_non_additive_shap_is_rejected():
    """Se base + soma(shap) não reconstrói a predição, os valores não são
    interpretáveis — tem de falhar, não ser reportado."""
    from src.xai.explain import _assert_additive, _estimator_and_transform

    art, data = _toy()
    estimator, to_model_space = _estimator_and_transform(art)
    x_model = to_model_space(data.x)

    class _ExplainerFalso:
        expected_value = 0.0

    zeros = np.zeros((len(x_model), len(FEATS)))
    with pytest.raises(ValueError, match="não aditivo"):
        _assert_additive(_ExplainerFalso(), estimator, x_model, zeros)


# ── Ablação ─────────────────────────────────────────────────────────────────

def test_ablation_finds_the_feature_that_carries_the_signal():
    art, data = _toy()
    abl = ablation_importance(art, data, strategy="mean")

    assert set(abl.index) == set(FEATS)
    assert abl.index[0] == "wind_mag_max"
    # Neutralizar a feature informativa destrói o modelo; a irrelevante quase
    # não muda nada. O que importa é a separação entre as duas, não um limiar
    # absoluto — a árvore usa a feature de ruído em alguns splits e move um
    # pouco a métrica mesmo sem ela carregar sinal.
    assert abl.loc["wind_mag_max", "d_R2"] < -0.5
    assert abl.loc["wind_mag_max", "d_RMSE"] > 10 * abs(abl.loc["ruido", "d_RMSE"])
    assert abl.attrs["strategy"] == "mean"


def test_shuffle_strategy_reports_spread_and_mean_does_not():
    art, data = _toy()
    media = ablation_importance(art, data, strategy="mean")
    embaralhado = ablation_importance(art, data, strategy="shuffle", n_repeats=3)

    assert (media["d_RMSE_std"] == 0).all()             # determinística
    assert (embaralhado["d_RMSE_std"] > 0).any()        # variabilidade entre repetições
    assert embaralhado.index[0] == "wind_mag_max"


def test_invalid_strategy_is_rejected():
    art, data = _toy()
    with pytest.raises(ValueError, match="strategy inválida"):
        ablation_importance(art, data, strategy="zero")


# ── Reconstrução da matriz ──────────────────────────────────────────────────

def test_eval_matrix_requires_the_split_recorded_in_the_artifact():
    art, _ = _toy()
    del art["split"]
    with pytest.raises(ValueError, match="sem 'split'"):
        build_eval_matrix("x", "y", art)


def test_eval_matrix_uses_the_artifacts_partition(synthetic_nf_raw_dir, synthetic_shp_dir):
    """A matriz tem de vir da partição do artefato: um modelo antigo precisa
    ser explicado com o dado que ele viu."""
    art, _ = _toy()
    art["features"] = ["wind_mag_max", "surface_pressure", "latitude"]
    art["cluster_id"] = 9

    data = build_eval_matrix(str(synthetic_nf_raw_dir), str(synthetic_shp_dir), art, split_label="test")

    assert list(data.x.columns) == art["features"]
    assert len(data.x) == len(data.y) == len(data.era5)
    assert data.x.notna().all().all()                    # sem imputação
    meses = set(pd.DatetimeIndex(data.meta["time"]).month)
    assert meses <= {1, 4, 7, 10}                        # só os meses de teste


def test_load_artifact_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nenhum artefato"):
        load_artifact(tmp_path, 3)
