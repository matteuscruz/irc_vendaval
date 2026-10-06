"""Triagem top-5 por trimestre (`analysis.top_models_by_season`) e a impressão
digital de conjuntos de modelos (`worker._models_fingerprint`).

A triagem substitui rodar os 39 regressores em todos os arms: os 39 rodam uma
vez só, no arm `base`, e os 5 melhores de cada trimestre ficam congelados num
JSON que o estudo inteiro passa a usar.

Leaderboards montados à mão — nenhum modelo é ajustado aqui, então a resposta
certa é conhecida de antemão e a suíte roda em menos de um segundo.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_study.core.analysis import top_models_by_season, top_models_payload
from src.feature_study.core.worker import _models_fingerprint, select_regressors

SEASONS = ("DJF", "MAM", "JJA", "SON")


def _leaderboard(tmp_path, linhas, tag="r42"):
    """Grava um `units/<tag>/metrics__*.parquet` no formato que
    `models_leaderboard` lê. `linhas` = (model, split, R2, RMSE_P90)."""
    unidade = tmp_path / "units" / tag
    unidade.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        df = pd.DataFrame([
            {"arm": "base", "season": season, "tag": tag, "model": m, "split": sp,
             "R2": r2, "RMSE": 2.0, "Bias": 0.0, "Bias_P90": -3.0, "RMSE_P90": p90,
             "fit_seconds": 1.0, "n_nonfinite": 0, "model_seed": 42}
            for m, sp, r2, p90 in linhas
        ])
        df.to_parquet(unidade / f"metrics__{season}__base.parquet", index=False)
    return tmp_path


# ── A regra de escolha ──────────────────────────────────────────────────────

def test_top_models_prefers_lower_rmse_p90_among_models_within_the_r2_slack(tmp_path):
    """Maximizar R² é minimizar erro quadrático, cujo minimizador é a média
    condicional: a seleção por R² puro prefere sistematicamente o modelo mais
    COMPRIMIDO — o oposto do que um produto de extremos precisa. A folga
    corrige isso sem aceitar um modelo ruim na média.

    A (R²=0,80 / P90=5,0) é o melhor R²; B (0,77 / 3,0) está dentro da folga de
    0,05 e tem cauda muito melhor ⇒ vence. C (0,60 / 1,0) tem a melhor cauda de
    todas mas está fora da folga ⇒ não pode liderar.
    """
    linhas = [("A", "val", 0.80, 5.0), ("B", "val", 0.77, 3.0), ("C", "val", 0.60, 1.0)]
    linhas += [(m, "test", r2 - 0.02, p90 + 0.1) for m, _, r2, p90 in linhas]
    _leaderboard(tmp_path, linhas)

    top = top_models_by_season(tmp_path, ["r42"], k=3, slack=0.05)
    djf = top[top["season"] == "DJF"].sort_values("rank")
    assert list(djf["model"]) == ["B", "A", "C"]


def test_selection_happens_on_validation_while_test_metrics_are_only_reported(tmp_path):
    """Escolher no teste seria medir o próprio teste. Aqui o teste é invertido
    de propósito: se a escolha olhasse para ele, o vencedor seria outro."""
    linhas = [
        ("A", "val", 0.80, 3.0), ("B", "val", 0.79, 9.0),
        ("A", "test", 0.10, 9.0), ("B", "test", 0.90, 1.0),   # teste diz o contrário
    ]
    _leaderboard(tmp_path, linhas)

    top = top_models_by_season(tmp_path, ["r42"], k=1, slack=0.05)
    assert set(top["model"]) == {"A"}                       # escolhido pela validação
    assert (top["RMSE_P90_test"] == 9.0).all()              # mas o teste é reportado
    assert (top["R2_test"] == 0.10).all()


def test_top_models_returns_k_models_even_when_the_slack_admits_fewer(tmp_path):
    """Um trimestre com menos modelos que os outros desbalancearia a média por
    trimestre da análise — melhor um quinto modelo medíocre do que um buraco."""
    linhas = [("A", "val", 0.80, 3.0), ("B", "val", 0.20, 1.0), ("C", "val", 0.10, 2.0)]
    linhas += [(m, "test", r2, p90) for m, _, r2, p90 in linhas]
    _leaderboard(tmp_path, linhas)

    top = top_models_by_season(tmp_path, ["r42"], k=3, slack=0.05)
    djf = top[top["season"] == "DJF"]
    assert len(djf) == 3
    assert list(djf.sort_values("rank")["model"]) == ["A", "B", "C"]   # completou por R²


def test_models_that_diverged_never_enter_the_screening(tmp_path):
    """Um modelo que divergiu no arm base voltaria a divergir nos outros e
    levaria um resultado inteiro consigo — melhor descartá-lo na triagem."""
    _leaderboard(tmp_path, [("A", "val", 0.80, 3.0), ("RUIM", "val", 0.99, 0.1),
                            ("A", "test", 0.78, 3.1), ("RUIM", "test", 0.99, 0.1)])
    caminho = tmp_path / "units" / "r42" / "metrics__DJF__base.parquet"
    df = pd.read_parquet(caminho)
    df.loc[df["model"] == "RUIM", "n_nonfinite"] = 7          # divergiu em 7 linhas
    df.to_parquet(caminho, index=False)

    djf = top_models_by_season(tmp_path, ["r42"], k=2)
    djf = djf[djf["season"] == "DJF"]
    assert "RUIM" not in set(djf["model"])


def test_the_val_to_test_gap_is_reported_because_controls_do_not_see_overfitting(tmp_path):
    """`ctrl__noise` e `ctrl__perm_static` detectam ganho espúrio de FEATURE.
    Com ~680 colunas e 9 estações, o modelo pode identificar a estação pelo
    próprio perfil horário — sobreajuste global que nenhum dos dois enxerga. O
    gap val→teste é o termômetro que falta."""
    _leaderboard(tmp_path, [("A", "val", 0.90, 3.0), ("A", "test", 0.40, 5.0)])
    top = top_models_by_season(tmp_path, ["r42"], k=1)
    assert np.allclose(top["gap_R2_val_test"], 0.50)


def test_empty_leaderboard_returns_an_empty_frame_instead_of_raising(tmp_path):
    assert top_models_by_season(tmp_path, ["r42"]).empty


# ── O JSON congelado ────────────────────────────────────────────────────────

def test_payload_freezes_one_model_list_per_season_in_rank_order(tmp_path):
    """Congelar em vez de recalcular é deliberado: se cada unidade refizesse a
    escolha, uma seed a mais no volume mudaria o conjunto de modelos no meio do
    fan-out e metade dos arms sairia com modelos diferentes da outra metade."""
    linhas = [("A", "val", 0.80, 5.0), ("B", "val", 0.77, 3.0)]
    linhas += [(m, "test", r2, p90) for m, _, r2, p90 in linhas]
    _leaderboard(tmp_path, linhas)

    top = top_models_by_season(tmp_path, ["r42"], k=2, slack=0.05)
    payload = top_models_payload(top, tags=["r42"], k=2)

    assert set(payload["by_season"]) == set(SEASONS)
    assert payload["by_season"]["DJF"] == ["B", "A"]             # ordem de rank
    assert payload["rule"] == "r2_slack_then_rmse_p90"
    assert payload["split_select"] == "val"
    json.loads(json.dumps(payload))                               # serializável


# ── Impressão digital do conjunto de modelos ────────────────────────────────

def test_models_fingerprint_differs_for_different_sets_so_skip_existing_cannot_reuse(tmp_path):
    """A armadilha que isto fecha: sem a impressão digital, o top-5 do DJF e o
    do JJA carimbam ambos a string literal `"custom"`, `_is_current` os
    considera equivalentes e o `skip_existing` devolve o ajuste ANTIGO como se
    fosse o novo — sem erro, sem aviso, com métricas plausíveis."""
    a = _models_fingerprint(["Ridge", "XGBRegressor"])
    b = _models_fingerprint(["Ridge", "LGBMRegressor"])
    assert a != b


def test_models_fingerprint_is_order_independent_and_stable_across_processes():
    """O fan-out do Modal monta a lista em containers diferentes; se a ordem
    mudasse a impressão digital, cada container refaria o ajuste do outro."""
    assert _models_fingerprint(["Ridge", "XGBRegressor"]) == _models_fingerprint(["XGBRegressor", "Ridge"])
    assert _models_fingerprint(["Ridge"]) == "e9eeca48aa5c"       # valor fixo: estável entre processos


def test_fingerprint_accepts_both_names_and_classes():
    """A triagem entrega nomes (vindos do JSON); o `run_unit` tem classes."""
    from sklearn.linear_model import Ridge
    assert _models_fingerprint([Ridge]) == _models_fingerprint(["Ridge"])


# ── `select_regressors` com lista de nomes ──────────────────────────────────

def test_select_regressors_accepts_an_explicit_list_of_model_names():
    escolhidos = select_regressors(["Ridge", "ExtraTreesRegressor"])
    assert sorted(c.__name__ for c in escolhidos) == ["ExtraTreesRegressor", "Ridge"]


def test_select_regressors_raises_on_an_unknown_model_name():
    """Um erro de digitação devolveria lista vazia e o LazyRegressor treinaria
    ZERO modelos sem reclamar — o parquet sairia vazio e só apareceria como
    resultado faltando na análise, horas e vários containers depois."""
    with pytest.raises(ValueError, match="modelos ausentes"):
        select_regressors(["Ridge", "NãoExisteRegressor"])
    with pytest.raises(ValueError, match="lista de modelos vazia"):
        select_regressors([])


def test_select_regressors_preserves_pool_order_not_rank_order():
    """Determinismo: a ordem em que o LazyPredict recebe os modelos não pode
    depender do ranking da triagem, senão o mesmo conjunto rodaria diferente
    conforme o trimestre."""
    a = [c.__name__ for c in select_regressors(["Ridge", "ExtraTreesRegressor"])]
    b = [c.__name__ for c in select_regressors(["ExtraTreesRegressor", "Ridge"])]
    assert a == b


# ── Modelos equivalentes contam uma vez ─────────────────────────────────────

def test_models_with_identical_validation_metrics_take_only_one_slot(tmp_path):
    """Na triagem real de JJA, `LinearRegression`, `TransformedTargetRegressor`,
    `RidgeCV` e `Ridge` saíram com R² e RMSE_P90 iguais até a terceira casa: são
    a mesma regressão linear, e ocupavam quatro das cinco vagas. "Cinco modelos"
    eram duas ideias."""
    linhas = [
        ("Linear", "val", 0.738, 3.430), ("Ridge", "val", 0.738, 3.433),
        ("RidgeCV", "val", 0.738, 3.433), ("Cat", "val", 0.772, 3.404),
        ("Floresta", "val", 0.735, 3.900), ("Arvore", "val", 0.730, 4.100),
    ]
    linhas += [(m, "test", r2 - 0.03, p90 + 0.2) for m, _, r2, p90 in linhas]
    _leaderboard(tmp_path, linhas)

    top = top_models_by_season(tmp_path, ["r42"], k=4, slack=0.05)
    djf = top[top["season"] == "DJF"]
    nomes = set(djf["model"])

    assert len(djf) == 4
    assert len(nomes & {"Linear", "Ridge", "RidgeCV"}) == 1     # só um representante
    assert {"Cat", "Floresta", "Arvore"} <= nomes               # as vagas foram para ideias diferentes
    rep = djf[djf["model"].isin({"Linear", "Ridge", "RidgeCV"})].iloc[0]
    assert len(rep["equivalentes_omitidos"].split(", ")) == 2   # e o registro diz quem ficou de fora


def test_clearly_different_models_are_never_collapsed(tmp_path):
    """A tolerância é estreita de propósito: dois modelos com 0,01 de R² de
    diferença são escolhas diferentes, não cópias."""
    linhas = [("A", "val", 0.800, 3.0), ("B", "val", 0.790, 3.2), ("C", "val", 0.780, 3.4)]
    linhas += [(m, "test", r2, p90) for m, _, r2, p90 in linhas]
    _leaderboard(tmp_path, linhas)
    top = top_models_by_season(tmp_path, ["r42"], k=3, slack=0.05)
    assert len(top[top["season"] == "DJF"]) == 3
