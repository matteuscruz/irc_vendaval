"""Arms de seleção (`src/feature_study/selected.py`)."""
from __future__ import annotations

import json

import pytest

from src.feature_study import selected as sel
from src.feature_study.analysis import comparisons
from src.feature_study.arms import SELECTION_AXIS, Arm

FULL = ["gust10fg", "gust10fg_lag1h", "w10", "w10_r75_max", "cape", "cape_r75_max", "anor_ponto",
        "anor_r75_mean", "sdor_ponto", "latitude"]
SELECAO = {"tag": "t", "variables": ["gust10fg", "cape", "anor"], "ablate": ["anor"]}


def test_columns_of_collects_every_statistic_of_a_variable_and_nothing_else():
    cols = sel.columns_of(["gust10fg", "anor"], FULL)
    assert cols == ["gust10fg", "gust10fg_lag1h", "anor_ponto", "anor_r75_mean"]


def test_an_unknown_variable_raises_instead_of_building_a_smaller_arm():
    """Um erro de digitação viraria um arm com menos colunas, e o resultado seria
    lido como 'a variável não fazia falta'."""
    with pytest.raises(ValueError, match="sem coluna"):
        sel.columns_of(["gust10fgg"], FULL)


def test_ablation_removes_only_the_ablated_variable_and_points_to_the_selected_arm():
    sel_arm, sem = sel.build_selected_arms(FULL, SELECAO)
    assert set(sel_arm.features) - set(sem.features) == {"anor_ponto", "anor_r75_mean"}
    assert sel_arm.reference == "full" and sem.reference == "sel__t"
    assert sel_arm.axis == sem.axis == SELECTION_AXIS


def test_ablating_a_variable_that_was_not_selected_raises():
    with pytest.raises(ValueError, match="ablate"):
        sel.build_selected_arms(FULL, {**SELECAO, "ablate": ["w10"]})


def _arms_json(tmp_path):
    base = [Arm("base", ("w10",), "base", "anchors"), Arm("full", tuple(FULL), "full", "anchors")]
    (tmp_path / "arms.json").write_text(json.dumps([a.to_dict() for a in base]))


def test_append_arms_is_idempotent_and_keeps_existing_arms(tmp_path):
    _arms_json(tmp_path)
    novos = sel.build_selected_arms(FULL, SELECAO)
    sel.append_arms(tmp_path, novos)
    nomes = sel.append_arms(tmp_path, novos)
    assert nomes == ["base", "full", "sel__t", "sel__t_sem_anor"]


def test_append_arms_refuses_to_overwrite_an_arm_with_different_features(tmp_path):
    """Sobrescrever invalidaria as unidades já ajustadas com o arm antigo."""
    _arms_json(tmp_path)
    sel.append_arms(tmp_path, sel.build_selected_arms(FULL, SELECAO))
    outro = sel.build_selected_arms(FULL, {**SELECAO, "variables": ["gust10fg", "anor"], "ablate": ["anor"]})
    with pytest.raises(ValueError, match="features diferentes"):
        sel.append_arms(tmp_path, outro)


def test_comparisons_cover_the_three_questions_with_the_right_direction():
    base = [Arm("base", ("w10",), "base", "anchors"), Arm("full", tuple(FULL), "full", "anchors")]
    arms = base + sel.build_selected_arms(FULL, SELECAO)
    comps = {n: (w, b) for n, w, b in comparisons(arms, {a.name for a in arms})}
    assert comps["sel__t"] == ("sel__t", "full")              # efeito > 0 = o corte custou
    assert comps["sel__t_sem_anor"] == ("sel__t_sem_anor", "sel__t")
    assert comps["sel_vs_base"] == ("base", "sel__t")         # efeito > 0 = o reduzido vence o grupo 1
