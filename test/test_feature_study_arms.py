"""Arms e controles negativos (src/feature_study/core/arms.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.core.arms import (
    DEFAULT_ARM_SETS, Arm, add_control_columns, build_arms, expand_arm_sets,
    extra_comparisons, feature_groups,
)
from src.feature_study.core.config import NOISE_COLUMN, PERM_PREFIX

BASE = ("b1", "b2", "b3")
STATIC = ("e_orog", "e_isor")
NEW = ("c_cape", "c_cin", "s_mslp", "s_w850", "e_orog", "e_isor")
GROUP_OF = {"b1": "grupo1", "b2": "grupo1", "b3": "grupo1", "c_cape": "grupo2", "c_cin": "grupo2",
            "s_mslp": "grupo3", "s_w850": "grupo3", "e_orog": "grupo4", "e_isor": "grupo4"}


def _arms(sets=DEFAULT_ARM_SETS, new=NEW, base=BASE, group_of=GROUP_OF):
    return build_arms(base, new, STATIC, sets, group_of=group_of)


def _by_name(arms):
    return {a.name: a for a in arms}


def test_default_run_has_anchors_groups_and_controls_but_not_the_single_feature_sweep():
    """A varredura individual produz centenas de arms e responde mal (árvores capturam
    interação internamente): entrar por omissão multiplicaria o custo sem ninguém pedir."""
    nomes = {a.name for a in _arms()}
    assert {"base", "full", "ctrl__noise", "ctrl__perm_static"} <= nomes
    assert not any(n.startswith(("add__", "drop__")) for n in nomes)
    assert "singles" not in DEFAULT_ARM_SETS


def test_arm_counts_for_the_real_structure():
    """base + full; somar os 3 grupos de novas; remover os 4 grupos; 2 controles."""
    arms = _arms()
    assert len(arms) == 2 + 3 + 4 + 2
    por_tipo = pd.Series([a.kind for a in arms]).value_counts().to_dict()
    assert por_tipo == {"base": 1, "full": 1, "add_group": 3, "drop_group": 4, "control": 2}


def test_group_one_is_only_ever_removed_never_added_to_its_own_base():
    """O grupo da base já está nela: somá-lo daria um arm idêntico ao `base`."""
    nomes = {a.name for a in _arms()}
    assert "drop_grp__grupo1" in nomes and "add_grp__grupo1" not in nomes


def test_role_and_reference_tell_which_side_of_the_comparison_an_arm_is():
    arms = _by_name(_arms())
    assert (arms["add_grp__grupo2"].reference, arms["add_grp__grupo2"].role) == ("base", "add")
    assert (arms["drop_grp__grupo2"].reference, arms["drop_grp__grupo2"].role) == ("full", "drop")
    assert arms["ctrl__noise"].reference == "base"


def test_drop_group_removes_exactly_the_columns_of_the_group():
    arms = _by_name(_arms())
    full, sem = set(arms["full"].features), set(arms["drop_grp__grupo2"].features)
    assert full - sem == {"c_cape", "c_cin"}


def test_every_arm_uses_valid_unique_features_and_names_are_unique():
    arms = _arms(("core", "groups", "controls"))
    assert len({a.name for a in arms}) == len(arms)
    for a in arms:
        assert len(set(a.features)) == len(a.features)
        assert a.features


def test_arm_round_trips_through_json_dict():
    for a in _arms():
        assert Arm.from_dict(a.to_dict()) == a


def test_an_arms_json_with_keys_that_no_longer_exist_still_loads():
    """`arms.json` já gravado no volume traz campos removidos (ex.: `loss`); ler o estudo
    antigo não pode quebrar por isso."""
    d = _arms()[0].to_dict() | {"loss": "", "campo_que_nao_existe_mais": 1}
    assert Arm.from_dict(d) == _arms()[0]


def test_rejects_unknown_sets_empty_new_overlap_with_base_and_missing_groups():
    with pytest.raises(ValueError, match="desconhecidos"):
        _arms(("nope",))
    with pytest.raises(ValueError, match="nenhuma feature nova"):
        _arms(new=())
    with pytest.raises(ValueError, match="já estão na base"):
        _arms(new=("b1",))
    with pytest.raises(ValueError, match="obrigatório"):
        _arms(group_of=None)


def test_a_column_without_a_source_group_raises_instead_of_escaping_every_drop():
    """Coluna sem grupo escaparia de todo `drop_grp__` e mediria zero por construção."""
    incompleto = {k: v for k, v in GROUP_OF.items() if k != "c_cape"}
    with pytest.raises(ValueError, match="sem grupo de origem"):
        feature_groups(NEW, BASE, incompleto)


def test_extra_comparisons_only_when_both_arms_exist():
    names = {c[0] for c in extra_comparisons(_arms())}
    assert names == {"full_vs_base", "real_vs_perm_static"}
    # sem `controls` não há `ctrl__perm_static`, então só sobra full_vs_base
    sem_controles = extra_comparisons(_arms(("anchors", "groups")))
    assert {c[0] for c in sem_controles} == {"full_vs_base"}


def test_perm_static_control_survives_when_statics_are_a_group_with_another_name():
    """O controle sai de `static_features`, não do nome de um grupo: amarrá-lo ao nome o
    faria sumir justamente quando é o portão de validade do estudo."""
    renomeado = GROUP_OF | {"e_orog": "grupoX", "e_isor": "grupoX"}
    assert "ctrl__perm_static" in _by_name(_arms(group_of=renomeado))


# ── Guardas de `expand_arm_sets` ────────────────────────────────────────────

def test_core_alias_still_expands_to_anchors_plus_singles():
    """Comandos e `arms.json` já gravados no volume usam `core`."""
    assert expand_arm_sets(("core",)) == ("anchors", "singles")
    assert expand_arm_sets(("core", "groups")) == ("anchors", "singles", "groups")


def test_the_old_population_name_is_accepted_and_ignored():
    """`era5_groups` virou a única população; comandos e `meta.json` antigos o citam."""
    assert expand_arm_sets(("anchors", "groups", "era5_groups")) == ("anchors", "groups", "era5_groups")
    assert {a.name for a in _arms(("anchors", "groups", "era5_groups"))} == {a.name for a in _arms(("anchors", "groups"))}


def test_group_or_control_arm_sets_without_anchors_raise_instead_of_producing_orphan_arms():
    """Sem `base`/`full`, os arms de grupo ficam com um `reference` inexistente e
    `comparisons()` apenas não gera a comparação: o estudo roda inteiro e devolve uma
    tabela de efeitos vazia, sem erro nenhum."""
    for dependente in ("groups", "controls"):
        with pytest.raises(ValueError, match="exigem o arm_set 'anchors'"):
            _arms((dependente,))


def test_the_individual_feature_sweep_builds_add_and_drop_per_new_feature():
    nomes = {a.name for a in _arms(("anchors", "singles"))}
    assert {f"add__{f}" for f in NEW} <= nomes and {f"drop__{f}" for f in NEW} <= nomes


# ── Controles negativos ─────────────────────────────────────────────────────

def _frame(n_days=40, stations=("A", "B", "C", "D")):
    days = pd.date_range("2010-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"estacao": np.repeat(stations, n_days), "time": np.tile(days, len(stations))})
    vals = {s: (10.0 * (i + 1), 0.1 * (i + 1)) for i, s in enumerate(stations)}
    df["e_orog"] = df["estacao"].map(lambda s: vals[s][0])
    df["e_isor"] = df["estacao"].map(lambda s: vals[s][1])
    return df


def test_control_columns_are_deterministic_and_row_order_independent():
    df = _frame()
    a = add_control_columns(df, ["e_orog", "e_isor"], seed=42)
    b = add_control_columns(df.sample(frac=1.0, random_state=1), ["e_orog", "e_isor"], seed=42)

    assert np.allclose(a[NOISE_COLUMN], b[NOISE_COLUMN])
    assert a[NOISE_COLUMN].std() == pytest.approx(1.0, abs=0.15)


def test_permuted_statics_come_from_another_station_and_keep_the_marginal():
    df = _frame()
    out = add_control_columns(df, ["e_orog", "e_isor"], seed=42)
    real = out.groupby("estacao")["e_orog"].first()
    perm = out.groupby("estacao")[PERM_PREFIX + "e_orog"].first()

    assert (real != perm).all()                              # nenhuma estação fica com o próprio valor
    assert sorted(real) == sorted(perm)                      # mesma marginal (só reatribuída)
    # A estrutura CONJUNTA das estáticas se preserva: o par (altitude, isor) que
    # uma estação recebe é o par de UMA estação real.
    pairs_real = set(zip(out["e_orog"], out["e_isor"]))
    pairs_perm = set(zip(out[PERM_PREFIX + "e_orog"], out[PERM_PREFIX + "e_isor"]))
    assert pairs_perm == pairs_real
