"""Arms e controles negativos (src/feature_study/arms.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.arms import (
    ARM_SETS, DEFAULT_ARM_SETS, LOSS_ANCHORS, Arm, add_control_columns, build_arms,
    build_loss_arms, extra_comparisons, feature_groups, usable_new_features,
)
from src.feature_study.config import NOISE_COLUMN, PERM_PREFIX
from src.feature_study.losses import loss_keys

BASE = ("b1", "b2", "b3")
STATIC = ("nf_orog_height", "nf_isor")
NEW = ("nf_cape_max", "nf_cape_mean", "nf_fg10_max", "nf_isor", "nf_orog_height", "nf_ws10_std")


def _by_name(arms):
    return {a.name: a for a in arms}


def test_core_has_base_full_and_one_add_and_drop_per_feature():
    arms = _by_name(build_arms(BASE, NEW, STATIC, arm_sets=("core",)))

    assert len(arms) == 2 + 2 * len(NEW)
    assert arms["base"].features == BASE
    assert arms["full"].features == BASE + NEW
    for f in NEW:
        assert arms[f"add__{f}"].features == BASE + (f,)
        assert set(arms[f"drop__{f}"].features) == set(BASE + NEW) - {f}


def test_duplicates_of_original_features_are_excluded():
    new, _ = usable_new_features(
        ["b1", "nf_ws10_max", "nf_ws10_mean", "nf_ws10_std", "nf_isor"], ["nf_isor"],
    )
    assert new == ["nf_isor", "nf_ws10_std"]

    arms = build_arms(BASE, ("nf_ws10_max", "nf_ws10_mean", "nf_cape_max"), (), ("core",))
    assert not any("ws10_max" in a.name or "ws10_mean" in a.name for a in arms)


def test_role_and_reference_tell_which_side_of_the_comparison_an_arm_is():
    arms = _by_name(build_arms(BASE, NEW, STATIC, ("core", "groups", "controls")))
    assert (arms["add__nf_isor"].reference, arms["add__nf_isor"].role) == ("base", "add")
    assert (arms["drop__nf_isor"].reference, arms["drop__nf_isor"].role) == ("full", "drop")
    assert (arms["ctrl__noise"].reference, arms["ctrl__noise"].role) == ("base", "add")
    assert arms["base"].reference == "" and arms["full"].reference == ""


def test_group_arms_only_exist_for_nonempty_groups():
    groups = feature_groups(NEW, STATIC)
    assert set(groups) == {"static", "dynamic", "topography", "fg10", "cape"}
    assert set(groups["topography"]) == {"nf_orog_height", "nf_isor"}

    sem_fg10 = tuple(f for f in NEW if "fg10" not in f)
    names = {a.name for a in build_arms(BASE, sem_fg10, STATIC, ("anchors", "groups"))}
    assert "drop_grp__fg10" not in names and "drop_grp__cape" in names

    sem_estaticas = build_arms(BASE, ("nf_cape_max", "nf_fg10_max"), (), ("anchors", "groups", "controls"))
    assert "ctrl__perm_static" not in {a.name for a in sem_estaticas}


def test_arm_counts_for_the_real_configuration():
    """Configuração histórica do estudo diário: 7 estáticas + 6 dinâmicas.

    Cada arm_set é contado SOMADO a `anchors`, porque `groups`, `controls` e
    `losses` medem contra `base`/`full` e não existem sem eles — contá-los
    isolados mediria uma configuração que nunca roda.
    """
    static = ("nf_orog_height", "nf_lsm", "nf_sdor", "nf_isor", "nf_anor", "nf_slor", "nf_sdfor")
    dynamic = ("nf_cape_max", "nf_cape_mean", "nf_fg10_max", "nf_fg10_mean", "nf_fsr_mean", "nf_ws10_std")
    b = tuple(f"b{i}" for i in range(40))

    def n(*sets):
        return len(build_arms(b, static + dynamic, static, ("anchors", *sets))) - 2

    assert n() == 0                                   # anchors sozinho = base + full
    assert n("singles") == 26                         # 13 add + 13 drop
    assert n("groups") == 7
    assert n("controls") == 2
    assert n("losses") == len(LOSS_ANCHORS) * len(loss_keys())
    assert n("hourly") == n("hourly_only") == 0
    # `hourly_raw` não entra aqui: ele TROCA o esquema de grupos e exige
    # colunas `hf_`/`hfn_`; com nomes diários ele levanta, de propósito.

    # o default deixou de arrastar a varredura individual: 2 + 7 + 2
    assert len(build_arms(b, static + dynamic, static)) == 11
    # o alias `core` reconstrói a configuração antiga, de 37 arms
    assert len(build_arms(b, static + dynamic, static, ("core", "groups", "controls"))) == 37


def test_the_loss_axis_is_opt_in_and_not_part_of_the_default_run():
    """O eixo de perda multiplica o fan-out no Modal; entrar por omissão
    mudaria silenciosamente o custo e o escopo de toda execução existente."""
    assert "losses" not in DEFAULT_ARM_SETS
    assert set(DEFAULT_ARM_SETS) < set(ARM_SETS)

    padrao = {a.axis for a in build_arms(BASE, NEW, STATIC)}
    assert padrao == {"features"}


def test_every_arm_uses_valid_unique_features_and_names_are_unique():
    arms = build_arms(BASE, NEW, STATIC)
    universe = set(BASE) | set(NEW) | {NOISE_COLUMN} | {PERM_PREFIX + f for f in STATIC}
    assert len({a.name for a in arms}) == len(arms)
    for a in arms:
        assert len(set(a.features)) == len(a.features)
        assert set(a.features) <= universe


def test_arm_round_trips_through_json_dict():
    a = build_arms(BASE, NEW, STATIC)[3]
    assert Arm.from_dict(a.to_dict()) == a


def test_rejects_unknown_sets_empty_new_and_overlap_with_base():
    with pytest.raises(ValueError, match="desconhecidos"):
        build_arms(BASE, NEW, STATIC, ("nope",))
    with pytest.raises(ValueError, match="nenhuma feature nova"):
        build_arms(BASE, (), STATIC)
    with pytest.raises(ValueError, match="já estão na base"):
        build_arms(BASE, ("b1",), ())


def test_extra_comparisons_only_when_both_arms_exist():
    full = build_arms(BASE, NEW, STATIC)
    names = {c[0] for c in extra_comparisons(full)}
    assert names == {"full_vs_base", "real_vs_perm_static"}
    # sem `controls` não há `ctrl__perm_static`, então só sobra full_vs_base
    sem_controles = extra_comparisons(build_arms(BASE, NEW, STATIC, ("anchors", "groups")))
    assert {c[0] for c in sem_controles} == {"full_vs_base"}


# ── Controles negativos ─────────────────────────────────────────────────────

def _frame(n_days=40, stations=("A", "B", "C", "D")):
    days = pd.date_range("2010-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"estacao": np.repeat(stations, n_days), "time": np.tile(days, len(stations))})
    vals = {s: (10.0 * (i + 1), 0.1 * (i + 1)) for i, s in enumerate(stations)}
    df["nf_orog_height"] = df["estacao"].map(lambda s: vals[s][0])
    df["nf_isor"] = df["estacao"].map(lambda s: vals[s][1])
    return df


def test_control_columns_are_deterministic_and_row_order_independent():
    df = _frame()
    a = add_control_columns(df, ["nf_orog_height", "nf_isor"], seed=42)
    b = add_control_columns(df.sample(frac=1.0, random_state=1), ["nf_orog_height", "nf_isor"], seed=42)

    assert np.allclose(a[NOISE_COLUMN], b[NOISE_COLUMN])
    assert a[NOISE_COLUMN].std() == pytest.approx(1.0, abs=0.15)


def test_permuted_statics_come_from_another_station_and_keep_the_marginal():
    df = _frame()
    out = add_control_columns(df, ["nf_orog_height", "nf_isor"], seed=42)
    real = out.groupby("estacao")["nf_orog_height"].first()
    perm = out.groupby("estacao")[PERM_PREFIX + "nf_orog_height"].first()

    assert (real != perm).all()                              # nenhuma estação fica com o próprio valor
    assert sorted(real) == sorted(perm)                      # mesma marginal (só reatribuída)
    # A estrutura CONJUNTA das estáticas se preserva: o par (altitude, isor) que
    # uma estação recebe é o par de UMA estação real.
    pairs_real = set(zip(out["nf_orog_height"], out["nf_isor"]))
    pairs_perm = set(zip(out[PERM_PREFIX + "nf_orog_height"], out[PERM_PREFIX + "nf_isor"]))
    assert pairs_perm == pairs_real


# ── Eixo de perda ───────────────────────────────────────────────────────────

def _loss_arms():
    return _by_name(build_arms(BASE, NEW, STATIC, ("core", "losses")))


def test_the_loss_token_is_a_suffix_so_the_resid_glob_does_not_capture_the_anchor():
    """`analysis.load_residuals` busca os resíduos com `resid__*__{arm}.parquet`.
    Com o token de perda como PREFIXO, o arm `base` capturaria também os
    resíduos das perdas e misturaria os dois eixos sem erro nenhum."""
    from fnmatch import fnmatch

    padrao = "resid__*__base.parquet"
    assert fnmatch("resid__DJF__base.parquet", padrao)
    assert fnmatch("resid__DJF__exp90__base.parquet", padrao)      # prefixo colide
    assert not fnmatch("resid__DJF__base__exp90.parquet", padrao)  # sufixo, não

    for a in _loss_arms().values():
        if a.axis == "loss":
            assert a.name == f"{a.reference}__{a.loss}"


def test_loss_arms_reuse_the_anchor_features_untouched():
    """O eixo de perda tem de variar SÓ a perda: uma feature a mais ou a menos
    confundiria os dois eixos num efeito só."""
    arms = _loss_arms()
    for a in arms.values():
        if a.axis == "loss":
            assert a.features == arms[a.reference].features


def test_loss_arms_orient_the_comparison_from_the_mse_anchor():
    """`role="add"` aqui não quer dizer "tem mais features" — é a ORIENTAÇÃO:
    pior = a âncora (MSE), melhor = o braço de perda. É o que faz
    `analysis.comparisons`, que é agnóstica de eixo, parear os dois lados na
    direção certa sem nenhuma mudança."""
    arms = _loss_arms()
    for key in loss_keys():
        a = arms[f"full__{key}"]
        assert (a.reference, a.role, a.axis, a.kind) == ("full", "add", "loss", "loss")
        assert a.subject == key


def test_loss_arms_do_not_enter_the_feature_rankings():
    """`ranking_features` itera `kind == "add"` e `ranking_groups` filtra por
    prefixo de nome. Reaproveitar `kind="add"` aqui faria os braços de perda
    entrarem no ranking de features como se fossem features."""
    arms = _loss_arms().values()
    de_perda = [a for a in arms if a.axis == "loss"]

    assert de_perda
    assert all(a.kind == "loss" for a in de_perda)
    assert all(not a.name.startswith(("add__", "drop__", "add_grp__", "drop_grp__", "ctrl__"))
               for a in de_perda)


def test_an_arms_json_written_before_the_loss_axis_still_loads():
    """O `arms.json` já gravado no volume do Modal não tem `loss` nem `axis`.
    `from_dict` tolera chave AUSENTE (default) mas levanta em chave EXTRA — por
    isso os campos novos precisam de default e o código sobe antes do
    `prepare`."""
    antigo = Arm("base", ("b1",), "base", "core").to_dict()
    del antigo["loss"], antigo["axis"]

    recuperado = Arm.from_dict(antigo)

    assert recuperado.loss == "" and recuperado.axis == "features"


def test_build_loss_arms_accepts_an_explicit_subset_of_losses():
    arms = build_loss_arms({"full": BASE}, keys=("exp90",))

    assert [a.name for a in arms] == ["full__exp90"]
    assert len(LOSS_ANCHORS) == 2


# ── Grupo "hourly" (segunda fonte, arquivo por estação) ─────────────────────

def test_hourly_prefixed_features_form_their_own_group_not_dynamic():
    """`nf_h_*` vem de uma fonte e um método de agregação diferentes da grade
    lat/lon (arquivo já por estação) — misturar com `dynamic` esconderia se o
    ganho vem de uma fonte ou da outra."""
    new = ("nf_cape_max", "nf_h_ws_mean", "nf_h_ws_jump")
    groups = feature_groups(new, ())

    assert set(groups["hourly"]) == {"nf_h_ws_mean", "nf_h_ws_jump"}
    assert groups["dynamic"] == ["nf_cape_max"]


def test_hourly_group_gets_add_and_drop_arms_like_static_and_dynamic():
    new = ("nf_cape_max", "nf_h_ws_mean", "nf_h_ws_jump")
    arms = _by_name(build_arms(BASE, new, (), ("core", "groups")))

    assert "add_grp__hourly" in arms
    assert set(arms["add_grp__hourly"].features) == set(BASE) | {"nf_h_ws_mean", "nf_h_ws_jump"}
    assert "drop_grp__hourly" in arms
    full = set(BASE) | set(new)
    assert set(arms["drop_grp__hourly"].features) == full - {"nf_h_ws_mean", "nf_h_ws_jump"}


def test_arms_without_any_hourly_feature_have_no_hourly_arms():
    arms = build_arms(BASE, NEW, STATIC, ("core", "groups"))
    assert not any(a.name.endswith("hourly") for a in arms)


# ── Modo `hourly_raw`: base horária crua + blocos temáticos ─────────────────

# Composição REAL, confirmada na primeira execução no Modal (312 colunas = 13×24).
# Duas surpresas que o teste agora espelha:
#  - o nome da VARIÁVEL difere do nome do arquivo em três casos: `gust10fg_*.nc`
#    contém `fg10`, `mbld_*.nc` contém `avg_ibld`, `msshf_*.nc` contém `avg_ishf`;
#  - `u10/v10/u100/v100` NÃO viram coluna: `WIND_PAIRS` os substitui por
#    `ws10`/`ws100`/`shear_100_10`.
NEW_VARS_RAW = ("blh", "cape", "d2m", "fsr", "fg10", "avg_ibld", "avg_ishf",
                "sp", "t2m", "zust", "ws10", "ws100", "shear_100_10")


def _raw_features():
    from src.feature_study.hourly_flat import BASE_HOURLY_VARS, hourly_flat_columns
    static = ["nf_anor", "nf_isor", "nf_lsm", "nf_orog_height", "nf_sdfor", "nf_sdor", "nf_slor"]
    base = hourly_flat_columns(BASE_HOURLY_VARS) + ["latitude", "longitude"]
    new = hourly_flat_columns(NEW_VARS_RAW, "hfn_") + static
    return base, new, static


def test_hourly_raw_uses_flat_columns_as_base_and_keeps_the_new_variables_separate():
    """A pergunta do estudo RAW é outra: nada de feature diária em lugar
    nenhum. A base passa a ser o perfil horário cru (`hf_`) e as variáveis
    novas entram igualmente horárias (`hfn_`), nunca agregadas."""
    base, new, static = _raw_features()
    arms = _by_name(build_arms(base, new, static, ("anchors", "groups", "controls", "hourly_raw")))

    assert len(arms["base"].features) == 266           # 11 vars x 24 h + lat/lon
    assert len(arms["full"].features) == 585           # + 13 vars x 24 h + 7 estáticas
    assert all(f.startswith("hf_") or f in ("latitude", "longitude") for f in arms["base"].features)
    assert not any(f.startswith("nf_h_") for f in arms["full"].features)


def test_thematic_groups_partition_the_features_without_overlap():
    """Um bloco que aparece em dois grupos torna `drop_grp__` incoerente: a
    variável sobrevive pelo outro grupo, o efeito medido é zero por construção
    e o resultado parece uma conclusão em vez de um bug."""
    base, new, static = _raw_features()
    groups = feature_groups(new, static, base, scheme="hourly_raw")

    todas = [c for cols in groups.values() for c in cols]
    assert len(todas) == len(set(todas))                       # sem sobreposição
    assert set(todas) == set(new) | (set(base) - {"latitude", "longitude"})  # sem sobras


def test_a_new_variable_without_a_thematic_block_raises_instead_of_escaping_every_group():
    """Falha alto em vez de sumir: uma variável fora de todo grupo nunca é
    removida por nenhum `drop_grp__`, e o estudo mediria zero para ela sem que
    nada indicasse o motivo."""
    from src.feature_study.hourly_flat import hourly_flat_columns
    base, _, static = _raw_features()
    new = hourly_flat_columns(("blh", "variavel_nova_sem_bloco"), "hfn_") + static
    with pytest.raises(ValueError, match="sem bloco temático"):
        feature_groups(new, static, base, scheme="hourly_raw")


def test_base_blocks_only_produce_removal_arms_never_additions():
    """Somar um bloco da base à própria base daria um arm idêntico ao `base`.
    A pergunta legítima é a inversa — o perfil horário cru carrega peso próprio
    ou as variáveis novas já o substituem? — e ela é um `drop_grp__`."""
    base, new, static = _raw_features()
    nomes = {a.name for a in build_arms(base, new, static, ("anchors", "groups", "hourly_raw"))}

    assert "drop_grp__base_vento" in nomes
    assert "add_grp__base_vento" not in nomes
    assert {"add_grp__vento", "drop_grp__vento"} <= nomes     # grupo de NOVAS: os dois lados


def test_perm_static_control_survives_when_statics_live_inside_a_thematic_block():
    """O controle é o portão de validade do estudo RAW (679 colunas contra
    ~5.900 linhas). Amarrá-lo ao nome de um grupo o faria sumir em silêncio
    justamente no desenho em que ele mais importa."""
    base, new, static = _raw_features()
    arms = _by_name(build_arms(base, new, static, ("anchors", "groups", "controls", "hourly_raw")))

    assert "ctrl__perm_static" in arms
    assert "ctrl__noise" in arms
    assert {c[0] for c in extra_comparisons(list(arms.values()))} == {"full_vs_base", "real_vs_perm_static"}


# ── Guardas de `expand_arm_sets` ────────────────────────────────────────────

def test_core_alias_still_expands_to_anchors_plus_singles():
    """Comandos e `arms.json` já gravados no volume usam `core`; quebrá-los
    obrigaria a refazer estudos que já estão prontos."""
    from src.feature_study.arms import expand_arm_sets
    assert expand_arm_sets(("core",)) == ("anchors", "singles")
    assert expand_arm_sets(("core", "groups")) == ("anchors", "singles", "groups")


def test_group_or_control_arm_sets_without_anchors_raise_instead_of_producing_orphan_arms():
    """Sem `base`/`full`, os arms de grupo ficam com um `reference` inexistente
    e `comparisons()` apenas não gera a comparação: o estudo roda inteiro,
    grava tudo e devolve uma tabela de efeitos vazia, sem erro nenhum."""
    for dependente in ("groups", "controls", "losses"):
        with pytest.raises(ValueError, match="exigem o arm_set 'anchors'"):
            build_arms(BASE, NEW, STATIC, (dependente,))


def test_the_three_hourly_modes_are_mutually_exclusive():
    """Cada um responde uma pergunta diferente sobre o que é "base" e o que é
    "novo"; combinados, essa fronteira fica ambígua."""
    for a, b in (("hourly", "hourly_only"), ("hourly", "hourly_raw"), ("hourly_only", "hourly_raw")):
        with pytest.raises(ValueError, match="modos horários são exclusivos"):
            build_arms(BASE, NEW, STATIC, ("anchors", a, b))


def test_the_individual_feature_sweep_is_opt_in_and_not_part_of_the_default_run():
    """É a varredura que produz ~100 arms. Entrar por omissão multiplicaria o
    custo de qualquer execução por ~8 sem que ninguém pedisse."""
    assert "singles" not in DEFAULT_ARM_SETS
    assert "anchors" in DEFAULT_ARM_SETS
    nomes = {a.name for a in build_arms(BASE, NEW, STATIC)}
    assert not any(n.startswith(("add__", "drop__")) for n in nomes)
    assert {"base", "full"} <= nomes
