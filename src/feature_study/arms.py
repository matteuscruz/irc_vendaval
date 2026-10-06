"""Arms (conjuntos de features) do estudo e colunas de controle negativo.

Um arm é só um conjunto de colunas. Todos treinam nas MESMAS linhas — o que
muda entre arms é apenas o que o modelo enxerga.

  anchors  base; full (base + novas). SEMPRE necessário: todo grupo e todo
           controle mede a própria contribuição CONTRA uma dessas duas.
  singles  base+f (uma por vez); full−f (uma por vez). OPT-IN — é essa varredura
           que produz ~100 arms, para responder "quanto vale UMA coluna", que
           ensembles de árvores respondem mal (capturam interação internamente)
           e que não é a pergunta de negócio. Roda depois, restrita ao grupo
           que venceu.
  core     ALIAS de ("anchors", "singles"), para comandos e `arms.json` antigos.
  groups   base+grupo; full−grupo — LOO individual entre features colineares
           tende a ≈ 0, o grupo é onde a redundância aparece
  controls base+ruído; base+estáticas permutadas entre estações. O segundo
           mantém "identifica a estação" (21 valores) e quebra a física: se a
           estática real ≈ permutada, ela só atua como ID de estação.
  losses   MESMAS features do âncora, função de PERDA diferente (ver
           `losses.py`). É o segundo EIXO do estudo: não mede o que o modelo
           enxerga, mede o que ele é instruído a minimizar.

`role` diz de que lado da comparação o arm está em relação a `reference`:
"add" (o arm é o lado esperado como melhor) ou "drop" (o lado esperado como
pior). No eixo de perda não há "mais features" — `role` é só a ORIENTAÇÃO da
comparação, e a convenção de sinal é a mesma do eixo de features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.feature_study.config import (
    EXCLUDED_DUPLICATES, MODEL_SEED, NOISE_COLUMN, PERM_PREFIX, TOPOGRAPHY,
)
from src.feature_study.losses import loss_keys

ARM_SETS = (
    "anchors", "singles", "groups", "controls", "losses",
    "hourly", "hourly_only", "hourly_raw", "era5_groups",
)
# "losses", "hourly", "hourly_only" e "hourly_raw" são OPT-IN.
#   losses       eixo de perda (multiplica o fan-out).
#   hourly       SOMA a segunda fonte (`hourly_source.py`, 18 agregados
#                `nf_h_*` do arquivo por estação) às features diárias da grade.
#   hourly_only  TROCA: só as `nf_h_*` entram no pool de NOVAS, mantendo a base
#                diária ("o horário sozinho ajuda?").
#   hourly_raw   troca a BASE: nada de diário em lugar nenhum. As 24 horas
#                cruas viram 24 colunas (`hourly_flat.py`), tanto na base
#                (`hf_`) quanto nas novas (`hfn_`). É outra pergunta e outra
#                população — por isso exclui `hourly` e `hourly_only`.
#   era5_groups  a ESTRUTURA NOVA (2026-09-30): quatro parquets por grupo do spec
#                (`groups_source.py`). A base é o grupo 1 (+ lat/lon) e os
#                grupos 2, 3 e 4 entram por cima, como grupos INTEIROS. Troca
#                toda a população, por isso exclui os três acima.
# Os quatro modos que trocam a base são mutuamente exclusivos; `build_arms` e
# `prepare._population` recusam qualquer combinação deles.
HOURLY_ARM_SETS = ("hourly", "hourly_only", "hourly_raw", "era5_groups")
# `core` não é um arm_set próprio: expande para os dois que o substituíram,
# para que comandos e `arms.json` já gravados no volume continuem válidos.
CORE_ALIAS = ("anchors", "singles")
DEFAULT_ARM_SETS = ("anchors", "groups", "controls")

# Eixo próprio (família de hipóteses própria no BH) dos arms de `selected.py`.
SELECTION_AXIS = "selection"
SELECTION_EXTRA = ("sel_vs_base",)

# Sobre quais conjuntos de features o eixo de perda roda. Os dois permitem
# checar se o efeito da perda independe do conjunto de features; reduzir para
# ("full",) corta o custo pela metade.
LOSS_ANCHORS = ("base", "full")


@dataclass(frozen=True)
class Arm:
    name: str
    features: tuple[str, ...]
    kind: str        # base | full | add | drop | add_group | drop_group | control | loss
    arm_set: str
    subject: str = ""      # feature, grupo ou perda medida ("" em base/full)
    reference: str = ""    # arm contra o qual a contribuição é medida
    role: str = ""         # "add" | "drop" | ""
    # Campos do eixo de perda. Ambos COM DEFAULT: `from_dict` tolera chave
    # ausente, então o `arms.json` já gravado no volume continua carregando.
    loss: str = ""             # chave em losses.loss_keys(); "" = perda nativa (MSE)
    axis: str = "features"     # "features" | "loss"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Arm":
        return cls(**{**d, "features": tuple(d["features"])})


def expand_arm_sets(arm_sets) -> tuple[str, ...]:
    """Resolve o alias `core`, valida os nomes e impõe as duas regras que, se
    violadas, falhariam em SILÊNCIO mais adiante:

    1. Os três modos horários são mutuamente exclusivos — juntos, o que é
       "base" e o que é "novo" fica ambíguo.
    2. `groups`, `controls` e `losses` exigem `anchors`. Sem ele, `base`/`full`
       não existem, os arms ficam órfãos com um `reference` inexistente e
       `comparisons()` simplesmente não gera a comparação: o estudo roda,
       grava tudo e produz uma tabela de efeitos vazia, sem erro nenhum.
    """
    pedidos = list(arm_sets)
    out: list[str] = []
    for s in pedidos:
        out += list(CORE_ALIAS) if s == "core" else [s]

    unknown = set(out) - set(ARM_SETS)
    if unknown:
        raise ValueError(f"arm_sets desconhecidos: {sorted(unknown)} (válidos: {ARM_SETS + ('core',)})")

    horarios = [s for s in HOURLY_ARM_SETS if s in out]
    if len(horarios) > 1:
        raise ValueError(f"modos horários são exclusivos, recebidos {horarios} — escolha um")

    dependentes = [s for s in ("groups", "controls", "losses") if s in out]
    if dependentes and "anchors" not in out:
        raise ValueError(
            f"{dependentes} medem contra `base`/`full` e exigem o arm_set 'anchors'"
        )
    # dedup preservando a ordem pedida
    return tuple(dict.fromkeys(out))


def usable_new_features(columns, static_features) -> tuple[list[str], list[str]]:
    """(novas, estáticas) presentes em `columns`, sem as duplicatas conhecidas."""
    new = sorted(
        c for c in columns
        if str(c).startswith("nf_") and c not in EXCLUDED_DUPLICATES
    )
    return new, [f for f in new if f in set(static_features)]


# ── Grupos temáticos do modo `hourly_raw` ───────────────────────────────────
# Com tudo horário, os buckets antigos (static/dynamic/topography/fg10/cape)
# deixam de fazer sentido: não há mais "diário vs horário" para separar, e a
# pergunta passa a ser física — que BLOCO METEOROLÓGICO carrega informação.
# Roteamento por variável de origem (ver `hourly_flat.base_variable_of`), não
# por prefixo de nome.
# ATENÇÃO: as chaves são os nomes das VARIÁVEIS dentro do NetCDF, que em três
# casos diferem do nome do arquivo — `gust10fg_*.nc` contém `fg10`,
# `mbld_*.nc` contém `avg_ibld` e `msshf_*.nc` contém `avg_ishf`. Mapear pelo
# nome do arquivo faz as colunas escaparem do roteamento (foi o que aconteceu
# na primeira execução real; a guarda de `_thematic_groups` pegou).
# `u10/v10/u100/v100` não aparecem como coluna: `WIND_PAIRS` os SUBSTITUI pelas
# magnitudes `ws10`/`ws100`/`shear_100_10`. Ficam listados para o caso de a
# composição mudar.
THEMATIC_NEW: dict[str, tuple[str, ...]] = {
    "vento": ("u10", "v10", "u100", "v100", "zust", "fg10", "ws10", "ws100", "shear_100_10"),
    "termo": ("d2m", "sp", "t2m"),
    "convec": ("cape", "blh", "avg_ibld", "avg_ishf"),
    "superf": ("fsr",),
}
# As 7 estáticas ficam num grupo PRÓPRIO, chamado `static` como no esquema
# diário, e não dentro de `superf`. Dois motivos: `extra_comparisons` monta
# `real_vs_perm_static` a partir de `add_grp__static`, e esse diagnóstico —
# "a estática real vale mais que uma estática de outra estação?" — é o que já
# mostrou que elas funcionam como identificador de estação. Misturar `fsr`
# nele contaminaria a comparação com uma variável que varia no tempo.
# Blocos da BASE horária. Só geram arm de REMOÇÃO (`drop_grp__base_*`): um
# `add_grp__` contra a base seria vazio, porque eles JÁ estão na base. A
# pergunta que respondem é a inversa e é legítima — "o perfil horário cru de
# vento carrega peso próprio, ou as variáveis novas já o substituem?".
# Grupos do esquema diário que geram `add_grp__` além de `drop_grp__`.
# topography/fg10/cape são subconjuntos colineares de `dynamic` e sempre
# entraram só como remoção.
ADD_GROUPS_DAILY = ("static", "dynamic", "hourly")

THEMATIC_BASE: dict[str, tuple[str, ...]] = {
    "base_vento": ("ws_h", "sin_dir_h", "cos_dir_h"),
    "base_termo": ("msl_h", "t2m_h", "rh_h", "td_dep_h"),
    "base_precip": ("tp_h", "tp_roll24h", "tp_roll48h", "tp_roll72h"),
}


def feature_groups(new_features, static_features, base_features=(), scheme: str = "daily",
                   group_of=None) -> dict[str, list[str]]:
    """Grupos correlacionados. Só entram os não vazios.

    `scheme="daily"` (padrão) mantém os buckets históricos. `hourly` é a
    segunda fonte (`hourly_source.py`, prefixo `nf_h_`) — arquivo já por
    estação, sem grade lat/lon. Fica fora de `dynamic` de propósito: são fontes
    e métodos de agregação diferentes, e misturar os dois esconderia se o ganho
    vem de UMA fonte ou da outra.

    `scheme="hourly_raw"` usa os blocos temáticos acima. Os grupos de
    `THEMATIC_BASE` saem prefixados por `base_` e `build_arms` só gera
    `drop_grp__` para eles.
    """
    new = list(new_features)
    static = [f for f in new if f in set(static_features)]
    if scheme == "hourly_raw":
        return _thematic_groups(new, static, list(base_features))
    if scheme == "era5_groups":
        return _source_groups(new, list(base_features), group_of)
    if scheme != "daily":
        raise ValueError(f"scheme de grupos desconhecido: {scheme!r} (daily | hourly_raw | era5_groups)")
    groups = {
        "static": static,
        "dynamic": [f for f in new if f not in set(static_features) and not f.startswith("nf_h_")],
        "hourly": [f for f in new if f.startswith("nf_h_")],
        "topography": [f for f in new if f in TOPOGRAPHY],
        "fg10": [f for f in new if f.startswith("nf_fg10")],
        "cape": [f for f in new if f.startswith("nf_cape")],
    }
    return {k: v for k, v in groups.items() if v}


def _source_groups(new: list[str], base: list[str], group_of) -> dict[str, list[str]]:
    """Grupos do spec (grupo 1…4), pelo arquivo de origem de cada coluna.

    Diferente do esquema temático, aqui o grupo é a unidade de pergunta — "o
    grupo 2 acrescenta?" — e vem de `groups_source`, que registra a origem de
    cada coluna. Coluna sem origem levanta: ela escaparia de todo `drop_grp__`
    e mediria zero por construção.
    """
    if not group_of:
        raise ValueError("o esquema era5_groups exige `group_of` (coluna → grupo), de groups_source")
    groups: dict[str, list[str]] = {}
    for col in (*base, *new):
        if col not in group_of:
            raise ValueError(f"coluna sem grupo de origem: {col!r}")
        groups.setdefault(group_of[col], []).append(col)
    return {g: sorted(cols) for g, cols in sorted(groups.items())}


def _thematic_groups(new: list[str], static: list[str], base: list[str]) -> dict[str, list[str]]:
    """Cada coluna achatada é roteada pela variável que a originou. Uma coluna
    que não casar com nenhum bloco levanta, em vez de sumir da análise: um
    `drop_grp__` que deixa a variável sobreviver por fora mede zero e pareceria
    um resultado, não um bug."""
    from src.feature_study.hourly_flat import base_variable_of

    por_var: dict[str, list[str]] = {}
    for f in new:
        por_var.setdefault(base_variable_of(f), []).append(f)

    groups: dict[str, list[str]] = {}
    roteadas: set[str] = set()
    for nome, variaveis in THEMATIC_NEW.items():
        cols = [c for v in variaveis for c in por_var.get(v, [])]
        if cols:
            groups[nome] = sorted(cols)
            roteadas.update(cols)
    if static:
        groups["static"] = sorted(static)
        roteadas.update(static)

    sobrando = sorted(set(new) - roteadas)
    if sobrando:
        raise ValueError(
            f"features novas sem bloco temático: {sobrando} — acrescente a variável a "
            "THEMATIC_NEW, senão ela escapa de todo drop_grp__ e o efeito medido é zero por construção"
        )

    por_var_base: dict[str, list[str]] = {}
    for f in base:
        por_var_base.setdefault(base_variable_of(f), []).append(f)
    for nome, variaveis in THEMATIC_BASE.items():
        cols = [c for v in variaveis for c in por_var_base.get(v, [])]
        if cols:
            groups[nome] = sorted(cols)
    return groups


def build_arms(
    base_features,
    new_features,
    static_features,
    arm_sets=DEFAULT_ARM_SETS,
    group_of=None,
) -> list[Arm]:
    arm_sets = expand_arm_sets(arm_sets)
    base = tuple(base_features)
    new = tuple(f for f in new_features if f not in EXCLUDED_DUPLICATES)
    if not new:
        raise ValueError("nenhuma feature nova disponível para o estudo")
    if set(base) & set(new):
        raise ValueError(f"features novas já estão na base: {sorted(set(base) & set(new))}")
    full = base + new
    raw = "hourly_raw" in arm_sets
    por_fonte = "era5_groups" in arm_sets
    groups = feature_groups(
        new, static_features, base,
        scheme="era5_groups" if por_fonte else "hourly_raw" if raw else "daily",
        group_of=group_of,
    )
    # Grupos que viram `add_grp__` (base + grupo). No esquema RAW, todo grupo
    # de NOVAS é somável; os blocos da base (`base_*`) não, porque já estão na
    # base e somá-los daria um arm idêntico ao `base`. No esquema diário a
    # lista é explícita e histórica: topography/fg10/cape sempre foram só
    # remoções (são subconjuntos colineares de `dynamic`), e somá-las agora
    # mudaria a contagem de arms de estudos já publicados.
    somaveis = (
        [g for g, cols in groups.items() if set(cols) <= set(new)] if (raw or por_fonte)
        else [g for g in ADD_GROUPS_DAILY if g in groups]
    )
    arms: list[Arm] = []

    if "anchors" in arm_sets:
        arms.append(Arm("base", base, "base", "anchors"))
        arms.append(Arm("full", full, "full", "anchors"))

    if "singles" in arm_sets:
        for f in new:
            arms.append(Arm(f"add__{f}", base + (f,), "add", "singles", f, "base", "add"))
        for f in new:
            arms.append(Arm(
                f"drop__{f}", tuple(x for x in full if x != f),
                "drop", "singles", f, "full", "drop",
            ))

    if "groups" in arm_sets:
        for g in somaveis:
            arms.append(Arm(
                f"add_grp__{g}", base + tuple(groups[g]),
                "add_group", "groups", g, "base", "add",
            ))
        for g in groups:
            gone = set(groups[g])
            arms.append(Arm(
                f"drop_grp__{g}", tuple(x for x in full if x not in gone),
                "drop_group", "groups", g, "full", "drop",
            ))

    if "controls" in arm_sets:
        arms.append(Arm(
            "ctrl__noise", base + (NOISE_COLUMN,), "control", "controls",
            "noise", "base", "add",
        ))
        # As estáticas vêm de `static_features`, não de um grupo chamado
        # "static": no esquema RAW elas moram dentro do bloco `superf`, e
        # amarrar o controle ao nome do grupo o faria sumir em silêncio —
        # justamente o controle que é o portão de validade do estudo RAW.
        estaticas = [f for f in new if f in set(static_features)]
        if estaticas:
            perm = tuple(PERM_PREFIX + f for f in estaticas)
            arms.append(Arm(
                "ctrl__perm_static", base + perm, "control", "controls",
                "perm_static", "base", "add",
            ))

    if "losses" in arm_sets:
        anchors = {"base": base, "full": full}
        arms += build_loss_arms({k: anchors[k] for k in LOSS_ANCHORS})

    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("nomes de arm duplicados")
    return arms


def build_loss_arms(anchors: dict[str, tuple[str, ...]], keys=None) -> list[Arm]:
    """Um arm por (âncora, perda), com as MESMAS features da âncora.

    O token da perda é SUFIXO do nome (`full__exp90`, nunca `exp90__full`):
    `analysis.load_residuals` busca os resíduos com o glob
    `resid__*__{arm}.parquet`, e um prefixo faria o arm `full` capturar também
    os resíduos das perdas, misturando os dois eixos em silêncio.

    Não existe arm de perda MSE: a âncora já é o braço MSE. Ela roda com 7
    modelos e o braço de perda com 3, e `compute_effects` exige os dois lados
    finitos por (tag, modelo) — a comparação se restringe sozinha aos 3 comuns.
    """
    return [
        Arm(f"{anchor}__{key}", features, "loss", "losses",
            subject=key, reference=anchor, role="add", loss=key, axis="loss")
        for anchor, features in anchors.items()
        for key in (keys if keys is not None else loss_keys())
    ]


def arms_by_name(arms) -> dict[str, Arm]:
    return {a.name: a for a in arms}


def extra_comparisons(arms) -> list[tuple[str, str, str]]:
    """Comparações entre arms que não são "arm vs. referência":
    (nome, pior, melhor). `real_vs_perm_static` mede se a estática real
    acrescenta algo além de identificar a estação."""
    have = {a.name for a in arms}
    out = []
    if {"base", "full"} <= have:
        out.append(("full_vs_base", "base", "full"))
    # o grupo de estáticas se chama `static` no esquema antigo e `grupo4` no novo
    for g in ("static", "grupo4"):
        if {f"add_grp__{g}", "ctrl__perm_static"} <= have:
            out.append(("real_vs_perm_static", "ctrl__perm_static", f"add_grp__{g}"))
            break
    # conjunto reduzido contra o grupo 1 (a base): ele supera o que já tínhamos?
    for nome in sorted(n for n in have if n.startswith("sel__") and "_sem_" not in n):
        if "base" in have:
            out.append(("sel_vs_base", "base", nome))
        break
    return out


def add_control_columns(df: pd.DataFrame, static_features, seed: int = MODEL_SEED) -> pd.DataFrame:
    """Colunas dos controles negativos, calculadas na POPULAÇÃO (antes de
    amostrar), para serem idênticas entre réplicas.

    - ruído: gaussiano padrão, sem relação com nada;
    - estáticas permutadas: cada estação recebe os valores de OUTRA estação
      (deslocamento cíclico sobre uma ordem aleatória ⇒ sem ponto fixo). A
      distribuição marginal e a estrutura conjunta das 7 se preservam; a
      ligação com a estação verdadeira, não.
    """
    out = df.sort_values(["estacao", "time"]).reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    out[NOISE_COLUMN] = rng.standard_normal(len(out))

    stations = sorted(out["estacao"].astype(str).unique())
    if static_features and len(stations) >= 2:
        order = list(rng.permutation(stations))
        donor = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        per_station = out.assign(_e=out["estacao"].astype(str)).groupby("_e")[list(static_features)].first()
        donor_of = out["estacao"].astype(str).map(donor)
        for f in static_features:
            out[PERM_PREFIX + f] = donor_of.map(per_station[f]).to_numpy()
    return out
