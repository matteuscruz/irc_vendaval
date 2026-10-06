"""Arms (conjuntos de features) do estudo e colunas de controle negativo.

Um arm é só um conjunto de colunas. Todos treinam nas MESMAS linhas — o que
muda entre arms é apenas o que o modelo enxerga. A população é sempre a da
estrutura por grupo (`data/groups_source.py`): o grupo de cada coluna vem do
arquivo de origem, e o grupo 1 (mais latitude/longitude) é a base.

  anchors  base; full (base + novas). SEMPRE necessário: todo grupo e todo
           controle mede a própria contribuição CONTRA uma dessas duas.
  singles  base+f (uma por vez); full−f (uma por vez). OPT-IN — produz
           centenas de arms, para responder "quanto vale UMA coluna", que
           ensembles de árvores respondem mal (capturam interação internamente).
           Roda depois, restrita ao grupo que venceu.
  core     ALIAS de ("anchors", "singles"), para comandos e `arms.json` antigos.
  groups   base+grupo; full−grupo — LOO individual entre features colineares
           tende a ≈ 0, o grupo é onde a redundância aparece
  controls base+ruído; base+estáticas permutadas entre estações. O segundo
           mantém "identifica a estação" e quebra a física: se a estática real
           ≈ permutada, ela só atua como ID de estação.
  era5_groups  nome histórico da população por grupo. Hoje é a única; é aceito
           (e ignorado) para que comandos e `meta.json` antigos continuem válidos.

Os arms de SELEÇÃO (`sel__*`) não nascem aqui: entram depois, em `arms.json`, por
`selection/selected.py`, para não refazer o `prepare`.

`role` diz de que lado da comparação o arm está em relação a `reference`:
"add" (o arm é o lado esperado como melhor) ou "drop" (o lado esperado como pior).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from src.feature_study.core.config import MODEL_SEED, NOISE_COLUMN, PERM_PREFIX

ARM_SETS = ("anchors", "singles", "groups", "controls", "era5_groups")
# `core` não é um arm_set próprio: expande para os dois que o substituíram,
# para que comandos e `arms.json` já gravados no volume continuem válidos.
CORE_ALIAS = ("anchors", "singles")
DEFAULT_ARM_SETS = ("anchors", "groups", "controls")

# Eixo próprio (família de hipóteses própria no BH) dos arms de `selected.py`.
SELECTION_AXIS = "selection"
SELECTION_EXTRA = ("sel_vs_base",)


@dataclass(frozen=True)
class Arm:
    name: str
    features: tuple[str, ...]
    kind: str        # base | full | add | drop | add_group | drop_group | control | select
    arm_set: str
    subject: str = ""      # feature, grupo ou variável medida ("" em base/full)
    reference: str = ""    # arm contra o qual a contribuição é medida
    role: str = ""         # "add" | "drop" | ""
    axis: str = "features"     # "features" | "selection"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Arm":
        # Filtra às chaves conhecidas: `arms.json` gravado no volume antes de
        # campos serem removidos (ex.: `loss`) continua carregando.
        nomes = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in nomes}
        return cls(**{**d, "features": tuple(d["features"])})


def expand_arm_sets(arm_sets) -> tuple[str, ...]:
    """Resolve o alias `core`, valida os nomes e impõe a regra que, se violada,
    falharia em SILÊNCIO mais adiante: `groups` e `controls` exigem `anchors`.
    Sem ele, `base`/`full` não existem, os arms ficam órfãos com um `reference`
    inexistente e `comparisons()` simplesmente não gera a comparação: o estudo
    roda, grava tudo e produz uma tabela de efeitos vazia, sem erro nenhum.
    """
    out: list[str] = []
    for s in arm_sets:
        out += list(CORE_ALIAS) if s == "core" else [s]

    unknown = set(out) - set(ARM_SETS)
    if unknown:
        raise ValueError(f"arm_sets desconhecidos: {sorted(unknown)} (válidos: {ARM_SETS + ('core',)})")

    dependentes = [s for s in ("groups", "controls") if s in out]
    if dependentes and "anchors" not in out:
        raise ValueError(
            f"{dependentes} medem contra `base`/`full` e exigem o arm_set 'anchors'"
        )
    return tuple(dict.fromkeys(out))      # dedup preservando a ordem pedida


def feature_groups(new_features, base_features, group_of) -> dict[str, list[str]]:
    """Grupos do spec (grupo 1…4), pelo arquivo de origem de cada coluna.

    O grupo é a unidade de pergunta — "o grupo 2 acrescenta?" — e vem de
    `groups_source`, que registra a origem de cada coluna. Coluna sem origem
    levanta: ela escaparia de todo `drop_grp__` e mediria zero por construção.
    """
    if not group_of:
        raise ValueError("`group_of` (coluna → grupo, de groups_source) é obrigatório")
    groups: dict[str, list[str]] = {}
    for col in (*base_features, *new_features):
        if col not in group_of:
            raise ValueError(f"coluna sem grupo de origem: {col!r}")
        groups.setdefault(group_of[col], []).append(col)
    return {g: sorted(cols) for g, cols in sorted(groups.items())}


def build_arms(base_features, new_features, static_features, arm_sets=DEFAULT_ARM_SETS,
               group_of=None) -> list[Arm]:
    arm_sets = expand_arm_sets(arm_sets)
    base = tuple(base_features)
    new = tuple(new_features)
    if not new:
        raise ValueError("nenhuma feature nova disponível para o estudo")
    if set(base) & set(new):
        raise ValueError(f"features novas já estão na base: {sorted(set(base) & set(new))}")
    full = base + new
    groups = feature_groups(new, base, group_of)
    # Só os grupos de NOVAS são somáveis: o grupo da base já está nela, e somá-lo
    # daria um arm idêntico ao `base`.
    somaveis = [g for g, cols in groups.items() if set(cols) <= set(new)]
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
        # As estáticas vêm de `static_features`, não de um grupo pelo nome:
        # amarrar o controle ao nome do grupo o faria sumir em silêncio —
        # justamente o controle que é o portão de validade do estudo.
        estaticas = [f for f in new if f in set(static_features)]
        if estaticas:
            perm = tuple(PERM_PREFIX + f for f in estaticas)
            arms.append(Arm(
                "ctrl__perm_static", base + perm, "control", "controls",
                "perm_static", "base", "add",
            ))

    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("nomes de arm duplicados")
    return arms


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
