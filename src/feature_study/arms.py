"""Arms (conjuntos de features) do estudo e colunas de controle negativo.

Um arm é só um conjunto de colunas. Todos treinam nas MESMAS linhas — o que
muda entre arms é apenas o que o modelo enxerga.

  core     base(40); full(40 + novas); base+f (uma por vez); full−f (uma por vez)
  groups   base+estáticas, base+dinâmicas; full−{estáticas, dinâmicas,
           topografia, fg10, cape}  — LOO individual entre features colineares
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

ARM_SETS = ("core", "groups", "controls", "losses")
# "losses" é OPT-IN: é o segundo eixo do estudo e multiplica o fan-out, então
# não entra por omissão — quem o quer pede `--arm-sets core,groups,controls,losses`.
DEFAULT_ARM_SETS = ("core", "groups", "controls")

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


def usable_new_features(columns, static_features) -> tuple[list[str], list[str]]:
    """(novas, estáticas) presentes em `columns`, sem as duplicatas conhecidas."""
    new = sorted(
        c for c in columns
        if str(c).startswith("nf_") and c not in EXCLUDED_DUPLICATES
    )
    return new, [f for f in new if f in set(static_features)]


def feature_groups(new_features, static_features) -> dict[str, list[str]]:
    """Grupos correlacionados. Só entram os não vazios."""
    new = list(new_features)
    static = [f for f in new if f in set(static_features)]
    groups = {
        "static": static,
        "dynamic": [f for f in new if f not in set(static_features)],
        "topography": [f for f in new if f in TOPOGRAPHY],
        "fg10": [f for f in new if f.startswith("nf_fg10")],
        "cape": [f for f in new if f.startswith("nf_cape")],
    }
    return {k: v for k, v in groups.items() if v}


def build_arms(
    base_features,
    new_features,
    static_features,
    arm_sets=DEFAULT_ARM_SETS,
) -> list[Arm]:
    unknown = set(arm_sets) - set(ARM_SETS)
    if unknown:
        raise ValueError(f"arm_sets desconhecidos: {sorted(unknown)} (válidos: {ARM_SETS})")
    base = tuple(base_features)
    new = tuple(f for f in new_features if f not in EXCLUDED_DUPLICATES)
    if not new:
        raise ValueError("nenhuma feature nova disponível para o estudo")
    if set(base) & set(new):
        raise ValueError(f"features novas já estão na base: {sorted(set(base) & set(new))}")
    full = base + new
    groups = feature_groups(new, static_features)
    arms: list[Arm] = []

    if "core" in arm_sets:
        arms.append(Arm("base", base, "base", "core"))
        arms.append(Arm("full", full, "full", "core"))
        for f in new:
            arms.append(Arm(f"add__{f}", base + (f,), "add", "core", f, "base", "add"))
        for f in new:
            arms.append(Arm(
                f"drop__{f}", tuple(x for x in full if x != f),
                "drop", "core", f, "full", "drop",
            ))

    if "groups" in arm_sets:
        for g in ("static", "dynamic"):
            if g in groups:
                arms.append(Arm(
                    f"add_grp__{g}", base + tuple(groups[g]),
                    "add_group", "groups", g, "base", "add",
                ))
        for g in ("static", "dynamic", "topography", "fg10", "cape"):
            if g in groups:
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
        if "static" in groups:
            perm = tuple(PERM_PREFIX + f for f in groups["static"])
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
    if {"add_grp__static", "ctrl__perm_static"} <= have:
        out.append(("real_vs_perm_static", "ctrl__perm_static", "add_grp__static"))
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
