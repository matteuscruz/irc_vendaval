"""Arms de SELEÇÃO: o conjunto reduzido de variáveis, testado dentro do estudo.

Os arms do estudo (`base`, `full`, grupos, controles) já estão no volume do Modal.
Estes são acrescentados DEPOIS, sem refazer o `prepare`: o `prepare` regeneraria
dados e `row_id`, e o pareamento com os resultados existentes é justamente o que
se quer preservar. Por isso `append_arms` só ESCREVE arms novos em `arms.json`.

A seleção vem de um JSON congelado (`config/selected_features_*.json`), escolhido
na VALIDAÇÃO — a regra e o piso de ruído ficam registrados nele.

Três perguntas, uma por comparação (eixo `selection`, família própria no
Benjamini–Hochberg para não mexer nos q-valores do eixo de features):
  sel__<tag>            vs full      o que se perde ao cortar? (efeito = full melhor)
  sel__<tag>_sem_<v>    vs sel       o que vale a variável ablacionada? (ex.: anor)
  sel_vs_base           vs base      o conjunto reduzido supera o grupo 1?
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.feature_study.arms import SELECTION_AXIS, Arm

# Mesma redução de coluna → variável usada na análise por variável do notebook.
_SUFIXOS = re.compile(
    r"(_ponto|_r75_\w+|_r250_\w+|_c75_250|_l75_250|_lag1h|_max_prev3h|_std_prev3h|_delta3h)$"
)


def variable_of(coluna: str) -> str:
    return _SUFIXOS.sub("", coluna)


def columns_of(variaveis, features) -> list[str]:
    """Colunas de `features` cuja variável está em `variaveis`; levanta se alguma
    variável pedida não existir (um erro de digitação não pode virar arm menor)."""
    variaveis = set(variaveis)
    achadas = {variable_of(c) for c in features} & variaveis
    faltam = variaveis - achadas
    if faltam:
        raise ValueError(f"variáveis sem coluna no conjunto full: {sorted(faltam)}")
    return [c for c in features if variable_of(c) in variaveis]


def fingerprint(colunas) -> str:
    return hashlib.md5("\n".join(sorted(colunas)).encode()).hexdigest()[:12]


def load_selection(path) -> dict:
    sel = json.loads(Path(path).read_text())
    for chave in ("tag", "variables", "ablate"):
        if chave not in sel:
            raise ValueError(f"seleção sem a chave {chave!r}")
    return sel


def build_selected_arms(full_features, selection: dict) -> list[Arm]:
    """`sel__<tag>` e uma ablação por variável de `ablate`. As colunas saem de
    `full_features`, nunca de uma lista solta: garante que existem no estudo."""
    tag = selection["tag"]
    cols = columns_of(selection["variables"], full_features)
    nome = f"sel__{tag}"
    arms = [Arm(nome, tuple(cols), "select", "selected", subject=tag, reference="full",
                role="drop", axis=SELECTION_AXIS)]
    for v in selection["ablate"]:
        if v not in selection["variables"]:
            raise ValueError(f"ablate {v!r} não está entre as variáveis selecionadas")
        sem = tuple(c for c in cols if variable_of(c) != v)
        arms.append(Arm(f"{nome}_sem_{v}", sem, "select", "selected", subject=v,
                        reference=nome, role="drop", axis=SELECTION_AXIS))
    return arms


def append_arms(data_dir, novos: list[Arm]) -> list[str]:
    """Acrescenta a `arms.json`. Idempotente: arm igual é ignorado; mesmo nome com
    features diferentes levanta (sobrescrever invalidaria unidades já ajustadas)."""
    caminho = Path(data_dir) / "arms.json"
    atuais = json.loads(caminho.read_text())
    por_nome = {a["name"]: a for a in atuais}
    for a in novos:
        if a.name in por_nome:
            if tuple(por_nome[a.name]["features"]) != a.features:
                raise ValueError(f"arm {a.name!r} já existe com features diferentes")
            continue
        atuais.append(a.to_dict())
    caminho.write_text(json.dumps(atuais, indent=1))
    return [a["name"] for a in atuais]
