"""Congela a seleção de variáveis escolhida na VALIDAÇÃO (seção 8–9 do notebook
`grupos_1_e_4_importancia_por_variavel`) em `config/selected_features_val12.json`.

Conferências: toda variável tem coluna no `full` do estudo e os dois `arms.json`
(local e Modal) têm o MESMO `full`, senão os arms novos não parearam com os antigos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.feature_study.selected import columns_of, fingerprint  # noqa: E402

VARIAVEIS = {
    "grupo1": ["gust", "gust10fg", "hora_solar_sin", "v10", "v100", "w10", "w100"],
    "grupo2": ["cape", "mcpr"],
    "grupo3": ["grad_mslp", "grad_mslp_hpa_100km"],
    "grupo4": ["anor"],
}
FONTES = [RAIZ / "artifacts/feature_study/cluster3_groups_modal/data/arms.json",
          RAIZ / "artifacts/feature_study/cluster3_groups/data/arms.json"]


def full_de(path):
    return next(a["features"] for a in json.loads(path.read_text()) if a["name"] == "full")


full = full_de(FONTES[0])
assert full == full_de(FONTES[1]), "o `full` local e o do Modal diferem"
variaveis = [v for g in VARIAVEIS.values() for v in g]
cols = columns_of(variaveis, full)
saida = {
    "schema": 1, "tag": "val12", "variables": variaveis, "by_group": VARIAVEIS,
    "ablate": ["anor"], "n_columns": len(cols), "columns_fp": fingerprint(cols),
    "selected_on": "validação (permutação por variável, LightGBM, 4 trimestres)",
    "rule": "d_rmse médio > piso do ruído (ctrl_noise, maior entre trimestres) e positivo em >= 3 de 4 trimestres",
    "noise_floor_val": 0.00399,
}
destino = RAIZ / "config/selected_features_val12.json"
destino.write_text(json.dumps(saida, indent=1, ensure_ascii=False))
print(f"{destino.relative_to(RAIZ)}: {len(variaveis)} variáveis, {len(cols)} colunas, fp {saida['columns_fp']}")
