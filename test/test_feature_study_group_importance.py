"""Importância por variável dentro dos grupos do Paulo
(`src/feature_study/group_importance.py`).

Sintético e leve: um modelo de brinquedo com dependência conhecida, para que a
resposta certa seja sabida de antemão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_study.diagnostics.group_importance import (
    GRUPO_1, GRUPO_4, ROTULO_GRUPO, blocos_das_novas, blocos_por_variavel,
    permutacao_por_bloco, permutacao_por_faixa_horaria,
)


def hourly_flat_columns(variaveis, prefixo="hf_"):
    """Nomes das 24 colunas horárias do desenho RAW (`hf_ws_h07`, `hfn_blh_h13`)."""
    return [f"{prefixo}{v.removesuffix('_h')}_h{h:02d}" for v in variaveis for h in range(24)]


class ModeloFalso:
    """Prevê a partir de UMA coluna só. Assim sabemos exatamente qual bloco a
    permutação tem de acusar."""

    def __init__(self, coluna: str) -> None:
        self.coluna = coluna

    def predict(self, x):
        return np.asarray(x[self.coluna], dtype=float)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cols = hourly_flat_columns(("ws10", "cape"), "hfn_") + ["nf_lsm", "latitude", "longitude"]
    x = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    estacoes = np.array(["A", "B"] * (n // 2))
    return x, estacoes


# ── Agrupamento ─────────────────────────────────────────────────────────────

def test_the_24_hours_of_a_variable_form_a_single_block():
    """A unidade de interesse é a VARIÁVEL, não a coluna: tratar cada hora
    separadamente responderia "qual hora importa", que é outra pergunta (e é a
    da seção de faixas horárias)."""
    cols = hourly_flat_columns(("ws10", "cape"), "hfn_") + ["nf_lsm"]
    b = blocos_por_variavel(cols)
    assert len(b["ws10"]) == 24
    assert len(b["cape"]) == 24
    assert b["nf_lsm"] == ["nf_lsm"]


def test_coordinates_are_not_treated_as_a_variable():
    """lat/lon identificam a estação; embaralhá-las mediria "o modelo sabe onde
    está?", não a contribuição de uma variável meteorológica."""
    b = blocos_por_variavel(["hfn_cape_h00", "latitude", "longitude"])
    assert set(b) == {"cape"}


def test_base_variables_that_the_spec_puts_in_group_one_are_evaluated_too():
    """`W10`, `T2m` e a depressão do ponto de orvalho chegam ao modelo pela base
    horária por estação, mas o spec as lista no **grupo 1**. Deixá-las de fora
    da avaliação esconderia parte do grupo — e foi o que revelou que `W10` entra
    duas vezes, com o sinal partido entre as duas cópias."""
    cols = (hourly_flat_columns(("ws_h", "t2m_h", "td_dep_h", "tp_h", "msl_h"))
            + hourly_flat_columns(("cape",), "hfn_"))
    b = blocos_das_novas(cols)

    assert {"ws_h", "t2m_h", "td_dep_h", "cape"} <= set(b)
    # `tp` e `msl` são da base e não estão em nenhuma tabela do spec: ficam fora
    assert {"tp_h", "msl_h"}.isdisjoint(b)


def test_the_duplicated_pairs_are_declared_so_their_score_is_read_as_a_floor():
    """`hfn_ws10` e `hf_ws` são o MESMO campo ERA5 por dois caminhos (corr 0,993
    medida no teste). Embaralhar um deles quase não dói, porque o modelo lê o
    outro — então a importância medida é um piso, não uma medida. Declarar o par
    é o que impede a leitura errada de "esta variável é inútil"."""
    from src.feature_study.diagnostics.group_importance import GEMEO_NA_BASE

    assert GEMEO_NA_BASE["ws10"] == "ws_h"
    assert GEMEO_NA_BASE["t2m"] == "t2m_h"
    # d2m é recuperável da base porque td_dep = t2m − d2m
    assert GEMEO_NA_BASE["d2m"] == "td_dep_h"
    for grade, base in GEMEO_NA_BASE.items():
        assert ROTULO_GRUPO[grade] == "grupo 1"
        assert ROTULO_GRUPO[base].startswith("grupo 1")


def test_every_group_variable_has_a_label_and_the_groups_do_not_overlap():
    assert set(GRUPO_1).isdisjoint(GRUPO_4)
    for v in (*GRUPO_1, *GRUPO_4):
        assert ROTULO_GRUPO[v] in ("grupo 1", "grupo 4")


# ── Permutação por bloco ────────────────────────────────────────────────────

def test_permutation_blames_the_block_the_model_actually_uses():
    x, est = _frame()
    y = x["hfn_cape_h05"].to_numpy() * 3.0
    imp = permutacao_por_bloco(ModeloFalso("hfn_cape_h05"), x, y, est,
                               blocos_das_novas(x.columns), n_repeticoes=3)
    medio = imp.groupby("variavel").d_rmse.mean()

    assert medio["cape"] > 1.0                    # o bloco usado dói muito
    assert abs(medio["ws10"]) < 1e-9              # os não usados, nada
    assert abs(medio["nf_lsm"]) < 1e-9


def test_all_24_hours_get_the_same_row_permutation():
    """Embaralhar hora a hora destruiria o perfil intradiário e mediria a
    importância da FORMA da curva, não da variável. Com uma permutação única de
    linhas, o perfil de cada dia continua íntegro — só muda de dia."""
    x, est = _frame(n=120)
    y = np.zeros(len(x))
    cols = [f"hfn_ws10_h{h:02d}" for h in range(24)]

    class Espia:
        def predict(self, frame):
            Espia.visto = frame[cols].copy()
            return np.zeros(len(frame))

    permutacao_por_bloco(Espia(), x, y, est, {"ws10": cols}, n_repeticoes=1)
    original, visto = x[cols].to_numpy(), Espia.visto.to_numpy()
    # cada linha do resultado é uma linha INTEIRA do original, não uma mistura
    for linha in visto:
        assert np.isclose(original, linha).all(axis=1).any()


def test_per_station_columns_let_the_same_pass_serve_the_map():
    """O mapa espacial sai da MESMA passagem do ranking — repetir as predições
    só para agrupar por estação seria desperdício puro."""
    x, est = _frame()
    y = x["hfn_cape_h05"].to_numpy() * 3.0
    imp = permutacao_por_bloco(ModeloFalso("hfn_cape_h05"), x, y, est,
                               blocos_das_novas(x.columns), n_repeticoes=2)
    assert {"d_rmse__A", "d_rmse__B"} <= set(imp.columns)
    assert (imp[imp.variavel == "cape"].d_rmse__A > 0).all()


# ── Faixas horárias ─────────────────────────────────────────────────────────

def test_hour_bands_locate_the_hour_the_model_depends_on():
    x, est = _frame()
    y = x["hfn_cape_h17"].to_numpy() * 3.0
    faixas = permutacao_por_faixa_horaria(
        ModeloFalso("hfn_cape_h17"), x, y, blocos_das_novas(x.columns),
        ["cape"], largura=4, n_repeticoes=2)
    pico = faixas.groupby("faixa").d_rmse.mean().idxmax()
    assert pico == "16-19h"                       # a faixa que contém a hora 17


def test_a_band_width_that_does_not_divide_24_is_rejected():
    """Faixas desiguais tornariam as barras incomparáveis entre si — uma de 5h
    dói mais que uma de 4h só por ser maior."""
    x, _ = _frame(n=48)
    with pytest.raises(ValueError, match="não divide 24"):
        permutacao_por_faixa_horaria(ModeloFalso("hfn_cape_h00"), x, np.zeros(len(x)),
                                     blocos_das_novas(x.columns), ["cape"], largura=5)


def test_static_variables_are_skipped_in_the_daily_cycle_view():
    """Uma estática não tem ciclo diário; incluí-la produziria uma linha vazia
    no gráfico e sugeriria que a pergunta faz sentido para ela."""
    x, _ = _frame(n=48)
    faixas = permutacao_por_faixa_horaria(
        ModeloFalso("nf_lsm"), x, np.zeros(len(x)), blocos_das_novas(x.columns),
        ["nf_lsm", "cape"], largura=4, n_repeticoes=1)
    assert set(faixas.variavel) == {"cape"}
