"""Purga temporal nas fronteiras de mês (`prepare.purge_split_boundaries`).

As features com defasagem (`*_lag1h`, `*_max_prev3h`, `*_delta3h`, `mslp_tend_3h`)
olham para trás e atravessam a fronteira entre meses de treino e de teste.

Sintético e leve: calendário montado à mão, resposta conhecida de antemão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_study.core.prepare import purge_split_boundaries
from src.feature_study.data.purge import purge_days

MESES_TESTE = (1, 4, 7, 10)
# Meses de treino que vêm LOGO DEPOIS de um mês de teste — é neles que o
# vazamento acontece.
MESES_APOS_TESTE = (2, 5, 8, 11)


def _calendario(anos=(2020, 2021), estacoes=("A", "B"), val_blocos=((2021, 3),)) -> pd.DataFrame:
    """Todos os dias de `anos`, para cada estação, com `_split` pela mesma
    regra do `MonthBlockSplit`: teste = jan/abr/jul/out, validação = blocos
    (ano, mês) inteiros dentro dos meses de treino, resto treino."""
    dias = pd.date_range(f"{anos[0]}-01-01", f"{anos[-1]}-12-31", freq="D")
    linhas = []
    for e in estacoes:
        for d in dias:
            if d.month in MESES_TESTE:
                s = "test"
            elif (d.year, d.month) in set(val_blocos):
                s = "val"
            else:
                s = "train"
            linhas.append({"estacao": e, "time": d, "year": d.year, "month": d.month, "_split": s})
    return pd.DataFrame(linhas)


def test_training_rows_within_lookback_of_a_test_month_are_purged():
    """O vazamento real, e o menos óbvio dos dois.

    Não é 1º de janeiro (teste) puxando dezembro (treino) — isso é a realidade
    operacional, um modelo em produção tem mesmo o passado recente. É o
    INVERSO: 1º-3 de fevereiro são TREINO e o `tp_roll72h` deles alcança 30-31
    de janeiro, que é TESTE. A linha de treino passa a conter informação de
    dias de teste.
    """
    pop = _calendario()
    out, removidas = purge_split_boundaries(pop, 3)

    treino = out[out["_split"] == "train"]
    for mes in MESES_APOS_TESTE:
        nos_tres_primeiros = treino[(treino["month"] == mes) & (treino["time"].dt.day <= 3)]
        assert nos_tres_primeiros.empty, f"mês {mes} ainda tem dias que alcançam o mês de teste"
        # o dia 4 já está fora do alcance de 3 dias e tem de sobreviver
        assert not treino[(treino["month"] == mes) & (treino["time"].dt.day == 4)].empty

    assert removidas > 0
    assert len(out) + removidas == len(pop)


def test_the_purge_is_symmetric_so_test_rows_reaching_into_training_also_go():
    """A convenção do projeto é `purge: "strict"` (`lstm_sources.py`), e
    `purge_keep` exige que TODOS os dias da janela tenham o rótulo do dia-alvo.
    Manter só um lado seria uma regra diferente da que a LSTM já usa."""
    out, _ = purge_split_boundaries(_calendario(), 3)
    teste = out[out["_split"] == "test"]
    for mes in MESES_TESTE:
        assert teste[(teste["month"] == mes) & (teste["time"].dt.day <= 3)].empty
        assert not teste[(teste["month"] == mes) & (teste["time"].dt.day == 4)].empty


def test_validation_blocks_are_purged_against_their_neighbours_too():
    """A validação são blocos (ano, mês) inteiros dentro dos meses de treino:
    a fronteira val↔treino também vaza, e sem purgá-la o `EarlyStopping`
    escolheria o modelo por um sinal contaminado."""
    pop = _calendario(val_blocos=((2021, 3),))
    out, _ = purge_split_boundaries(pop, 3)
    val = out[out["_split"] == "val"]

    assert not val.empty
    assert val[val["time"].dt.day <= 3].empty          # março começa colado em fevereiro (treino)


def test_gap_of_zero_is_a_no_op():
    """Sem nenhuma feature que olhe para trás, purgar descartaria linhas sem motivo."""
    pop = _calendario()
    out, removidas = purge_split_boundaries(pop, 0)
    assert removidas == 0
    pd.testing.assert_frame_equal(out, pop)


def test_purge_days_is_derived_from_the_features_not_hardcoded():
    """Fixar o gap deixaria um vazamento silencioso no dia em que uma feature com
    defasagem maior entrasse no estudo."""
    assert purge_days(["b1", "b2", "latitude"]) == 0
    assert purge_days(["w10", "gust10fg_lag1h"]) == 1
    assert purge_days(["w10", "blh_max_prev3h"]) == 1                # 3 h alcançam o dia anterior
    assert purge_days(["w10", "x_delta48h"]) == 2                    # 48 h pedem 2 dias


def test_all_rows_share_one_gap_so_every_arm_keeps_the_same_row_ids():
    """`compute_effects` é pareado por `row_id`: os dois lados de cada
    comparação têm de ser exatamente as mesmas linhas. Um gap por arm — cada um
    com seu conjunto de features — quebraria o pareamento sem erro visível, e o
    bootstrap devolveria IC estreitos demais.
    """
    pop = _calendario()
    out, _ = purge_split_boundaries(pop, 3)

    # a purga é função do calendário, nunca da estação: todas veem o mesmo corte
    por_estacao = out.groupby("estacao")["time"].apply(lambda s: set(s))
    assert por_estacao.iloc[0] == por_estacao.iloc[1]
    assert np.array_equal(
        np.sort(out["time"].unique()),
        np.sort(pop[pop["time"].isin(out["time"])]["time"].unique()),
    )


def test_a_day_missing_from_the_frame_still_counts_as_a_boundary():
    """Uma falha do INMET tira o dia do frame, mas o ERA5 continua existindo e
    alimentando a janela de 72h. Por isso o rótulo é recalculado pela regra do
    calendário, e não lido linha a linha do frame — senão a purga deixaria
    passar exatamente as fronteiras com dado faltando.
    """
    pop = _calendario()
    sem_31_jan = pop[~((pop["month"] == 1) & (pop["time"].dt.day == 31))]
    out, _ = purge_split_boundaries(sem_31_jan, 3)

    treino = out[out["_split"] == "train"]
    assert treino[(treino["month"] == 2) & (treino["time"].dt.day <= 3)].empty
