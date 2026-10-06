"""Importância por VARIÁVEL dentro dos grupos de entrega do Paulo (1 e 4).

O estudo no Modal mede contribuição com bootstrap pareado, ao nível de BLOCO
temático (vento, convecção, termodinâmica, superfície, estáticas). Este módulo
desce um nível — para a variável individual — com um instrumento diferente e
mais fraco: **importância por permutação**, medida no conjunto de teste.

A diferença entre os dois instrumentos importa para ler o resultado:

| | estudo (Modal) | este módulo |
|---|---|---|
| o que mede | ganho ao TREINAR com/sem o bloco | perda ao EMBARALHAR a variável num modelo já treinado |
| incerteza | bootstrap pareado em blocos (ano, mês) | desvio entre repetições da permutação |
| colinearidade | tratada: o modelo é reajustado sem o bloco | **não** tratada: duas variáveis redundantes dividem o crédito e ambas parecem fracas |
| custo | ~12 container-h pagas | local, minutos |

A ressalva da colinearidade é a mais séria: `ws10`, `ws100` e `fg10` medem
vento quase no mesmo lugar. Se o modelo puder substituir uma pela outra,
embaralhar uma sozinha quase não dói, e a variável parece inútil quando na
verdade é redundante. Por isso a permutação aqui é feita também por GRUPO
inteiro — a comparação entre "grupo inteiro" e "soma das variáveis" é o que
revela redundância.

As 24 colunas de uma variável são sempre permutadas JUNTAS, com a mesma
permutação de linhas: embaralhar hora a hora destruiria o perfil intradiário e
mediria outra coisa (a importância da *forma*, não da variável).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.feature_study.hourly_flat import base_variable_of

# Grupos conforme o SPEC (`features_grupos_1_e_4.md`, Tabelas 1 e 4 de
# `ERA5_INMET_Gust_BiasCorrection_ML_Technical_Spec.pdf` v1.0), que é a fonte
# autoritativa — e não o README montado a partir das mensagens do Paulo.
GRUPO_1 = ("ws10", "ws100", "shear_100_10", "fg10", "sp", "zust",
           "blh", "avg_ibld", "avg_ishf", "t2m", "d2m")
GRUPO_4 = ("fsr",)
GRUPO_4_ESTATICAS = ("nf_anor", "nf_isor", "nf_lsm", "nf_orog_height",
                     "nf_sdfor", "nf_sdor", "nf_slor")
SEM_GRUPO = ("cape",)

# Variáveis do grupo 1 que chegam ao modelo pela BASE horária por estação
# (`hf_*`), não pela grade. O spec as lista no grupo 1; o estudo as classificou
# como base só por causa da fonte do arquivo.
# Sufixo `_h`: é como `hourly_flat.base_variable_of` nomeia as colunas da base
# (`hf_ws_h12` → `ws_h`), para não colidir com a versão de grade (`hfn_t2m_h12`
# → `t2m`). Os dois nomes coexistem de propósito — são as duas fontes.
GRUPO_1_NA_BASE = ("ws_h", "t2m_h", "td_dep_h")
COORDENADAS = ("latitude", "longitude")   # o spec também as põe no grupo 1

# PARES DUPLICADOS — o mesmo campo ERA5 chega por dois caminhos: o arquivo
# horário já interpolado nas estações (`hf_`) e a grade extraída a pontos
# (`hfn_`). Medido no teste: corr(hf_ws, hfn_ws10) = 0,9927 e
# corr(hf_t2m, hfn_t2m) = 0,9980, com diferença média de 0,15 m/s e 0,19 K.
#
# Consequência para QUALQUER leitura de importância: embaralhar a versão
# `hfn_` quase não dói, porque o modelo lê a mesma informação no gêmeo da base.
# Essas variáveis aparecem artificialmente fracas, e o número delas NÃO deve
# ser comparado com o de uma variável sem gêmeo (`fg10`, `ws100`, `cape`...).
GEMEO_NA_BASE = {
    "ws10": "ws_h",      # W10 do spec, pelas duas fontes
    "t2m": "t2m_h",
    "d2m": "td_dep_h",   # td_dep = t2m − d2m ⇒ d2m é recuperável da base
}

# Features que o spec pede e que NÃO existem no estudo — registradas para que a
# ausência seja uma lacuna conhecida, não um esquecimento silencioso.
AUSENTES_DO_SPEC = {
    "grupo 1": ("hora solar média (sin/cos)", "dia do ano (sin/cos)"),
    "grupo 4": ("distância à costa",),
}

ROTULO_GRUPO = {
    **{v: "grupo 1" for v in GRUPO_1},
    **{v: "grupo 4" for v in GRUPO_4},
    **{v: "grupo 4" for v in GRUPO_4_ESTATICAS},
    **{v: "sem grupo" for v in SEM_GRUPO},
    **{v: "grupo 1 (via base)" for v in GRUPO_1_NA_BASE},
}

# Descrição física, conforme o spec. `isor` e `anor` estavam TROCADAS numa
# versão anterior deste arquivo: pelo spec, `isor` (161) é a anisotropia e
# `anor` (162) é o ângulo principal do relevo alongado — não o contrário.
DESCRICAO = {
    "ws10": "vento a 10 m — W10 (|u,v|)", "ws100": "vento a 100 m — W100",
    "shear_100_10": "cisalhamento 10↔100 m (BWD)",
    "fg10": "rajada máx. a 10 m desde o pós-proc. anterior",
    "sp": "pressão de superfície", "zust": "velocidade de fricção (u*)",
    "blh": "altura da camada limite", "avg_ibld": "dissipação média na camada limite",
    "avg_ishf": "fluxo de calor sensível médio", "t2m": "temperatura a 2 m",
    "d2m": "ponto de orvalho a 2 m", "fsr": "rugosidade aerodinâmica (z0)",
    "cape": "energia convectiva disponível",
    "ws_h": "vento a 10 m — W10, via base",
    "t2m_h": "temperatura a 2 m, via base",
    "td_dep_h": "depressão do ponto de orvalho (T2m − Td2m)",
    "nf_anor": "ângulo principal do relevo alongado",
    "nf_isor": "anisotropia do relevo subgrade",
    "nf_lsm": "máscara terra-mar (fração de terra)",
    "nf_orog_height": "altura da superfície",
    "nf_sdfor": "desvio-padrão da orografia filtrada",
    "nf_sdor": "desvio-padrão da orografia subgrade",
    "nf_slor": "declividade subgrade",
}


def blocos_por_variavel(features) -> dict[str, list[str]]:
    """{variável: colunas}. As 24 horas de uma variável formam um bloco; uma
    estática é um bloco de uma coluna só."""
    blocos: dict[str, list[str]] = {}
    for c in features:
        if c in ("latitude", "longitude"):
            continue
        blocos.setdefault(base_variable_of(c), []).append(c)
    return {k: sorted(v) for k, v in sorted(blocos.items())}


def blocos_das_novas(features) -> dict[str, list[str]]:
    """Só as variáveis NOVAS (`hfn_*` e as estáticas `nf_*`) — as da base
    (`hf_*`) não pertencem a grupo nenhum do Paulo."""
    return {k: v for k, v in blocos_por_variavel(features).items()
            if k in ROTULO_GRUPO}


def _metricas(y: np.ndarray, pred: np.ndarray, estacoes: np.ndarray) -> dict:
    cauda = y >= np.percentile(y, 90)
    por_estacao = {
        e: root_mean_squared_error(y[estacoes == e], pred[estacoes == e])
        for e in np.unique(estacoes)
    }
    return {
        "rmse": root_mean_squared_error(y, pred),
        "rmse_p90": root_mean_squared_error(y[cauda], pred[cauda]),
        "por_estacao": por_estacao,
    }


def permutacao_por_bloco(
    modelo, x: pd.DataFrame, y: np.ndarray, estacoes: np.ndarray,
    blocos: dict[str, list[str]], n_repeticoes: int = 5, seed: int = 42,
) -> pd.DataFrame:
    """Quanto o erro PIORA ao embaralhar cada bloco de colunas.

    Devolve uma linha por (bloco, repetição) com o RMSE global, o da cauda e o
    de cada estação — a mesma passagem serve para o ranking, o mapa e o recorte
    de extremos, sem repetir predições.
    """
    rng = np.random.default_rng(seed)
    base = _metricas(y, np.asarray(modelo.predict(x), float), estacoes)

    linhas = []
    for nome, colunas in blocos.items():
        for r in range(n_repeticoes):
            embaralhado = x.copy()
            # MESMA permutação de linhas para as 24 horas: embaralhar hora a
            # hora destruiria o perfil intradiário e mediria outra coisa.
            ordem = rng.permutation(len(x))
            embaralhado[colunas] = x[colunas].to_numpy()[ordem]
            m = _metricas(y, np.asarray(modelo.predict(embaralhado), float), estacoes)
            linhas.append({
                "variavel": nome, "grupo": ROTULO_GRUPO.get(nome, "base"),
                "repeticao": r, "n_colunas": len(colunas),
                "d_rmse": m["rmse"] - base["rmse"],
                "d_rmse_p90": m["rmse_p90"] - base["rmse_p90"],
                **{f"d_rmse__{e}": m["por_estacao"][e] - base["por_estacao"][e]
                   for e in base["por_estacao"]},
            })
    return pd.DataFrame(linhas)


def permutacao_por_faixa_horaria(
    modelo, x: pd.DataFrame, y: np.ndarray, blocos: dict[str, list[str]],
    variaveis, largura: int = 4, n_repeticoes: int = 3, seed: int = 42,
) -> pd.DataFrame:
    """Que FAIXA do dia de cada variável carrega o sinal.

    Só o desenho RAW permite esta pergunta: no estudo diário as 24 horas já
    tinham sido colapsadas em `mean`/`max` antes de o modelo ver qualquer
    coisa.

    Em faixas de `largura` horas, e não hora a hora, por necessidade: horas
    vizinhas da mesma variável são quase idênticas, então embaralhar UMA quase
    não dói — o modelo lê a mesma informação nas 23 restantes — e o resultado
    vira ruído. Embaralhar a faixa inteira quebra a redundância local e deixa
    a assinatura aparecer: um sinal concentrado no fim da tarde é convecção;
    espalhado pelo dia é circulação sinótica.
    """
    if 24 % largura:
        raise ValueError(f"largura {largura} não divide 24 — as faixas ficariam desiguais")
    rng = np.random.default_rng(seed)
    base = root_mean_squared_error(y, np.asarray(modelo.predict(x), float))

    linhas = []
    for nome in variaveis:
        por_hora = {}
        for coluna in blocos[nome]:
            if "_h" in coluna:
                por_hora[int(coluna.rsplit("_h", 1)[1])] = coluna
        if len(por_hora) != 24:
            continue                       # estática: não tem ciclo diário
        for inicio in range(0, 24, largura):
            colunas = [por_hora[h] for h in range(inicio, inicio + largura)]
            for r in range(n_repeticoes):
                embaralhado = x.copy()
                ordem = rng.permutation(len(x))
                embaralhado[colunas] = x[colunas].to_numpy()[ordem]
                erro = root_mean_squared_error(
                    y, np.asarray(modelo.predict(embaralhado), float))
                linhas.append({"variavel": nome, "faixa_inicio": inicio,
                               "faixa": f"{inicio:02d}-{inicio + largura - 1:02d}h",
                               "repeticao": r, "d_rmse": erro - base})
    return pd.DataFrame(linhas)
