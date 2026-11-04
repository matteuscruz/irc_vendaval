"""Empirical Quantile Mapping (EQM): devolve à previsão do modelo a distribuição observada (INMET).

É a etapa 2 do "Corrigido ML" (a etapa 1 é o modelo de ML). Documentação completa, com resultados:
`documentation/11_mapeamento_de_quantis.md`.

O QUE É
-------
O procedimento é o Empirical Quantile Mapping (EQM), também chamado de mapeamento quantil-quantil: uma técnica
consolidada de correção de viés e pós-processamento estatístico em clima, hidrologia e previsão. Cada valor
previsto p é levado ao valor observado que ocupa o MESMO quantil:

    f(p) = Q_INMET( F_modelo(p) )

com F_modelo a distribuição empírica (acumulada) das previsões e Q_INMET a função quantil empírica do INMET,
ambas na mesma amostra de calibração. É a formulação clássica: ajusta a distribuição MARGINAL do modelo à de
referência, quantil a quantil.

POR QUE AQUI
------------
O modelo minimiza o erro quadrático e, quando não sabe se haverá pico, aposta no meio. Ele ordena os dias melhor
que o ERA5, mas sai com a distribuição comprimida (cluster 3: desvio de 3,0 contra 3,7 m/s do INMET, P99 ~3 m/s
abaixo). O EQM corrige mais que a média: corrige dispersão, forma da distribuição e vieses que dependem do quantil
(cauda alta subestimada).

PROPRIEDADES (e o que NÃO esperar)
----------------------------------
- Ordem: a função é monotônica não decrescente, então a ORDEM dos valores é preservada, e com ela a correlação de
  posto (Spearman). A correlação de Pearson pode mudar um pouco, porque a transformação é não linear e pode criar
  empates (no cluster 12 ela foi de 0,60 para 0,59).
- Sem informação nova: o EQM não descobre eventos extremos. Ele reescala a distribuição para parecer com a do
  INMET. Se o modelo não pôs um dia na cauda alta, o mapa não o transforma em extremo. Ele corrige o viés
  marginal ("os valores que o modelo já considera altos estavam baixos demais"), mas não melhora necessariamente a
  discriminação, o acerto condicional ou a identificação dos dias que de fato foram extremos.
- Troca: em geral aumenta o RMSE diário (estica também os erros) e melhora percentis, POD e a forma da cauda.

ESCOLHAS DESTA IMPLEMENTAÇÃO
----------------------------
- Calibração: um mapa por trimestre (DJF, MAM, JJA, SON), prática comum porque viés e distribuição do vento mudam
  com a estação. É feito na VALIDAÇÃO do estudo (nunca no teste nem na região de aplicação). Ver `fit`.
- Quantis: 199 níveis igualmente espaçados, de 0,5 % a 99,5 %, com interpolação linear entre eles. Os 0,5 % de
  cada ponta ficam de fora porque ali há poucos pontos e o quantil empírico é instável.
- Extrapolação: acima do maior quantil calibrado (e abaixo do menor), aplica-se o MESMO deslocamento
  (Q_INMET − Q_modelo) do quantil extremo mais próximo, ou seja, uma correção constante. É uma extrapolação padrão
  em EQM, mas delicada: não estica a cauda além do que a calibração viu, então o P99,9 tende a continuar
  subestimado. A alternativa (manter a inclinação do último trecho) inventaria uma tendência sem dados.
- Truncamento final em [0, 80] m/s, como no resto do projeto.

RESSALVAS DA LITERATURA QUE VALEM AQUI
--------------------------------------
- Estacionariedade: o mapa supõe que a relação entre as distribuições do modelo e do INMET vista na calibração
  vale onde é aplicado. Ele foi ajustado no cluster 3 (Sul). Aplicado a outra região (ex.: cluster 12), estica a
  distribuição para a forma do Sul, não para a local.
- Extremos: com ~45 dias acima do P99 por trimestre na validação, a cauda extrema do mapa é pouco amostrada.
- Marginal, não condicional: corrige a distribuição como um todo, não o erro de cada dia.

Referências clássicas: Panofsky & Brier (1968), *Some Applications of Statistics to Meteorology*; Wood et al.
(2004), *Climatic Change* 62; Themeßl, Gobiet & Leuprecht (2011), *Int. J. Climatol.* 31; Gudmundsson et al.
(2012), *Hydrol. Earth Syst. Sci.* 16; Cannon, Sobie & Murdock (2015), *J. Climate* 28 (extremos e tendências);
Maraun (2016), *Curr. Clim. Change Rep.* 2 (revisão crítica).
"""
from __future__ import annotations

import numpy as np

N_QUANTIS = 199          # níveis de 0,5 % a 99,5 % (as pontas extremas, pouco amostradas, ficam de fora)


def fit(pred, obs, n_quantis: int = N_QUANTIS) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta o mapa EQM: pares (quantis da previsão, quantis do observado) nos MESMOS níveis.

    `pred` e `obs` são da mesma amostra de calibração (aqui, a validação de um trimestre). Os quantis do observado
    não precisam estar pareados dia a dia com os da previsão: o EQM só compara as duas distribuições marginais.
    Linhas com NaN/infinito em qualquer um dos dois são descartadas.
    """
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    niveis = np.linspace(0.5 / 100, 1 - 0.5 / 100, n_quantis)
    xq, yq = np.quantile(pred[ok], niveis), np.quantile(obs[ok], niveis)
    # o interp exige x crescente; previsões repetidas geram quantis iguais, então separa por um epsilon
    xq = np.maximum.accumulate(xq + np.arange(len(xq)) * 1e-9)
    return xq, yq


def apply(pred, mapa: tuple[np.ndarray, np.ndarray], clip=(0.0, 80.0)) -> np.ndarray:
    """Aplica o mapa EQM: f(p) = Q_obs(F_pred(p)), por interpolação linear entre os quantis calibrados.

    Fora da faixa calibrada, aplica o deslocamento constante do quantil extremo mais próximo (extrapolação
    constante, ver o cabeçalho do módulo). Monotônico: preserva a ordem dos valores. NaN continua NaN.
    """
    xq, yq = mapa
    p = np.asarray(pred, float)
    out = np.interp(p, xq, yq)
    baixo, alto = p < xq[0], p > xq[-1]
    out[baixo] = p[baixo] + (yq[0] - xq[0])
    out[alto] = p[alto] + (yq[-1] - xq[-1])
    out[~np.isfinite(p)] = np.nan
    return np.clip(out, *clip)
