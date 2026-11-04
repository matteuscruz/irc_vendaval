# Mapeamento de quantis — Empirical Quantile Mapping (o "Corrigido ML")

**Corrigido ML** é a rajada corrigida que os notebooks de inferência mostram ao lado do ERA5. Ela sai em
duas etapas:

1. **modelo de ML**: o melhor modelo de cada trimestre do estudo de features, treinado no cluster 3 com as
   colunas selecionadas, **sem** as colunas `anor_*` (49 colunas, arm `sel__val12_sem_anor`);
2. **mapeamento de quantis**: uma correção monotônica aplicada à saída do modelo, que devolve a ela a
   distribuição do INMET.

Este documento explica a segunda etapa.

## O problema que o mapeamento resolve

O modelo é treinado para minimizar o erro quadrático. Nos dias em que as features não bastam para saber se
haverá pico, a previsão que minimiza esse erro é a do meio: o modelo "aposta na média". O resultado é uma
série que acerta mais o dia a dia que o ERA5, mas com a **distribuição comprimida**.

Cluster 3, linhas do teste do estudo (9 estações):

| | INMET | ERA5 | modelo |
|---|---|---|---|
| correlação com o INMET | — | 0,71 | **0,78** |
| RMSE diário (m/s) | — | 2,66 | **2,32** |
| desvio-padrão (m/s) | 3,69 | 3,12 | 3,00 |
| P99 (m/s) | 21,4 | 19,3 | 18,6 |

O modelo ordena os dias melhor que o ERA5 (correlação maior), mas a escala sai estreita: desvio 3,0 contra
3,7 do INMET, e o P99 fica 3 m/s abaixo. Nos gráficos de quantis isso aparece como a curva abaixo da diagonal
nos valores altos, com o modelo "cortando os extremos".

Tentar consertar isso dentro do modelo, com peso extra nos dias de cauda e perda assimétrica via Optuna, não
funcionou. O peso de cauda **deslocou a distribuição inteira para cima**, com viés de +0,5 a +0,6 m/s no
vento comum, em vez de esticar só a cauda.

## O que é o mapeamento

O procedimento é o **Empirical Quantile Mapping (EQM)**, também chamado de mapeamento quantil-quantil: uma
técnica consolidada de correção de viés e pós-processamento estatístico em clima, hidrologia e previsão. Ele
ajusta a distribuição **marginal** do modelo à distribuição observada de referência, quantil a quantil.

Para cada valor previsto `p`, o mapeamento devolve o valor do INMET que está **no mesmo quantil**:

```
f(p) = Q_INMET( F_modelo(p) )
```

- `F_modelo(p)` é a fração das previsões do modelo que ficam abaixo de `p`, ou seja, o quantil de `p` na
  distribuição do modelo;
- `Q_INMET(q)` é o valor do INMET nesse mesmo quantil `q`.

Exemplo: se 99 % das previsões do modelo ficam abaixo de 18,6 m/s e 99 % dos registros do INMET ficam abaixo
de 21,4 m/s, então o mapeamento leva 18,6 → 21,4. O mesmo vale para cada quantil entre 0,5 % e 99,5 %.

### O que ele preserva e o que ele muda

- **Preserva a ordem dos dias.** A função é monotônica não decrescente: se o modelo diz que o dia A venta
  mais que o dia B, o corrigido também diz. Com isso a correlação de **posto** (Spearman) é preservada. A de
  **Pearson** pode mudar um pouco, porque a transformação é não linear e pode criar empates. No cluster 12 ela
  foi de 0,60 para 0,59.
- **Muda a escala e a forma.** O EQM corrige mais que a média: corrige dispersão, forma da distribuição e
  vieses que dependem do quantil. O desvio-padrão volta para perto de 3,7 m/s, e os percentis altos sobem.
- **Não cria informação nova sobre extremos.** O EQM reescala a distribuição prevista para parecer com a do
  INMET. Se o modelo não colocou um dia na cauda alta, o mapeamento não o transforma em evento extremo. Ele
  corrige o viés marginal ("os valores que o modelo já considera altos estavam sistematicamente baixos"), mas não
  melhora necessariamente a discriminação, o acerto condicional ou a identificação dos dias de fato extremos.

## Como ele é ajustado

Um mapa por trimestre (DJF, MAM, JJA, SON), sempre com dados que **não** são de teste. Ajustar por estação do
ano é prática comum em EQM, porque o viés e a distribuição do vento mudam ao longo do ano.

1. o modelo, já treinado, prevê a **validação** do estudo (32 blocos ano-mês dos meses de treino, nunca usados
   no ajuste do modelo);
2. calculam-se 199 quantis, de 0,5 % a 99,5 %, das previsões da validação e do INMET da validação;
3. o par (quantis do modelo → quantis do INMET) é o mapa.

Na aplicação, valores entre dois quantis são interpolados linearmente. Valores fora da faixa vista na
validação, abaixo de 0,5 % ou acima de 99,5 %, recebem o **mesmo deslocamento** do quantil extremo mais
próximo. A extrapolação é um deslocamento constante, sem inventar inclinação, porque não há dados para
estimá-la. Essa extrapolação constante é padrão em EQM, mas é reconhecida como uma escolha delicada: ela não
estica a cauda além do que a calibração viu. O resultado é truncado em [0, 80] m/s, como o resto do projeto.

Os 199 níveis deixam de fora 0,5 % de cada ponta, onde há poucos pontos e o quantil empírico é instável.

O teste do estudo e os dias e regiões novos nunca entram no ajuste: são onde o resultado é medido.

## Resultados

**Cluster 3, teste do estudo** (meses jan/abr/jul/out; padrão = modelo sem o mapa, com `anor`):

| trimestre | RMSE padrão | RMSE mapeado | P99 previsto padrão → mapeado | P99 INMET | POD P99 padrão → mapeado |
|---|---|---|---|---|---|
| DJF | 2,69 | 2,82 | 17,9 → 19,3 | 21,1 | 0,03 → 0,07 |
| MAM | 2,12 | 2,18 | 17,2 → 19,4 | 19,6 | 0,19 → 0,38 |
| JJA | 2,17 | 2,22 | 18,7 → 20,6 | 21,8 | 0,05 → 0,21 |
| SON | 2,23 | 2,30 | 19,8 → 20,6 | 22,1 | 0,11 → 0,23 |

A correlação (Pearson) mudou no máximo 0,002 nos quatro trimestres. O mapeado custa de 0,05 a 0,13 m/s de RMSE e dobra o POD no
P99, ou seja, a fração dos extremos do INMET que a previsão alcança.

**Cluster 12, dado completamente novo** (19 estações, 2000–2025; nada do cluster 12 em treino ou calibração;
modelo sem `anor`):

| | ERA5 | modelo sem o mapa | **Corrigido ML** |
|---|---|---|---|
| RMSE diário (m/s) | **2,28** | 2,42 | 2,59 |
| correlação | **0,62** | 0,60 | 0,59 |
| POD P95 | 0,26 | 0,31 | **0,34** |
| POD P99 | 0,13 | 0,13 | **0,18** |
| FAR P99 (falsos alarmes) | **0,84** | 0,94 | 0,95 |
| inclinação entre estações (média) | 0,43 | 0,51 | **0,60** |

No cluster 12, o Corrigido ML tem a distribuição mais próxima da do INMET. No gráfico de quantis, a curva
fica sobre a diagonal até ~18 m/s, enquanto o ERA5 corta os picos. Ele pega mais extremos e acompanha melhor a
diferença entre estações mais e menos ventosas. Mas erra mais o dia a dia que o ERA5, que nessa região já tem
viés quase nulo (−0,03 m/s), e emite mais falsos alarmes.

## Embasamento na literatura

O procedimento é o EQM clássico, e a literatura sustenta cada escolha feita aqui:

| escolha | situação na literatura |
|---|---|
| `f(p) = Q_obs(F_mod(p))` | formulação clássica do mapeamento quantil-quantil |
| corrigir dispersão e cauda, não só a média | é o propósito do EQM: viés dependente do quantil |
| preservar a ordem (Spearman) | consequência da função monotônica |
| um mapa por trimestre | prática comum: o viés do vento é sazonal |
| extrapolação constante acima do último quantil | padrão, mas reconhecida como delicada |

As ressalvas que a literatura faz, todas aplicáveis aqui:

- **Estacionariedade.** O mapa supõe que a relação entre as distribuições do modelo e da observação vista na
  calibração vale onde é aplicado: em outro período e, principalmente, em outra região.
- **Extrapolação.** Além do último quantil calibrado, a correção é uma suposição, não um dado.
- **Interpretação dos extremos.** O EQM corrige a distribuição marginal, não o erro dia a dia. Um P99
  corrigido mais alto não significa que os dias extremos certos foram identificados.

Referências clássicas: Panofsky & Brier (1968), *Some Applications of Statistics to Meteorology*; Wood et
al. (2004), *Climatic Change* 62; Themeßl, Gobiet & Leuprecht (2011), *Int. J. Climatol.* 31;
Gudmundsson et al. (2012), *Hydrol. Earth Syst. Sci.* 16; Cannon, Sobie & Murdock (2015), *J. Climate* 28
(extremos e tendências); Maraun (2016), *Curr. Clim. Change Rep.* 2 (revisão crítica).

## Limites

1. **Sem ganho de acerto diário.** O mapeamento troca RMSE por distribuição. É a escolha certa quando o que
   importa são percentis e extremos (P90–P99, mapas de risco). Para minimizar o erro dia a dia, o modelo sem o
   mapa é melhor.
2. **A cauda extrema depende de poucos dias.** A validação tem ~45 dias acima do P99 por trimestre. Acima
   disso o mapa só desloca, e o P99,9 continua abaixo da diagonal.
3. **O mapa carrega o clima de onde foi ajustado.** Ele foi ajustado com o INMET do cluster 3, no Sul. Aplicado
   a outra região, estica a distribuição para a forma do Sul, não para a local. Em produção, o ideal é
   um mapa por região, ajustado com estações daquela região.
4. **Não resolve extremos que o ERA5 não enxerga.** As rajadas convectivas de verão (DJF) continuam com POD
   no P99 perto de zero, com ou sem mapa.

## Onde está no código

| o quê | arquivo |
|---|---|
| ajuste (`fit`) e aplicação (`apply`) do mapa | `src/feature_study/diagnostics/quantile_map.py` |
| testes | `test/test_feature_study_quantile_map.py` |
| mapa no cluster 3, comparação e grade | `scripts/calibrar_quantis.py` → `artifacts/feature_study/inferencia/results_feature_selction_qmap.nc`, `qmap_comparacao.csv` |
| outro cluster (modelo sem `anor` + mapa) | `scripts/inferencia_cluster.py --cluster N` → `artifacts/feature_study/inferencia_c<N>/results_cluster<N>.nc` (variável `pred_mapeado_sem_anor`) |
| notebooks | `notebooks/feature_study_grupos/atual/inferencia_selecao_features.ipynb` (cluster 3), `inferencia_cluster12.ipynb` |
