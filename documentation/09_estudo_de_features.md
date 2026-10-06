# Estudo de informação de features novas (`src/feature_study/`)

Pipeline **isolada** — não altera nenhuma pipeline de treino — que responde: *quais
das features `nf_*` acrescentam informação a um modelo?* Compara modelos treinados
COM e SEM cada feature. **Somente cluster 3** e **seeds fixas** (não são flags:
`src/feature_study/config.py`). Toda execução real é no Modal.

## Resumo (leia primeiro)

- **Vigente:** modo `era5_groups` (4 parquets por grupo; uma linha por estação-dia na hora do
  pico; referência = rajada ERA5 `era5_gust_max`). Execução de referência no Modal em
  `cluster3_groups_modal` (5 seeds, 10 modelos, 97 blocos).
- **Resultado dos grupos:** só o **grupo 1** acrescenta informação (remover custa +0,216 m/s).
  Somar o 2 ou o 3 não traz nada; o grupo 4 equivale ao controle de estáticas permutadas
  (identifica a estação, não é relevo).
- **Variáveis:** 12 selecionadas na validação (58 colunas, `config/selected_features_val12.json`).
  O arm `sel__val12`, treinado no Modal, custa 0,002 m/s contra o completo e não supera o
  grupo 1 sozinho de forma relevante: a vantagem é parcimônia (ver "Seleção de variáveis").
- **Partição:** blocos de mês (teste = jan/abr/jul/out de todos os anos; validação = blocos ano-mês
  nos meses de treino; purga de 1 dia), ver "Partição treino / validação / teste".
- **Onde ver:** `notebooks/README.md` (índice) e `notebooks/feature_study_grupos/atual/`.
- **Histórico:** os desenhos anteriores (diário, diário+horário, RAW horário, eixo de perda e
  amostragem) foram removidos do código; ficam no histórico do git. Os resultados que geraram
  continuam nos volumes e nos notebooks de `notebooks/feature_study_grupos/anteriores/`.

## Por que existe

A comparação "base vs. base + 7 estáticas" trata as estáticas como um bloco, foi
treinada uma vez e tinha IC de bootstrap de *linhas* (ignora autocorrelação). Ela
não separa "o modelo usa a feature" (SHAP) de "a feature acrescenta informação".

## Layout do código (`src/feature_study/`)

| pasta | conteúdo |
|---|---|
| `core/` | o estudo: `config`, `arms`, `prepare`, `worker`, `analysis`, `artifacts` (joblib do campeão) |
| `data/` | entrada: `groups_source` (4 parquets), `base_only` (BASE antiga na hora do pico), `purge` (gap temporal) |
| `selection/` | `selected`: seleção de variáveis e arms `sel__*` |
| `diagnostics/` | análises sobre resultados já gravados: `robustness`, `shap_groups`, `group_importance` |

Entradas de linha de comando: `src/modal/feature_study.py` (Modal), `scripts/run_feature_study_local.py`
(local) e `scripts/run_feature_study.py` (depuração). Análises: `scripts/analise_robustez.py`.

## Estágios

| Estágio | Onde | O que faz |
|---|---|---|
| `prepare` | 1 container (32 GB) | lê os 4 parquets, aplica completude da **união** de features e a **purga temporal**, grava treino/teste em parquet |
| `fit` | fan-out por (seed × trimestre × lote de 5 arms) | LazyPredict (ou os top-5) por trimestre, no próprio container |
| `screen` | 1 container | congela os **5 melhores modelos por trimestre** a partir do arm `base` → `summary/top_models.json` |
| `add-arms` | 1 container | acrescenta arms (`sel__*`) a `arms.json` sem refazer o `prepare` |
| `aggregate` | 1 container | efeitos pareados, IC por blocos, ranking, gráficos |

```bash
# ordem do estudo (cada etapa é um portão); --study isola o namespace no volume
modal run src/modal/feature_study.py --stage prepare --study <nome> \
    --arm-sets anchors,groups,controls                  # confira purge_days e n_arms no meta
modal run src/modal/feature_study.py --stage fit --arms base --models all --triage --study <nome>
modal run src/modal/feature_study.py --stage screen --study <nome>          # congela top_models.json
modal run src/modal/feature_study.py --stage fit --models top5 --study <nome>
modal run src/modal/feature_study.py --stage aggregate --study <nome>       # portão: controles ≈ 0
```

`--stage all` roda prepare → fit → aggregate e **não** inclui `screen` (ele só faz sentido entre
dois `fit`). O `fit` é **idempotente**: se um container falhar, repita o comando. Uma unidade só é
pulada se foi gerada com o **mesmo conjunto de modelos** (`models_mode`), a mesma seed e a mesma
lista de features do arm: o piloto grava 3 modelos na mesma pasta do run completo, e pular por
"o arquivo existe" entregaria métricas incompletas sem aviso.

## Treino completo, sem amostragem

O cluster 3 tem poucas linhas de treino (~6 mil por trimestre) e uma unidade de 39 modelos leva
~25 s. O estudo treina no **treino completo**, sem amostragem: amostras se sobreporiam em ~75 % e
subestimariam a variância. As réplicas são de **seed do modelo** (seção "Seeds fixas"), não de
amostra.

## Arms

| Conjunto | Arms |
|---|---|
| `anchors` (2) | `base` (grupo 1 + latitude/longitude); `full` (todos os grupos). **Exigido** por `groups` e `controls`: sem ele os arms ficam com um `reference` inexistente e `comparisons()` apenas não gera a comparação, sem erro |
| `groups` (7) | `add_grp__{grupo2,grupo3,grupo4}` e `drop_grp__{grupo1,grupo2,grupo3,grupo4}`. O grupo 1 só sai (somá-lo daria um arm idêntico ao `base`) |
| `controls` (2) | `ctrl__noise` (base + ruído); `ctrl__perm_static` (base + estáticas permutadas entre estações) |
| `singles` (2×N, **opt-in**) | `add__f` / `drop__f` por coluna. Roda depois, restrita ao grupo que venceu: árvores respondem mal a "quanto vale UMA coluna" |
| `core` | alias de `anchors` + `singles`, para comandos e `arms.json` antigos |
| `sel__*` (via `add-arms`) | conjunto reduzido e ablações; ver "Seleção de variáveis" |

`era5_groups` ainda é aceito como nome de arm set (e ignorado): era o nome da população, que
hoje é a única. Todos os arms usam **as mesmas linhas**.

## A estrutura por grupo (a única população)

Em 2026-09-30 o `dataset/raw` passou a ter quatro parquets, um por grupo do spec,
em `dataset/raw/feature_study_cluster_3/`. É a população do estudo: um ponto por (estação, dia), na hora do pico da rajada ERA5.

| grupo | formato | colunas | conteúdo |
|---|---|---|---|
| 1 | longo (estação, hora) | 43 | vento 10/100 m, rajada, `bwd_10_100`, `zust`, `blh`, `mbld`, `msshf`, T2m, Td2m, depressão, `sp`, hora solar, dia do ano, vizinhança (`r75`/`r250`), defasagens 1–3 h |
| 2 | longo | 30 | convecção: `cape`, `cin`, `mcpr`, `mtpr`, cisalhamento 0–6 km, *lapse rates*, vizinhança, defasagens |
| 3 | longo | 30 | sinótico: `mslp`, gradiente e tendência, `w850`, `w500`, `z500`, vorticidade, `omega700`, `t850`, vizinhança, defasagens |
| 4 | 1 linha por estação | 82 (63 usadas) | relevo e máscara terra-mar, no ponto e em raios de 75/250 km |

Vizinhança e defasagens **já vêm calculadas** — nada é recalculado aqui. O `cape`
pertence ao **grupo 2**. `cin` tem ~75 % de nulos e sai (sem imputação).

### Como uma linha é montada

O alvo é diário e as features são horárias, então algo liga uma coisa à outra. O
desenho é **um ponto por (estação, dia), na hora do pico da rajada do ERA5**
(`gust10fg`). A hora vem do ERA5, nunca do INMET — sem vazamento do alvo — e é o
que um produto de correção faria em produção. Nessa hora `gust10fg` é, por
construção, o máximo diário: **a linha de base honesta**, guardada como
`era5_gust_max` (referência, fora de todo arm). Dia com qualquer hora de rajada
faltando é descartado inteiro.

### ATENÇÃO: o "ERA5 cru" dos estudos anteriores não era a rajada

Nos notebooks e gráficos anteriores, "ERA5 cru" era `wind_mag_max` — o **máximo do
vento médio** a 10 m, não uma rajada. Medido no teste do estudo RAW (n = 12.665):

| referência | RMSE | RMSE P90 | viés | viés P90 | R² |
|---|---|---|---|---|---|
| `wind_mag_max` (vento médio) | 5,85 | 10,70 | −5,11 | −9,93 | −1,51 |
| **rajada ERA5 bruta** (`fg10`) | 2,65 | 5,47 | −0,03 | −3,32 | 0,484 |
| rajada + calibração linear simples | 2,59 | 5,61 | −0,04 | −4,00 | 0,507 |
| modelo RAW (585 colunas) | 2,24 | 5,07 | −0,05 | −3,33 | 0,631 |

Contra a rajada, o modelo melhora o RMSE em ~15 % e o R² de 0,48 para 0,63, mas
**não reduz o viés de cauda** (−3,32 → −3,33). As afirmações de que o ERA5
"subestima 4 a 7 m/s" e de que o modelo "remove 4,7 a 8,5 m/s de viés na cauda"
valiam para o vento médio, não para a rajada.

### Arms (11)

`base` = grupo 1 (+ lat/lon, que o spec põe no grupo 1). Os grupos 2, 3 e 4 entram
por cima como grupos **inteiros**:

`base`, `full`, `add_grp__grupo{2,3,4}`, `drop_grp__grupo{1,2,3,4}`,
`ctrl__noise`, `ctrl__perm_static`. O grupo 1 só tem `drop_grp__`: somá-lo à base
(que já é ele) daria um arm idêntico. Com 167 colunas no `full` e ~5.700 linhas de
treino por trimestre, são ~34 linhas por coluna (o RAW tinha ~10).

### Split e purga

Mesmo split do estudo anterior: teste = jan/abr/jul/out, validação sorteada em
blocos (ano, mês) sobre o eixo de dias **até 2024** — com 2025 o sorteio troca 15
dos 32 blocos e o estudo deixaria de ser comparável. Purga de **1 dia** (as
defasagens chegam a 3 h; a hora do pico às 00 UTC olha para 21–23 h do dia
anterior, que pode ser de outro split).

### Rodar LOCAL

Não precisa do ERA5 bruto antigo: só os quatro parquets e o `INMET_Stratified.nc`
(alvo). O `prepare` leva segundos e ~650 MB.

```bash
python scripts/run_feature_study_local.py plan                       # o que vai rodar
python scripts/run_feature_study_local.py run --stage prepare
python scripts/run_feature_study_local.py run --stage triage --seeds 42
python scripts/run_feature_study_local.py run --stage screen --seeds 42
python scripts/run_feature_study_local.py run --stage study  --seeds 42
python scripts/run_feature_study_local.py run --stage aggregate --seeds 42
python scripts/run_feature_study_local.py run --stage all --seeds 42,43,44,45,46   # tudo, 5 seeds
python scripts/run_feature_study_local.py status
```

Cuidados pensados para uma máquina com ~4 GB de RAM livres: teto de 4 threads
(`--threads`), prioridade baixa (`nice`), guarda de memória que espera em vez de
congelar o sistema (`--min-free-gb`), leitura dos parquets em lotes (pico ~1,9 GB
só na primeira montagem, que vai para cache). **Tudo é idempotente**: Ctrl-C, queda
de luz ou falta de memória não custam nada — rode o mesmo comando.

Medido nesta máquina: triagem de um trimestre em ~1 min (39 modelos no arm base);
uma unidade de estudo (11 arms × 5 modelos) em ~10 min. **Uma seed completa ≈ 45
min; as 5 seeds ≈ 4 h.**

### Triagem e estudo em pastas separadas

A triagem grava em `units/<tag>_triage/` e o estudo em `units/<tag>/`. Antes
dividiam a pasta, e como os dois gravam o mesmo arquivo para o arm `base`, o estudo
(5 modelos) sobrescrevia a triagem (39): a leaderboard deixava de ser auditável e
rodar `screen` depois escolhia entre os 5 já escolhidos (um top-5 encolheu para
4). Aconteceu no `cluster3_raw` do Modal — os resultados dele seguem válidos, pois
o `top_models.json` foi congelado antes, mas a trilha de auditoria dos 39 modelos
foi perdida.

### Top-5 sem modelos duplicados

Modelos com R² e RMSE_P90 de validação dentro de 0,002 e 0,01 contam **uma** vez. Na
primeira triagem de JJA, `LinearRegression`, `TransformedTargetRegressor`, `RidgeCV`
e `Ridge` ocupavam quatro das cinco vagas: eram a mesma regressão linear. A coluna
`equivalentes_omitidos` de `top_models.parquet` registra quem ficou de fora.

## Seleção de variáveis (`selection/selected.py`)

Responde "quais variáveis manter" depois de "quais grupos". Três passos:

1. **Seleção na validação** (notebook `grupos_1_e_4_importancia_por_variavel`): permutação por
   variável; mantida se supera o piso de uma coluna de ruído e é positiva em >= 3 de 4
   trimestres. Congelada por `python scripts/congelar_selecao_features.py` em
   `config/selected_features_val12.json` (12 variáveis, 58 colunas, impressão digital).
2. **Registrar os arms sem refazer o `prepare`** (o `prepare` regeneraria dados e `row_id`):
   `--stage add-arms` acrescenta `sel__val12` e `sel__val12_sem_anor` a `arms.json`. É
   idempotente e recusa sobrescrever um arm com features diferentes.
3. **Treinar só esses arms** e comparar com `base`/`full` já gravados (mesmas linhas).

```bash
MODAL_PROFILE=<perfil do volume> modal run src/modal/feature_study.py --stage add-arms --study <nome>
MODAL_PROFILE=<perfil do volume> modal run src/modal/feature_study.py --stage fit \
    --study <nome> --arms sel__val12 --models top5          # ~0,1 container-h
```

Comparações do eixo próprio `selection` (família própria no BH, não mexe nos q-valores de
features): `sel__val12` vs `full` (o que se perde ao cortar), `sel__val12_sem_anor` vs
`sel__val12` (o que vale o `anor`), `sel_vs_base` (reduzido vs grupo 1). O `aggregate
--label sel` dá o veredito oficial; o notebook traz uma versão enxuta (sem BH).

`data/base_only.py` lê a BASE antiga (12 variáveis do ERA5 horário) na hora do pico, para um modelo
"sem nenhum grupo" nas mesmas linhas. Resultado: reproduz o ERA5 (RMSE 2,653 contra 2,658).

**Cuidados.** Rode com o perfil do Modal que **tem** o volume: em outro workspace o volume é
recriado vazio e o erro aparece como `FileNotFoundError` de `arms.json`. A permutação divide o
crédito entre colunas colineares e mede uso, não ganho fora da amostra.

## Partição treino / validação / teste

Partição por **blocos de mês** (`MonthBlockSplit`, a mesma de lazy, mlp e LSTM v2), calculada uma vez
sobre o eixo de dias de 2000–2024 do alvo INMET e gravada na coluna `_split`:

| split | definição | linhas (referência) |
|---|---|---|
| teste | meses 1, 4, 7 e 10 de todos os anos (um por trimestre) | 13.583 (97 blocos ano-mês) |
| treino | meses 2, 3, 5, 6, 8, 9, 11, 12 | 22.827 |
| validação | 32 blocos (ano, mês) sorteados nos meses de treino (15 %, estratificado por mês, seed 42) | 4.452 |

- O **trimestre é confundido com o seu mês de teste** (DJF testa em janeiro, MAM em abril…), e o
  intervalo por trimestre tem só 24–25 blocos; só o agregado (97 blocos) tem IC confiável.
- A validação **escolhe** (top-5 de modelos, seleção de variáveis); o teste **pontua**. Nenhuma escolha
  olha o teste. O eixo de dias vai até 2024: com 2025 os blocos de validação mudam (15 de 32) e o
  estudo deixaria de ser comparável ao anterior.
- A purga (seção seguinte) tira 1.030 linhas das fronteiras de mês.
- Esta é a **única** partição do repositório. O split antigo por anos (treino 2000–2018 / validação
  2019 / teste 2020–2025) e o `cluster_gan` que o usava saíram: o GAN foi extraído para o repositório
  `irc_vendaval_gan`, e as constantes `TRAIN_SLICE`/`VAL_SLICE`/`TEST_SLICE` foram removidas.

## Purga temporal nas fronteiras de mês

As colunas com defasagem (`*_lag1h`, `*_max_prev3h`, `*_delta3h`, `mslp_tend_3h`) olham para trás e
podem atravessar a fronteira entre meses de treino e de teste. A hora do pico pode ser 00 UTC, e aí
`max_prev3h` alcança 21–23 h do dia anterior.

O vazamento é o **inverso** do que parece: não é 1º de janeiro (teste) puxando dezembro (treino) —
isso é a realidade operacional. É 1º de **fevereiro** (treino) olhando para 31 de janeiro (teste).
O mesmo em maio, agosto e novembro: todo mês de teste é seguido por um de treino.

`prepare` reusa `purge_keep` (`src/pipeline/data/splits.py`, a regra estrita que a LSTM já aplica),
e o gap é **derivado** do conjunto de features (`data/purge.purge_days`, pelo sufixo de defasagem):
3 h de alcance pedem **1 dia**. Gap **único** para todos os arms, porque `compute_effects` é
pareado por `row_id`. O gap e as linhas removidas vão para o `meta.json`.

## Triagem top-5 por trimestre (`screen`)

Rodar os 39 regressores do LazyPredict em todos os arms produz 39 rankings cruzados por
trimestre, nenhum dos quais é o modelo que vai para produção. A triagem inverte a ordem:

```bash
# 1. os 39 modelos, SÓ no arm base
modal run src/modal/feature_study.py --stage fit --arms base --models all
# 2. congela os 5 melhores por trimestre
modal run src/modal/feature_study.py --stage screen
# 3. o estudo inteiro roda só com eles
modal run src/modal/feature_study.py --stage fit --models top5
```

A regra é a de `champion_rules["r2_slack_then_rmse_p90"]`: entre os modelos a até 0,05 do
melhor R², fica o de menor `RMSE_P90`. Escolha sempre na **validação**, métricas de teste
apenas reportadas. Se a folga admitir menos de 5, completa por R² — um trimestre com menos
modelos que os outros desbalancearia a média por trimestre da análise.

O resultado é **congelado** em `summary/top_models.json` e cada container do `fit` apenas o
lê. Recalcular por unidade faria uma seed a mais no volume mudar o conjunto de modelos no
meio do fan-out, e metade dos arms sairia com modelos diferentes da outra metade.

**Sobre a mudança metodológica:** `config.py` documenta a decisão de FIXAR os modelos a
priori, porque escolher o top-K pelo arm base geraria regressão à média contra as features.
Isso continua valendo para `REFERENCE_MODELS`. Aqui o desenho é outro e a condição do
bootstrap é preservada: o **mesmo** conjunto de 5 vale para todos os arms daquele trimestre,
então o pareamento de `compute_effects` (mesmo modelo, trimestre e seed dos dois lados)
continua intacto. O que se perde é a garantia de que a lista não foi tocada pelos dados — e
o viés resultante é **conservador contra** as features novas, já que um modelo que só
brilhasse COM elas nunca é visto.

Dois cuidados que o código impõe:

- `models_mode` carrega a **impressão digital** do conjunto (`custom:<md5>`). Sem ela, o
  top-5 do DJF e o do JJA carimbariam ambos `"custom"`, `_is_current` os consideraria
  equivalentes e o `skip_existing` devolveria o ajuste errado — sem erro, sem aviso, com
  métricas plausíveis.
- O resíduo por linha passa a ser gravado para os modelos **efetivamente ajustados**, não
  para a tupla fixa `REFERENCE_MODELS`: um campeão fora dela (GradientBoosting, Bagging,
  AdaBoost são candidatos reais) ficaria sem resíduo e `compute_effects` cego.

## Série temporal e mapa espacial

O `fit` grava, para os arms de `worker.PRED_DUMP_ARMS` (`base` e `full`),
`units/<tag>/preds__<season>__<arm>.parquet` no schema de
`build_station_predictions_frame` (`src/pipelines/metrics_schema.py`): `estacao`,
`latitude`, `longitude`, `time`, `y_true`, `y_pred`, `era5_proxy`, `model`, `arm`,
`season`, `tag`, `model_seed`. É o insumo da série temporal (melhor/médio/pior caso) e do
mapa de `corr(modelo, ERA5)` por estação, sem precisar reconstruir predição a partir do
resíduo.

O rank-1 de cada trimestre é persistido em
`units/<tag>/fitted_models/best_model_c3_<season>.joblib` (`core/artifacts.py`),
no **mesmo schema** que o `cluster_lazy` produz — é o que destrava o mapa em GRADE via
`SpatialCorrector` sem retreinar nada. Só o arm âncora e só a seed base: são 4 arquivos, e
com 681 colunas um ExtraTrees passa de 100 MB (tudo sai com `compress=3`).

**O mapa terá 9 estações, não 21.** O arquivo horário cobre 21 estações do cluster 3 porque
é ERA5 interpolado, mas não há alvo INMET para 12 delas em 2000–2024: `B808/809/810/811/812/
819/820` só reportam a partir de 2025-09-09 e `B813/814/816/818/825` de 2025-10-13. Atenção
a `B807` — 1.016 dias, só desde 2022-12-08: ele pode dominar o "pior caso" por escassez de
amostra, não por erro do modelo, então reporte o `n` por estação ao lado da métrica.

## Regras de campeão (`champion_rules.csv`) — medição, não produção

Maximizar R² é minimizar erro quadrático, cujo minimizador é a média condicional: **a
seleção por R² prefere sistematicamente o modelo mais comprimido**. Medido no cluster 3, há
de 0,6 a 1,5 m/s de viés de cauda que o R² nunca encontra — 3 a 8× tudo o que as features
novas deram.

O `aggregate` compara três regras sobre os resultados que já existem: `r2`, `rmse_p90` e
`r2_slack_then_rmse_p90` (entre os modelos a até `CHAMPION_SLACK`=0,05 do melhor R², o de
menor RMSE_P90). A escolha é sempre na **validação** e a pontuação sempre no **teste**.

Sem a folga, a melhor cauda pode ser um modelo inútil: no DJF era o RANSAC, com R² = −0,349.
**Nada disso altera a produção** — `src/dataset/creation/best_model_selector.py` segue com
`metric="R2"`. É medição para decidir a troca com número na mão.

## Como o efeito é medido (`analysis.py`)

O campeão do LazyPredict oscila entre arms (efeito da feature confundido com troca de
algoritmo), então o efeito **não** é medido nele. É medido **por modelo e pareado** num
conjunto de referência **fixado a priori** — CatBoost, LightGBM, XGBoost, HistGB, ExtraTrees,
RandomForest, Ridge — e reportado por família. Os 39 modelos ficam como tabela descritiva.

- **Ganho** (positivo = o arm melhor é melhor): RMSE e RMSE_P90 = erro(pior) − erro(melhor);
  R² = R²(melhor) − R²(pior). Bias@P90 é reportado como **deslocamento com sinal**, não
  como ganho (o viés é negativo e pode cruzar zero). **Primária: RMSE.**
- **A_f** = ganho isolado (`base + f` vs `base`). **L_f** = perda ao remover (`full` vs `full − f`).
  `A_f > 0` com `L_f ≈ 0` sugere redundância com as demais novas — leitura heurística.
- **IC:** bootstrap em **blocos (ano, mês)** do teste, com o *mesmo* sorteio para todos os
  arms e modelos (pareado). Cada seed é uma réplica de treino (R = 5): o efeito é a média
  entre seeds e `se = √(se_boot² + se_rep²)`, com `se_rep` = desvio entre seeds / √R. Com
  uma só seed (R = 1), `se_rep = 0` e a incerteza é só a do bootstrap.
- **SESOI = 1% do RMSE do base.** "Sem informação" = IC dentro de ±SESOI — não "IC cruza zero".
- **BH** sobre as comparações (RMSE, todos os modelos de referência).

**Quantos blocos existem, de fato.** Cada trimestre testa num **único mês** (DJF em
janeiro, MAM em abril…), então o bootstrap tem apenas **24–25 blocos por trimestre** — um
por ano, com ~145 linhas cada:

| trimestre | mês de teste | blocos | linhas |
|---|---|---|---|
| DJF | jan | 24 | 3.489 |
| MAM | abr | 24 | 3.429 |
| JJA | jul | 24 | 3.512 |
| SON | out | 25 | 3.620 |

Com ~24 unidades independentes o IC é estreito por construção e tende a ser
**anticonservador**; a purga não muda esse número (tira dias dentro dos blocos, não
blocos). Por isso `effects.csv` traz a coluna `n_blocos` ao lado de cada IC, e o SESOI
continua sendo exigido **além** do q-valor: com 24 blocos, significância sozinha não é
evidência.

O bloco `(ano, mês)` agrupa **todas as 9 estações** daquele mês, então ele absorve não só a
autocorrelação temporal como a **dependência entre estações** no mesmo dia — que na mesma
bacia é alta. Blocos por estação seriam piores.

**Controles negativos.** `ctrl__perm_static` permuta as estáticas **no nível da estação**:
um desarranjo (deslocamento cíclico sobre ordem aleatória, sem ponto fixo) em que cada
estação recebe as 7 estáticas de **outra**, em bloco. Preserva a marginal e a estrutura
conjunta das 7; quebra só o vínculo com a estação verdadeira. Permutar linha a linha
destruiria essa estrutura e tornaria o controle trivialmente nulo — mascarando o problema
em vez de detectá-lo.

## Seeds fixas — o que isso significa

Uma seed fixa dá **reprodutibilidade**, não elimina o acaso: congela um único sorteio da
aleatoriedade dos modelos, sem dizer o quão grande ela é. Por isso o estudo usa **5 seeds
fixas** (`MODEL_SEEDS` = 42…46 em `config.py`, gravadas em `data/meta.json`), cada uma uma
réplica de treino sobre os mesmos dados e arms. Rodar de novo reproduz cada seed bit a bit
(o piloto repetiu uma unidade com os 39 modelos e a diferença foi **exatamente 0**).

- A seed 42 mantém a pasta `units/full/` (o run anterior vale como réplica 1); as demais
  ficam em `units/full_s43/` … `full_s46/`. O `fit` só pula uma unidade se modelo-set **e**
  seed conferem.
- **Nem todo modelo depende da seed.** Ridge, Lasso, LinearRegression, Huber etc. são
  determinísticos: repeti-los não acrescenta informação. O `aggregate` grava
  `seed_sensitivity.csv` (desvio do RMSE entre seeds por modelo; `estocastico` = a seed o
  muda) e `models_leaderboard_by_seed.csv`. O notebook usa os dois na seção 10.
- As seeds cobrem só a incerteza de **treino**. A de **teste** continua sendo o bootstrap em
  blocos, e o trimestre continua confundido com o seu mês de teste (jan/abr/jul/out).
- Os controles (`ctrl__noise`, `ctrl__perm_static`) são gerados no `prepare` com uma seed
  só: cada um é UM sorteio de ruído / permutação, comum às 5 seeds do modelo.

## Leitura correta dos resultados

- O **teste é reutilizado** por todos os arms × modelos. Não há seleção de modelo nele, mas ele
  vira o "teste do estudo": **não** estima o desempenho final de um modelo escolhido com base
  nele.
- Com 97 blocos (ano, mês) no teste, o IC de efeitos pequenos (~1 % do RMSE) é largo.
- Custo medido no Modal: ~29 s por ajuste de arm × trimestre × seed (5 modelos); o estudo de
  referência levou ~2 container-horas (triagem 0,2 h, estudo 1,8 h).

## Saídas (`feature_study/<estudo>/` no volume)

```
data/        meta.json, arms.json, test.parquet, sample_full.parquet
units/<tag>/ metrics__<season>__<arm>.parquet, resid__<season>__<arm>.parquet,
             preds__<season>__<arm>.parquet (base/full), fitted_models/ (campeão)
             <tag> = full, full_s43 … full_s46 (seeds); <tag>_triage = os 39 modelos no arm base
summary/<label>/  effects.csv, ranking_features.csv, ranking_groups.csv, effects_by_season.csv,
                  models_leaderboard.csv, champions.csv, ranking_features.png, champion_rules*.csv
                  [>1 seed] models_leaderboard_by_seed.csv, seed_sensitivity.csv
summary/top_models.json   os 5 melhores por trimestre (congelado pelo `screen`)
robustez/                 saídas de scripts/analise_robustez.py
```

`effects.csv` tem a coluna `axis` (`features` | `selection`): `q_bh`, `sesoi` e `verdict` são
calculados **dentro** de cada eixo.
