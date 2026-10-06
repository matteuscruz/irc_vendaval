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
- **Onde ver:** `notebooks/README.md` (índice) e `notebooks/feature_study_grupos/atual/`.
- **Seções históricas:** da "Modo RAW horário" até "Eixo de perda" descrevem desenhos
  anteriores (RAW horário, diário × horário, perda). Seguem corretas para o que descrevem,
  mas **não** são o desenho vigente.

## Por que existe

A comparação "base vs. base + 7 estáticas" trata as estáticas como um bloco, foi
treinada uma vez e tinha IC de bootstrap de *linhas* (ignora autocorrelação). Ela
não separa "o modelo usa a feature" (SHAP) de "a feature acrescenta informação".

## Estágios

| Estágio | Onde | O que faz |
|---|---|---|
| `prepare` | 1 container (32 GB) | carrega dados, aplica completude da **união** de features, aplica a **purga temporal**, grava treino/teste em parquet |
| `fit` | fan-out (~32 containers **por seed**) | LazyPredict por (seed × trimestre × lote de 5 arms), cada trimestre no seu container |
| `screen` | 1 container | congela os **5 melhores modelos por trimestre** a partir do arm `base` → `summary/top_models.json` |
| `aggregate` | 1 container | efeitos pareados, IC por blocos, ranking, gráficos |

```bash
modal run src/modal/feature_study.py --stage all        # prepare → fit → aggregate
                                                        # (NÃO inclui `screen`: ele só faz
                                                        #  sentido entre dois `fit`)
# ou, por etapa:
modal run src/modal/feature_study.py --stage prepare
modal run src/modal/feature_study.py --stage pilot      # custo e determinismo (opcional)
modal run src/modal/feature_study.py --stage fit                        # as 5 seeds (as já prontas são puladas)
modal run src/modal/feature_study.py --stage fit --seeds 43,44,45,46    # só as novas
modal run src/modal/feature_study.py --stage aggregate                  # combina as 5 seeds

# Estudo RAW horário + triagem top-5 (ordem obrigatória, cada etapa é um portão):
modal run src/modal/feature_study.py --stage prepare \
    --arm-sets anchors,groups,controls,hourly_raw       # confira purge_days e n_arms no meta
modal run src/modal/feature_study.py --stage pilot      # portão: 266 colunas cabem na memória?
modal run src/modal/feature_study.py --stage fit --arms base --models all
                                                        # portão: R² do RAW vs base diário
modal run src/modal/feature_study.py --stage screen     # congela top_models.json
modal run src/modal/feature_study.py --stage fit --models top5
modal run src/modal/feature_study.py --stage aggregate  # portão: controles negativos ≈ 0

# Eixo de PERDA (opt-in): reexecuta o prepare com os 10 arms novos e ajusta só eles.
modal run src/modal/feature_study.py --stage prepare --arm-sets core,groups,controls,losses
modal run src/modal/feature_study.py --stage fit --arms full__exp90 --seeds 42   # fumaça
modal run src/modal/feature_study.py --stage fit
modal run src/modal/feature_study.py --stage aggregate
```

Depois do `prepare` com `losses`, **confira o `meta.json`**: `n_test_rows` (14.816) e
`n_trainval_rows` (29.011) têm de ficar idênticos. Se mudarem, os `row_id` se deslocaram e
todo `resid__*.parquet` já gravado está desalinhado — `load_residuals` levantaria em
`pos.loc[...]`.

O `fit` é **idempotente**: se um container falhar, repita o comando. Uma unidade só é
pulada se foi gerada com o **mesmo conjunto de modelos** (coluna `models_mode`): o piloto
grava 3 modelos na mesma pasta do run completo, e pular por "o arquivo existe" entregaria
métricas incompletas sem aviso.

## Treino completo, sem amostragem

O cluster 3 tem **24.814 linhas de treino** no total (~6 mil por trimestre) e 4.197 de
validação. Uma unidade de 39 modelos leva ~25 s. Por isso o estudo treina no **treino
completo**:

- A estimativa inicial de ~29 mil linhas por trimestre estava errada: vinha do
  `cluster_lazy`, que soma o cluster vizinho ao treino (`n_neighbor_clusters=1`). Este estudo
  não usa vizinhos.
- Com a população tão pequena, uma amostra DKW (ε = 0,02 ⇒ 4612) já seria ~75% do treino, e
  amostras de n ≥ ~6 mil são o dataset inteiro (o piloto confirmou: `n10000` = `full`).
- Réplicas **amostradas** se sobreporiam em ~75%, subestimando a variância entre elas. As
  réplicas do estudo são, então, de **seed do modelo** (seção abaixo), não de amostra.

O código de amostragem (`sampling.py`) continua **testado e disponível** — passe `seeds` e
`pilot_sizes` a `prepare()` para reativá-lo (por exemplo, para clusters maiores).

## Arms

| Conjunto | Arms |
|---|---|
| `anchors` (2) | `base`; `full` (base + novas). **Exigido** por `groups`, `controls` e `losses`: sem ele os arms ficam com um `reference` inexistente e `comparisons()` apenas não gera a comparação — o estudo roda inteiro e devolve uma tabela de efeitos vazia, sem erro |
| `singles` (2×N, **opt-in**) | `add__f` (base + f) e `drop__f` (full − f), uma por feature. É esta varredura que produz ~100 arms, para responder "quanto vale UMA coluna" — pergunta que ensembles de árvores respondem mal (capturam interação internamente) e que não é a de negócio. Rode-a **depois**, restrita ao grupo que venceu |
| `core` | alias de `anchors` + `singles`, para comandos e `arms.json` antigos |
| `groups` (7) | `add_grp__{static,dynamic}`; `drop_grp__{static,dynamic,topography,fg10,cape}` |
| `controls` (2) | `ctrl__noise` (base + ruído); `ctrl__perm_static` (base + estáticas permutadas entre estações) |
| `losses` (10, **opt-in**) | `{base,full}__{exp60,exp70,exp80,exp90,huber44}` — mesmas features, **perda** diferente |
| `hourly` (18+2, **opt-in**) | `add__nf_h_*`/`drop__nf_h_*` + `add_grp__hourly`/`drop_grp__hourly` — segunda fonte de features, ver abaixo |

`nf_ws10_max/mean` ficam de fora: idênticas a `ws_max/ws_mean` (diferença medida 0,0000 m/s).
Todos os arms usam **as mesmas linhas** — exceto quando `hourly` está fora, caso em que as
colunas `nf_h_*` nem existem na população (ver seção seguinte).

## Estrutura nova por grupo (`era5_groups`) — o modo vigente

Em 2026-09-30 o `dataset/raw` passou a ter quatro parquets, um por grupo do spec,
em `dataset/raw/feature_study_cluster_3/`. É o desenho que substitui o RAW
horário como base do estudo.

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

## Seleção de variáveis (`selected.py`)

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

`base_only.py` lê a BASE antiga (12 variáveis do ERA5 horário) na hora do pico, para um modelo
"sem nenhum grupo" nas mesmas linhas. Resultado: reproduz o ERA5 (RMSE 2,653 contra 2,658).

**Cuidados.** Rode com o perfil do Modal que **tem** o volume: em outro workspace o volume é
recriado vazio e o erro aparece como `FileNotFoundError` de `arms.json`. A permutação divide o
crédito entre colunas colineares e mede uso, não ganho fora da amostra.

## Modo RAW horário (`hourly_raw`, opt-in) — nada de diário

Os três modos horários respondem perguntas diferentes e são **mutuamente exclusivos**:

| arm_set | o que muda | pergunta |
|---|---|---|
| `hourly` | SOMA 18 agregados `nf_h_*` às novas | "o horário ajuda **além** do que já temos?" |
| `hourly_only` | TROCA o pool de NOVAS por `nf_h_*`, base diária intacta | "o horário **sozinho** ajuda?" |
| `hourly_raw` | TROCA a **base**: nada de diário em lugar nenhum | "a agregação diária está **destruindo** o sinal?" |

O RAW é a representação mais crua das três. `mean`/`max` de 24 horas colapsam o perfil
intradiário, que é justamente o que separa uma rajada convectiva de fim de tarde de um
evento sinótico — o diagnóstico de DJF mostrou que é aí que o ERA5 falha. Aqui nada é
colapsado: as 24 horas viram **24 colunas**, e o modelo decide sozinho o que olhar.

| bloco | origem | vars | colunas |
|---|---|---|---|
| `hf_*` (base) | `dataset/raw/test_cluster_3_hourly.nc`, já por estação | 11 | **264** |
| `hfn_*` (novas) | `new_features/cluster3/sl/*.nc`, grade → pontos | 17 | **408** |
| `nf_*` (estáticas) | `new_features/cluster3/static/` | 7 | **7** |
| + `latitude`, `longitude` | | | 2 |
| **total (`full`)** | | | **681** |

`wd_h` fica de fora: direção em graus não interpola em 0°/360° (359 vs 1 viraria uma
diferença de 358); `sin_dir_h`/`cos_dir_h` cobrem. As `WIND_PAIRS` (`ws10`, `ws100`,
`shear_100_10`) são compostas **hora a hora**, nunca a partir de médias diárias — por
Jensen, `sqrt(u²+v²)` da média subestima a média de `sqrt(u²+v²)`, e num projeto de
extremos esse erro não apareceria em lugar nenhum do resultado final.

Os buckets `static/dynamic/topography/fg10/cape` não fazem sentido quando tudo é horário,
então o RAW usa **blocos temáticos** (`arms.THEMATIC_NEW` / `THEMATIC_BASE`), roteados pela
variável de origem: `vento`, `termo`, `convec`, `superf`, `static`, mais `base_vento`,
`base_termo` e `base_precip`. Os blocos da base só geram `drop_grp__` — somá-los à própria
base daria um arm idêntico ao `base`; a pergunta legítima é a inversa ("o perfil horário
cru carrega peso próprio, ou as variáveis novas já o substituem?"). Uma coluna que não case
com nenhum bloco **levanta**: ela escaparia de todo `drop_grp__` e o efeito medido seria
zero por construção, parecendo uma conclusão em vez de um bug.

```bash
modal run src/modal/feature_study.py --stage prepare \
    --arm-sets anchors,groups,controls,hourly_raw
```

### O risco central, declarado

**681 colunas contra ~5.900 linhas de treino por trimestre ≈ 9 linhas por coluna.** Nessa
faixa uma árvore consegue decorar `(estação, data)` em vez de aprender física. Por isso, no
modo RAW, os controles negativos deixam de ser diagnóstico e viram o **portão de validade**:
`ctrl__noise` e `ctrl__perm_static` têm de dar efeito ≈ 0 com IC cruzando zero. Se qualquer
um der ganho aparente, o estudo mediu memorização e **nenhum outro número dele vale** — o
recuo é subamostrar as horas (de 3 em 3 ⇒ ~230 colunas).

Os controles detectam ganho espúrio de **feature**, não sobreajuste global: com 9 estações,
o modelo pode identificar a estação pelo próprio perfil horário. O termômetro para isso é a
coluna `gap_R2_val_test` de `top_models.parquet`.

## Purga temporal nas fronteiras de mês

Até o modo RAW o estudo não purgava nada, e não precisava: as 40 features diárias eram
todas do próprio dia. Três colunas do RAW olham para trás — `tp_roll24h` (1 dia),
`tp_roll48h` (2) e `tp_roll72h` (3) — e atravessam a fronteira entre meses de treino e de
teste.

O vazamento é o **inverso** do que parece. Não é 1º de janeiro (teste) puxando dezembro
(treino): isso é a realidade operacional, um modelo em produção tem mesmo o passado
recente. É 1º–3 de **fevereiro**, que são **treino**, com `tp_roll72h` alcançando 30–31 de
janeiro, que é **teste**. Uma linha de treino passa a conter informação de dias de teste. O
mesmo em maio, agosto e novembro — todo mês de teste é seguido por um mês de treino.

`prepare` reusa `purge_keep` (`src/pipeline/data/splits.py`), a mesma regra estrita que a
LSTM já aplica, e o gap é **derivado** do conjunto de features
(`hourly_flat.purge_days`), nunca fixado: 3 hoje, 2 sem `tp_roll72h`, 0 sem nenhuma
`tp_roll*`. Gap **único** para todos os arms, porque `compute_effects` é pareado por
`row_id` e um gap por arm quebraria o pareamento sem erro visível.

Custo medido na população real (cluster 3, 2000–2024), com gap = 3:

| split | antes | depois | queda |
|---|---|---|---|
| teste | 14.050 | 12.665 | 9,9 % |
| treino | 23.268 | 21.951 | 5,7 % |
| validação | 4.587 | 4.183 | 8,8 % |

O gap e as linhas removidas vão para o `meta.json` (`purge_days`, `rows_purged`).

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
`units/<tag>/fitted_models/best_model_c3_<season>.joblib` (`feature_study/artifacts.py`),
no **mesmo schema** que o `cluster_lazy` produz — é o que destrava o mapa em GRADE via
`SpatialCorrector` sem retreinar nada. Só o arm âncora e só a seed base: são 4 arquivos, e
com 681 colunas um ExtraTrees passa de 100 MB (tudo sai com `compress=3`).

**O mapa terá 9 estações, não 21.** O arquivo horário cobre 21 estações do cluster 3 porque
é ERA5 interpolado, mas não há alvo INMET para 12 delas em 2000–2024: `B808/809/810/811/812/
819/820` só reportam a partir de 2025-09-09 e `B813/814/816/818/825` de 2025-10-13. Atenção
a `B807` — 1.016 dias, só desde 2022-12-08: ele pode dominar o "pior caso" por escassez de
amostra, não por erro do modelo, então reporte o `n` por estação ao lado da métrica.

## Diário × diário+horário (`hourly`, opt-in)

Segunda fonte de features novas, `src/feature_study/hourly_source.py`: agrega
`dataset/raw/test_cluster_3_hourly.nc` (já por ESTAÇÃO, sem grade lat/lon — não passa pelo
`new_features/` porque não há grade para o loader validar) de hora em dia, sem imputação
(um dia com qualquer hora faltando é descartado inteiro). Prefixo `nf_h_` (não `nf_`) para
não colidir por nome com as features da grade (`nf_t2m_max` do grupo 1 é uma variável
diferente de `nf_h_t2m_max`).

18 features: `ws_h` → mean/max/std/**jump** (maior salto absoluto entre horas
consecutivas — pega uma rajada de 1-2h que o mean/max/std diário dilui, motivado pela
investigação do DJF); `msl_h`/`t2m_h` → mean/max/min/**range** (amplitude diurna — frente
ou tempestade); `td_dep_h` (depressão do ponto de orvalho) → mean/max (potencial
convectivo); `rh_h`, `tp_h`, `tp_roll24h`, `sin/cos_dir_h` → estatística padrão.

**É opt-in de propósito** — sem `hourly`/`hourly_only` em `--arm-sets`, o `prepare` **nem
lê** o arquivo horário, e a população fica idêntica a como era antes dessa fonte existir.
Três escolhas, três perguntas diferentes:

```bash
# só diário (padrão, 73 arms) — a linha de base
modal run src/modal/feature_study.py --stage prepare --arm-sets core,groups,controls

# diário + horário (111 arms) — "o horário ajuda ALÉM do que já temos?"
modal run src/modal/feature_study.py --stage prepare --arm-sets core,groups,controls,hourly

# só horário (base 40 + 18 nf_h_*, sem as 31 nf_* da grade) — "o horário SOZINHO ajuda?"
modal run src/modal/feature_study.py --stage prepare --arm-sets core,groups,controls,hourly_only
```

`hourly` e `hourly_only` são **exclusivos** (`prepare` recusa os dois juntos): o primeiro
soma ao pool de features novas, o segundo troca — a base de 40 nunca muda, só o que entra
como candidato a `add__`/`drop__`.

**Controle `perm_static`:** as estáticas têm só 21 valores (um por estação) e
`latitude`/`longitude` já estão nas 40 originais, então elas podem funcionar apenas como
*ID de estação*. A permutada mantém "identifica a estação" e quebra a física — a
comparação `real_vs_perm_static` mede o que a estática real acrescenta além disso.

## Eixo de perda (`losses`, opt-in)

O segundo eixo do estudo: não mede o que o modelo **enxerga**, mede o que ele é instruído
a **minimizar**. Existe porque as 13 features novas juntas movem o `Bias_P90` em só
+0,18 m/s sobre um viés de ~−3,0 — elas não são a alavanca dos extremos. Um piloto
(LightGBM, cluster 3) mediu que a perda move muito mais:

| perda | RMSE | Bias_P90 | RMSE_P90 |
|---|---|---|---|
| MSE (atual) | 2,228 | −3,06 | 4,90 |
| Huber δ=4,4 | 2,214 | **−3,12** | 4,94 |
| expectila τ=0,8 | 2,335 | **−2,34** | 4,47 |

**Expectila** (`exp60`…`exp90`): quadrática assimétrica. `e = pred − y`, peso `tau` quando
`e < 0` (subestimar) e `1 − tau` quando `e > 0`. Em **τ = 0,5 ela É o MSE** — invariante
verificado por teste. τ regula a inclinação sem desancorar a média, ao contrário do
quantil puro, que no piloto levou o viés geral a +1,70.

**Huber δ=4,4** (`huber44`) é **diagnóstico, não candidato**: acima de δ ela troca MSE por
MAE, limitando o gradiente justamente nos erros grandes — 27,9% dos dias acima do P90 caem
nessa zona. É o δ efetivo da LSTM (`Huber(delta=1.0)` sobre alvo escalonado por
RobustScaler, 1 IQR = 4,4 m/s), e o braço mede quanto do viés de cauda dela vem da perda.

Três decisões de implementação, cada uma com uma armadilha atrás:

- **Só 3 modelos** (`LOSS_MODELS`: CatBoost, LGBM, XGBoost). HistGB não tem expectila;
  ExtraTrees/RandomForest (só `criterion`) e Ridge, também não. Incluir o HistGB apenas no
  braço Huber deixaria o conjunto irregular **entre perdas** e confundiria o contraste
  Huber×expectila com a troca de modelos. As famílias `bagging`/`linear` simplesmente não
  aparecem nas comparações de perda, e `n_models` = 3 registra isso.
- **Não existe arm MSE do eixo:** a âncora (`base`/`full`) já é o braço MSE. Ela roda com 7
  modelos e o braço de perda com 3; `compute_effects` exige os dois lados finitos por
  (tag, modelo) e restringe a comparação aos 3 comuns sozinho.
- **Alvo centrado** (`losses._CenteredTarget`): objetivo customizado desliga o
  `boost_from_average` do LightGBM e o `base_score` do XGBoost, e o boosting partiria de 0
  deixando um deslocamento constante para BAIXO (medido: −0,18 m/s) — exatamente a direção
  que o estudo mede. O resíduo não muda com a centralização, então o δ da Huber continua
  significando a mesma coisa.

**Métrica primária por eixo** (`config.PRIMARY_METRIC`): features → `rmse`; perda →
`rmse_p90`. Julgar o eixo de perda por RMSE o condenaria sempre, porque um modelo expectila
perde em RMSE **por construção**. O Benjamini-Hochberg é aplicado **por eixo**: são
famílias de hipóteses distintas, e juntá-las mudaria os q-valores já publicados do eixo de
features só por existir um eixo novo.

`loss_frontier.csv` põe ganho e preço na mesma linha (`rmse_p90_gain` × `rmse_cost`);
`loss_by_model.csv` mostra o efeito por modelo, porque o XGBoost responde bem menos a τ que
CatBoost e LGBM e diluiria a média sem que ninguém visse.

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

- O **teste é reutilizado** por ~37 arms × modelos. Não há seleção de modelo nele, mas ele
  vira o "teste do estudo": **não** estima o desempenho final de um modelo escolhido com base
  nele.
- Com ~104 blocos (ano, mês) no teste, o IC de efeitos pequenos (~1% do RMSE) é largo.
- Piloto: uma unidade (1 arm × 1 trimestre × 39 modelos) usa ~1,3 GB e ~25 s; o run completo
  (37 arms × 4 trimestres) é da ordem de 1,5–3 container-horas.

## Amostragem opcional (`sampling.py`)

Desligada por padrão. Se reativada: **alocação proporcional** por estrato (estação × mês ×
faixa do alvo), nunca sobre-amostrando a cauda; seleção por rank de hash da chave, de modo
que a amostra dependa só de (seed, chave, alvo) — **nunca das features** — e seja idêntica
entre arms; `n = ceil(ln(2/α)/(2ε²))` como heurística (a DKW assume iid). Gates
*estruturais* bloqueiam (chaves únicas, sem vazamento para o teste, sem NaN); os
*estatísticos* são relatados, e só uma falha grosseira (> 2× o limiar) bloqueia. Todo
limiar escala com n.

## Saídas (`feature_study/cluster3/` no volume)

```
data/        meta.json, arms.json, test.parquet, sample_full.parquet
units/full/  metrics__<season>__<arm>.parquet, resid__<season>__<arm>.parquet
summary/<label>/  effects.csv, ranking_features.csv, ranking_groups.csv,
                  effects_by_season.csv, models_leaderboard.csv, champions.csv,
                  ranking_features.png, champion_rules.csv,
                  champion_rules_summary.csv
                  [só com o eixo de perda] loss_frontier.csv, loss_by_model.csv
                  [só com >1 seed]         models_leaderboard_by_seed.csv,
                                           seed_sensitivity.csv
```

`effects.csv` ganhou a coluna `axis` (`features` | `loss`): `q_bh`, `sesoi` e `verdict` são
calculados **dentro** de cada eixo, na métrica primária dele.
