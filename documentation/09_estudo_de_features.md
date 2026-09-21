# Estudo de informação de features novas (`src/feature_study/`)

Pipeline **isolada** — não altera nenhuma pipeline de treino — que responde: *quais
das features `nf_*` acrescentam informação a um modelo?* Compara modelos treinados
COM e SEM cada feature. **Somente cluster 3** e **seeds fixas** (não são flags:
`src/feature_study/config.py`). Toda execução real é no Modal.

## Por que existe

A comparação "base vs. base + 7 estáticas" trata as estáticas como um bloco, foi
treinada uma vez e tinha IC de bootstrap de *linhas* (ignora autocorrelação). Ela
não separa "o modelo usa a feature" (SHAP) de "a feature acrescenta informação".

## Estágios

| Estágio | Onde | O que faz |
|---|---|---|
| `prepare` | 1 container (32 GB) | carrega dados, aplica completude da **união** de features, grava treino/teste em parquet |
| `fit` | fan-out (~32 containers **por seed**) | LazyPredict **inteiro** (39 modelos) por (seed × trimestre × lote de 5 arms), cada trimestre no seu container |
| `aggregate` | 1 container | efeitos pareados, IC por blocos, ranking, gráficos |

```bash
modal run src/modal/feature_study.py --stage all        # prepare → fit → aggregate
# ou, por etapa:
modal run src/modal/feature_study.py --stage prepare
modal run src/modal/feature_study.py --stage pilot      # custo e determinismo (opcional)
modal run src/modal/feature_study.py --stage fit                        # as 5 seeds (as já prontas são puladas)
modal run src/modal/feature_study.py --stage fit --seeds 43,44,45,46    # só as novas
modal run src/modal/feature_study.py --stage aggregate                  # combina as 5 seeds

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

## Arms (37 + 10 opcionais)

| Conjunto | Arms |
|---|---|
| `core` (28) | `base` (40 features); `full` (40 + 13 novas); 13× `add__f` (base + f); 13× `drop__f` (full − f) |
| `groups` (7) | `add_grp__{static,dynamic}`; `drop_grp__{static,dynamic,topography,fg10,cape}` |
| `controls` (2) | `ctrl__noise` (base + ruído); `ctrl__perm_static` (base + estáticas permutadas entre estações) |
| `losses` (10, **opt-in**) | `{base,full}__{exp60,exp70,exp80,exp90,huber44}` — mesmas features, **perda** diferente |

`nf_ws10_max/mean` ficam de fora: idênticas a `ws_max/ws_mean` (diferença medida 0,0000 m/s).
Todos os arms usam **as mesmas linhas**.

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
