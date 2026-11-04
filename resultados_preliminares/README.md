# Resultados preliminares — como chegamos às 12 variáveis do Cluster 3

Dados para o pôster. Os arquivos seguem **o mesmo caminho percorrido no projeto**, na mesma ordem e
com os mesmos instrumentos. Tudo foi refeito localmente ou compilado a partir do que o estudo já
gravou, e as conferências mostram que os números batem com os originais.

- Gerado por `scripts/gerar_resultados_preliminares.py`:
  `uv run python scripts/gerar_resultados_preliminares.py [--etapas compilar,permutacao,grupos]`.
- Entradas: `resultados_lfs/feature_study/cluster3_groups{,_modal}/` (`git lfs pull`).
- Detalhes do estudo: `documentation/09_estudo_de_features.md`.

> Estes parquets ficam **fora do Git LFS** (cada um tem menos de 30 MB): o envio ao
> `lfs.github.com` estava bloqueado no ambiente em que foram gerados.

---

## 0. Problema e população

- **Alvo:** rajada máxima diária observada pelo INMET (`daily_wind_gust_max`), Cluster 3.
- **Linha:** uma por (estação, dia), na **hora do pico da rajada ERA5** (`gust10fg`). A hora vem do ERA5,
  nunca do INMET, então não há vazamento do alvo.
- **Linha de base honesta:** a própria rajada ERA5 bruta (`era5_gust_max`). No teste ela tem
  **RMSE = 2,658 m/s** (13.583 linhas, 9 estações).
- **Partição por blocos de mês:**
  - teste = jan/abr/jul/out de 2000–2024 (13.583 linhas, 97 blocos ano-mês);
  - treino = demais meses (22.827 linhas);
  - validação = 32 blocos (ano, mês) sorteados nos meses de treino (4.452 linhas);
  - purga de 1 dia nas fronteiras de mês (1.030 linhas).
- **Regra de ouro:** a validação **escolhe** e o teste **pontua**. Um modelo por trimestre (DJF, MAM, JJA, SON).

## 1. Quatro grupos de variáveis (167 colunas)

| grupo | conteúdo | colunas |
|---|---|---|
| 1 | superfície/camada limite: vento 10/100 m, rajada, BLH, T2m, Td2m, p, hora solar, dia do ano, vizinhança, defasagens, lat/lon | 45 |
| 2 | convecção: CAPE, MCPR, cisalhamento 0–6 km, *lapse rates* | 29 |
| 3 | sinótico: MSLP e gradiente, w850/w500, z500, vorticidade, omega700 | 30 |
| 4 | relevo e máscara terra-mar (estáticas por estação, raios de 75/250 km) | 63 |

## 2. Etapa 1 — triagem de modelos e estudo por grupos

1. **Triagem:** os 39 regressores do LazyPredict rodaram no arm `base` (= grupo 1). Para cada trimestre
   ficaram congelados os **5 melhores na validação**: entre os modelos a até 0,05 do melhor R², os de menor
   RMSE_P90 (`top_models.json`).
2. **Estudo por grupos:** com esse top-5 fixo e **5 seeds** (42–46), foram treinados os arms
   `base`, `full`, `add_grp__grupo{2,3,4}`, `drop_grp__grupo{1..4}` e dois controles negativos
   (`ctrl__noise`, `ctrl__perm_static`).
3. **Medida do efeito:** efeito pareado por modelo, IC por bootstrap em blocos (ano, mês) e
   SESOI = 1 % do RMSE do `base`.

| comparação | efeito no RMSE (árvores, 5 seeds) | veredito | LSTM (seed 42) |
|---|---|---|---|
| remover grupo 1 | **+0,216** [0,187; 0,245] | **acrescenta** | +0,143 acrescenta |
| somar grupo 2 | −0,008 | sem informação | −0,158 prejudica |
| somar grupo 3 | −0,006 | sem informação | −0,001 sem informação |
| somar grupo 4 | +0,021 | inconclusivo (≈ estáticas permutadas, +0,019) | −0,232 prejudica |
| `full` vs `base` | +0,005 | sem informação | −0,290 prejudica |
| controle de ruído | −0,002 | sem informação | +0,003 sem informação |

**Leitura:** só o grupo 1 traz informação. O grupo 4 empata com o controle de estáticas permutadas,
ou seja, ele identifica a estação e não representa um efeito do relevo.

RMSE médio no teste (top-5 × 4 trimestres × 5 seeds):

| arm | RMSE no teste |
|---|---|
| ERA5 bruto | 2,658 |
| `base` | 2,299 |
| `full` | 2,300 |
| sem o grupo 1 | 2,491 |

### 2b. Cada grupo sozinho (arquivos `08_*`)

Mesmo caminho do estudo: o **top-5 congelado** de cada trimestre, ajustado com a própria `worker.fit_arm`
nas 5 seeds, agora com **um grupo de cada vez**.

- O grupo 1 sozinho **é** o arm `base` (mesmas 45 colunas, mesma ordem). Suas métricas reproduzem as do
  estudo (`08_conferencia_grupo1_vs_base.parquet`).
- Ao lado vai o LightGBM da seção 7 do notebook (“cada grupo isolado”), que foi como os grupos 2–4 sozinhos
  foram olhados na época.
- Referências: só ruído (o piso) e a rajada ERA5 bruta.

RMSE no teste:

- *top-5*: média das 5 seeds × 5 modelos × 4 trimestres;
- *melhor na validação*: em cada trimestre, o modelo do top-5 com maior R² de validação;
- *IC 95 %*: bootstrap em blocos (ano, mês), agregado dos 4 trimestres, seed 42, modelo da 1ª posição.

| entrada | top-5 (média) | melhor na validação | LightGBM (seção 7) | IC 95 % (posição 1) |
|---|---|---|---|---|
| **grupo 1 sozinho** (= `base`) | **2,300** | **2,250** | **2,282** | [2,12; 2,45] |
| grupo 3 sozinho | 2,896 | 2,859 | 2,887 | [2,74; 3,01] |
| grupo 2 sozinho | 3,184 | 3,139 | 3,143 | [3,03; 3,29] |
| grupo 4 sozinho | 3,261 | 3,259 | 3,259 | [3,12; 3,40] |
| só ruído (piso) | 3,886 | 3,624 | 3,677 | [3,73; 4,05] |
| ERA5 bruto | 2,658 | — | — | [2,51; 2,80] |

Melhor modelo na validação, por grupo e trimestre (`08_grupos_isolados_resumo`, coluna `melhor_val_r2`):

| grupo | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| 1 | CatBoost | ExtraTrees | CatBoost | CatBoost |
| 2 | RandomForest | ExtraTrees | CatBoost | ExtraTrees |
| 3 | CatBoost | RandomForest | CatBoost | CatBoost |
| 4 | RandomForest | ExtraTrees | LinearRegression | RandomForest |

**Leitura:** só o grupo 1, sozinho, supera o ERA5, com ~13 % a menos de RMSE. Os grupos 2, 3 e 4 sozinhos
ganham do piso de ruído (têm alguma informação), mas ficam **piores que o ERA5 bruto**. Somados ao grupo 1,
não acrescentam nada (seção 2). Por isso a seleção de variáveis parte do grupo 1.

**Conferência de reprodução** (`08_conferencia_grupo1_vs_base`), grupo 1 sozinho contra o `base` gravado
no Modal, por modelo:

- **ΔRMSE = 0 exato** para CatBoost, LightGBM, ExtraTrees, LinearRegression, BayesianRidge e SGD;
- diferença de até **0,015 m/s** (média 0,002) para RandomForest, HistGB, Bagging e MLP.

A diferença vem do ambiente: a imagem do Modal instala versões não fixadas em Python 3.11, e aqui foi
Python 3.13 com scikit-learn 1.8. Para os números **originais** do `base`, use `05_*`/`06_*`. Os grupos
2–4 sozinhos não existiam no estudo; foram ajustados aqui pelo mesmo caminho.

## 3. Etapa 2 — da escolha do grupo à escolha da variável (permutação)

Depois de “quais grupos”, a pergunta passou a ser “quais **variáveis** manter”.

**Instrumento:** importância por permutação **por variável**.

- Um LightGBM por trimestre (seed 42, parâmetros padrão) é treinado com as 167 colunas mais uma coluna
  de ruído (`ctrl_noise`).
- Cada **variável** é embaralhada como um bloco, 5 repetições, e mede-se quanto o RMSE piora.
- Variável = coluna sem os sufixos de estatística e de defasagem. Por exemplo, as 10 colunas de `w10`
  são embaralhadas juntas. São 51 variáveis.

**Regra:**

- **manter**: piora média acima do **piso do ruído** (a maior piora do `ctrl_noise` entre os 4 trimestres)
  **e** positiva em pelo menos 3 de 4 trimestres;
- **inconclusiva**: acima do piso, mas positiva em 2 ou menos;
- **descartar**: abaixo do piso.

A seleção passou por duas fases:

1. **No teste (seção 8 do notebook):** piso de 0,00125 m/s e **29 variáveis** mantidas. Usar o teste
   para escolher favorece a própria escolha.
2. **Refeita na validação (seção 9), a oficial:** piso de **0,00399 m/s** e **12 variáveis** mantidas.
   Elas foram congeladas em `config/selected_features_val12.json` (58 colunas).
   **A reprodução aqui deu o mesmo piso e as mesmas 12 variáveis.**

| variável | grupo | piora média do RMSE (val) | trimestres > 0 |
|---|---|---|---|
| gust10fg | 1 | 0,665 | 4 |
| anor | 4 | 0,244 | 4 |
| w10 | 1 | 0,185 | 4 |
| gust | 1 | 0,119 | 4 |
| cape | 2 | 0,040 | 3 |
| w100 | 1 | 0,023 | 3 |
| v10 | 1 | 0,013 | 3 |
| mcpr | 2 | 0,013 | 3 |
| hora_solar_sin | 1 | 0,010 | 4 |
| v100 | 1 | 0,006 | 3 |
| grad_mslp | 3 | 0,005 | 4 |
| grad_mslp_hpa_100km | 3 | 0,005 | 4 |

Inconclusivas: `mbld`, `hora_solar_cos`, `w850`. As outras 36 foram descartadas.

**Complemento (não fez parte da decisão):** a mesma permutação foi aplicada ao **top-5 do estudo** em cada
trimestre, com as pipelines do LazyPredict, RobustScaler e saída truncada em [0, 80].

- A coluna `n_modelos_que_mantem_val` de `03_decisao_selecao` mostra quantos modelos concordam com cada
  variável.
- Cuidado: um modelo que só está no top-5 de 1 ou 2 trimestres não consegue cumprir a regra de “≥ 3 de 4”.
- Na LSTM, a permutação por coluna está em `09_lstm_importancia_permutacao`. A ordem por grupo é
  1 > 4 > 2 > 3, igual à do LightGBM, mas por coluna os dois modelos quase não concordam.

## 4. Etapa 3 — o conjunto reduzido dentro do estudo

As 12 variáveis viraram o arm `sel__val12`, treinado com o mesmo top-5 e as 5 seeds:

| arm | RMSE no teste |
|---|---|
| `sel__val12` | 2,301 |
| `full` | 2,300 |
| `base` | 2,299 |

O corte custa ~0,002 m/s em relação ao `full` e não supera o grupo 1 de forma relevante. **O ganho é a
parcimônia: 58 colunas contra 167**, com a mesma qualidade e ~13 % melhor que o ERA5 (2,658).

**Na LSTM (seed 42, `units/full_lstm`) o quadro é outro:**

| arm | RMSE no teste (LSTM) |
|---|---|
| `base` | 2,281 |
| `sel__val12` | 2,400 |
| `sel__val12_sem_anor` | 2,457 |
| `full` | 2,575 |

- A seleção melhora muito em relação ao `full`, mas fica **0,12 m/s pior que o grupo 1 sozinho**.
- Tirar o `anor` piora ainda mais.

A seleção foi feita com LightGBM e vale para as árvores. Na LSTM, o melhor continua sendo o `base`.
Ambos estão em `05_*`/`06_*`, separados pela coluna `origem`.

## 5. O que foi conferido contra o estado atual

| item | conferência | resultado |
|---|---|---|
| Variáveis | impressão digital das 58 colunas de `config/selected_features_val12.json` vs. o `features_fp` gravado nas unidades `sel__val12` (Modal, 5 seeds; LSTM) | idênticas (`c4163d73d77d` / `35cd4c864465`) |
| Seleção | permutação refeita na validação | mesmo piso (0,00399) e mesmas 12 variáveis |
| Modelos (árvores) | top-5 congelado do estudo de referência no Modal (`cluster3_groups_modal/summary/top_models.json`, 10 modelos distintos) | é o conjunto vigente segundo `documentation/09` |
| Predições/métricas dos arms | RMSE refeito a partir dos resíduos vs. `metrics__*` | diferença < 1e-7 |
| Grupo 1 sozinho vs. `base` | reajuste local | exato nos modelos determinísticos; ≤ 0,015 m/s nos demais (ambiente) |

**Fora do escopo destes arquivos:**

- As pipelines de produção (`cluster_lazy`, `cluster_mlp`, `cluster_lstm`) ainda **não** usam a seleção
  `val12`. O `best_model_selector` segue escolhendo por R².
- A réplica local das árvores (`cluster3_groups/units/full`, com top-5 de uma triagem local diferente) não
  entra aqui. A referência é o estudo do Modal.
- O modelo usado no teste Cluster 3 → 12 (seção 7) não foi conferido, porque está no Modal.

## 6. Ressalvas

- A permutação **divide o crédito entre variáveis colineares** (`w10`, `gust`, `gust10fg`) e mede uso, não ganho
  fora da amostra.
- Cada trimestre testa num único mês, com 24–25 blocos, e o IC é anticonservador. Por isso só o agregado
  (97 blocos) tem IC confiável.
- A permutação usa uma seed (o LightGBM padrão é determinístico). O estudo por grupos usa 5.

## 7. Cluster 3 → Cluster 12 (`07_*`) — pendente

O modelo treinado no Cluster 3 e testado no Cluster 12 está no volume do Modal, ao qual a sessão que gerou
estes arquivos não tinha acesso. Para trazer:

```bash
MODAL_PROFILE=<perfil> uv run python scripts/baixar_transferencia_c3_c12.py --volume <volume> --caminho <caminho>
# ou, com o arquivo já baixado:
uv run python scripts/baixar_transferencia_c3_c12.py --local <arquivo.parquet|csv>
```

O resultado é gravado em `07_cluster3_para_cluster12_teste.parquet` (schema de
`build_station_predictions_frame`) e em `07_cluster3_para_cluster12_metricas.parquet` (total, por trimestre
e por estação).

---

## Dicionário dos arquivos

| arquivo | linha = | colunas principais |
|---|---|---|
| `01_linhas_val_teste.parquet` | linha (estação-dia) de validação e teste | `row_id, estacao, time, season, year, month, split, latitude, longitude, y_true, era5_gust_max` |
| `01_permutacao_predicoes_<modelo>.parquet` | (trimestre, split, variável embaralhada, repetição, linha) | `season, split, variavel, grupo, repeticao, row_id, y_true, y_pred_sem_permutar, y_pred`. `LGBMRegressor_selecao` é o instrumento da seleção; os outros são o top-5 do estudo |
| `02_permutacao_importancia.parquet` | modelo × trimestre × split × variável × repetição | `rmse_sem_permutar, rmse_permutado, d_rmse, d_rmse_p90, n_colunas, na_selecao_val12` (inclui `ctrl_noise`) |
| `03_decisao_selecao.parquet` | modelo × split × variável | `d_rmse_medio, d_rmse_<trimestre>, trimestres_positivos, piso_ruido, decisao, na_selecao_val12, n_modelos_que_mantem_val`. **Oficial: `modelo == "LGBMRegressor (seleção)"` e `split == "val"`** |
| `04_grupos_efeitos.parquet` | comparação × família/trimestre | `effects`, `effects_by_season`, `ranking_groups` do estudo (`origem`: árvores 5 seeds / LSTM) |
| `05_comparacao_arms_teste.parquet` | (origem, arm, trimestre, modelo, seed, linha de teste) | `origem` = árvores top-5 Modal (5 seeds) ou LSTM (seed 42); arms `base, full, sel__val12, sel__val12_sem_anor, drop_grp__grupo1, add_grp__*, ctrl__*`, com `y_true, y_pred, era5_gust_max, estacao, time, lat/lon`. `y_pred` vem do resíduo gravado mais `y_true` (conferido contra `preds__*`, diferença < 1e-6) |
| `06_metricas_arms.parquet` | arm × trimestre × modelo × seed × split | `R2, RMSE, Bias, Bias_P90, RMSE_P90`, como gravadas pelo estudo |
| `07_*` | — | pendente (seção 7) |
| `08_grupos_isolados_metricas.parquet` | grupo × trimestre × modelo × seed × split | `origem` (`top5_estudo` / `lgbm_notebook`), `rank_top5_estudo`, métricas |
| `08_grupos_isolados_predicoes_teste.parquet` | (grupo, trimestre, modelo, seed, linha de teste) | `y_pred, y_true, era5_gust_max, estacao, time` |
| `08_grupos_isolados_resumo.parquet` | grupo × trimestre × modelo (média das seeds) e agregado | `RMSE_test` (± dp das seeds), `RMSE_test_ic95_inf/sup` (bootstrap em blocos), `R2_val`, ERA5 |
| `08_conferencia_grupo1_vs_base.parquet` | trimestre × modelo × seed × split | RMSE do grupo 1 sozinho vs. arm `base` do estudo |
| `09_lstm_importancia_permutacao.parquet` | trimestre × coluna/grupo × repetição | permutação da LSTM no `full` |

```python
import pandas as pd
dec = pd.read_parquet("resultados_preliminares/03_decisao_selecao.parquet")
oficial = dec.query("modelo == 'LGBMRegressor (seleção)' and split == 'val'")

lin = pd.read_parquet("resultados_preliminares/01_linhas_val_teste.parquet")
p = pd.read_parquet("resultados_preliminares/01_permutacao_predicoes_LGBMRegressor_selecao.parquet")
gust = p.query("variavel == 'gust10fg' and split == 'test'").merge(lin, on=["row_id", "y_true"])
```
