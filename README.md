# IRC Vendaval — Correção de Viés de Rajadas de Vento Extremo

Bias correction de `daily_wind_gust_max` do ERA5 em relação a observações INMET,
estratificado em **14 clusters espaciais**. Foco em eventos extremos de vento (vendavais).

---

## Índice

1. [Contexto e Objetivo](#1-contexto-e-objetivo)
2. [Dados](#2-dados)
3. [Clusters Espaciais](#3-clusters-espaciais)
4. [Evolução da Metodologia](#4-evolução-da-metodologia)
5. [Pipeline Atual — MLP PyTorch](#5-pipeline-atual--mlp-pytorch)
6. [Estudo de Features por Grupo](#6-estudo-de-features-por-grupo)
7. [Como Rodar](#7-como-rodar)
8. [Estrutura do Projeto](#8-estrutura-do-projeto)
9. [Setup Local](#9-setup-local)
10. [Referências](#10-referências)

> **Fluxo principal:**
> ```
> cluster_lazy / cluster_mlp / cluster_lstm  →  seleção do melhor modelo por cluster × trimestre  →  corrected_grid
> ```
> O LazyPredict avalia dezenas de modelos por cluster com as features atuais, dados de clusters
> vizinhos e avaliação por janelas de deploy (mensal/quinzenal). Para a rede neural principal, o
> caminho ativo é `cluster_lstm` (LSTM(96) → Dropout(0.3) → Dense(1), Huber, saída em m/s).
> Todas as pipelines usam o **mesmo split por blocos de mês** (ver "Split temporal").
>
> O código de geração de dados sintéticos (GAN/difusão) foi extraído para o repositório
> `irc_vendaval_gan` e não é mais usado aqui.

---

## 1. Contexto e Objetivo

O ERA5 (reanálise global ECMWF) subestima sistematicamente rajadas máximas
observadas pelas estações INMET, especialmente durante eventos extremos (vendavais,
frentes frias, linhas de instabilidade). O objetivo é treinar modelos de **correção
de viés** que, dado apenas o ERA5, aproximem a rajada que seria registrada na estação.

**Formulação do target:** o modelo aprende a razão `INMET / ERA5` (fator de correção),
não o valor absoluto. Na inferência: `ŷ_corrigido = razão_predita × ERA5_wind_mag_max`.

**Princípio central — sem vazamento de dados:** todas as features de entrada
são derivadas exclusivamente do ERA5. Nenhuma observação de estação entra como
feature (apenas como target no treino) — nem lags da rajada observada, nem
climatologia/mediana por estação — porque na correção da grade ERA5 não existe
estação medindo. A inferência recusa artefatos treinados com essas features.

---

## 2. Dados

| Arquivo | Descrição |
|---------|-----------|
| `dataset/raw/Training_Dataset_INMET_ERA5_Paired.csv` | Fonte upstream do par INMET/ERA5 usada para gerar a base operacional do projeto |
| `dataset/raw/INMET_Stratified.nc` | Rajada observada do INMET em formato estratificado para o treino da pipeline |
| `dataset/raw/ERA5_Stratified.nc` | Reanálise ERA5 co-localizada em formato diário para as features de entrada |
| `dataset/shp/shp_vento.shp` | Polígonos dos 14 clusters espaciais (spatial join) |

O dataset é compartilhado via Modal Volume `irc-vendaval-dataset`. O upload ocorre
automaticamente na primeira execução de qualquer pipeline Modal.

### Split temporal

Partição por **blocos de mês** (`src/pipeline/data/splits.py::MonthBlockSplit`), a mesma em
`cluster_lazy`, `cluster_mlp`, `cluster_lstm` (v2) e no estudo de features:

| Split | Definição |
|-------|-----------|
| **Teste** | janeiro, abril, julho e outubro de **todos** os anos (um mês por trimestre climático) |
| **Treino** | os oito meses restantes |
| **Validação** | blocos (ano, mês) sorteados **dentro dos meses de treino** (15 %, estratificados por mês, seed 42), sempre para todas as estações ao mesmo tempo |
| **Purga** | gap, em dias, nas fronteiras entre meses de splits diferentes, derivado da defasagem máxima das features |

> No `cluster_lazy` a validação é avaliada por **janelas de deploy** (mensal ou quinzenal)
> com `--eval-window`, refletindo o cenário real de inferência mensal/quinzenal.

### Features utilizadas (27 no total)

Legenda:
- **ERA5 bruto**: variável vinda diretamente do NetCDF/ERA5.
- **Derivada na pipeline**: calculada a partir de ERA5, tempo ou histórico do treino.
- **Meta da estação**: valor fixo por estação/cluster usado como contexto.

| Grupo | Tipo | Features |
|-------|------|---------|
| Vento | ERA5 bruto | `10m_u/v_component_of_wind`, `wind_mag`, `wind_mag_max`, `wind_mag_min`, `wind_mag_std` |
| Direção do vento | Derivada na pipeline | `wind_dir_sin`, `wind_dir_cos` (arctan2 do vetor U/V codificado ciclicamente) |
| Convecção | Derivada na pipeline | `gust_factor` (pico/média intradiária — proxy convectivo) |
| Persistência | Derivada na pipeline | `lag1_wind_mag_max`, `lag2_wind_mag_max`, `lag3_wind_mag_max`, `lag7_wind_mag_max` |
| Tendência | Derivada na pipeline | `rolling7d_wind_mag_max` (média móvel 7 dias — evolução sinótica) |
| Termodinâmica | ERA5 bruto | `2m_temperature`, `2m_dewpoint_temperature`, `relative_humidity`, `t2m_range` |
| Pressão | ERA5 bruto | `surface_pressure`, `pressure_tendency` |
| Precipitação | ERA5 bruto | `total_precipitation` |
| Sazonalidade | Derivada na pipeline | `day_sin`, `day_cos` |
| Climatologia ERA5 | Derivada na pipeline | `era5_clim_wind` (média histórica do `wind_mag_max` por dia-do-ano) |
| Localização | Meta da estação | `latitude`, `longitude` (coordenadas da estação) |

---

## 3. Clusters Espaciais

14 regiões climáticas definidas por shapefile, com regimes distintos de vento:

| Cluster | Estações | Característica |
|---------|----------|----------------|
| C1 | 9 | Partição geográfica do shapefile |
| C2 | 16 | Partição geográfica do shapefile |
| C3 | 21 | Partição geográfica do shapefile |
| C4 | 31 | Partição geográfica do shapefile |
| C5 | 22 | Partição geográfica do shapefile |
| C6 | 16 | Partição geográfica do shapefile |
| C7 | 9 | Partição geográfica do shapefile |
| C8 | 15 | Partição geográfica do shapefile |
| C9 | 44 | Partição geográfica do shapefile |
| C10 | 32 | Partição geográfica do shapefile |
| C11 | 16 | Partição geográfica do shapefile |
| C12 | 19 | Partição geográfica do shapefile |
| C13 | 10 | Partição geográfica do shapefile |
| C14 | 11 | Partição geográfica do shapefile |

Cada pipeline treina um modelo independente por cluster, preservando os
regimes meteorológicos distintos.

---

## 4. Evolução da Metodologia

### 4.1 EMOS Probabilístico (baseline original)
Ensemble Model Output Statistics com redes neurais (Keras/TensorFlow):
- **NNModel** — rede densa 1-D sobre features meteorológicas
- **NNConvModel** — CNN sobre grades espaciais de velocidade do vento
- **CNNModel** — autoencoder 3-D CNN (PyTorch)
- **CGRU / CLSTM** — células convolucionais recorrentes

Loss: Threshold-weighted CRPS (twCRPS) para eventos extremos.

### 4.2 LSTM por Cluster (`cluster_lstm`)
Pipeline sequencial usando dados NetCDF brutos: uma LSTM de saída única por
cluster (`LSTM(96) → Dropout(0.3) → Dense(1)`, Huber + Adam, alvo em m/s),
com dados diários (janela de 7 dias até o dia-alvo) ou horários (24 h do
dia-alvo), interpolação bilinear grade→estação e split por blocos de mês
(teste = Jan/Abr/Jul/Out de todos os anos). Ver
`documentation/04_cluster_lstm.md`.

### 4.3 Screening LazyPredict (`cluster_lazy`)
LazyPredict avalia dezenas de modelos sklearn por cluster, com as seguintes capacidades:

**Feature engineering completo (27 features):** o pipeline usa o conjunto expandido de features descrito na seção 2, incluindo direção do vento, lags longos, rolling stats.

**Compartilhamento entre clusters vizinhos (`--n-neighbor-clusters N`):** dados reais dos N clusters geograficamente mais próximos entram no treino. A vizinhança é calculada por distância euclidiana entre centroides das estações. A validação permanece no cluster-alvo para medir a performance sem contaminação.

**Estratificação sazonal (`--stratify-seasons`):** treina um modelo separado por cluster × trimestre climático (DJF, MAM, JJA, SON) além do modelo geral.

**Avaliação orientada ao deploy (`--eval-window monthly|biweekly`):** em vez de um único R² anual, computa R² por janela mensal ou quinzenal para os top-5 modelos. Replica o cenário real de inferência e revela a variância mensal da performance. O resultado aparece como `R2_deploy_mean`, `R2_deploy_std` e `R2_deploy_min` no CSV de resultados.

**Artefatos gerados por cluster (`artifacts/lazy_clusters/expN/`):**

| Artefato | Descrição |
|----------|-----------|
| `predictions/predictions_c{N}.csv` | Predições de todos os modelos — colunas `split`, `y_true`, `season`, `cluster_id` e uma coluna por modelo |
| `rolling_r2/rolling_r2_c{N}.csv` | R² por janela de deploy (top-5 modelos × janelas mensais ou quinzenais) |
| `plots/rolling_r2_c{N}.png` | Série temporal de R² por janela — mostra estabilidade da performance ao longo do ano |
| `plots/scatter_all_c{N}.pdf` | PDF com ~40 páginas — uma por modelo, scatter treino + validação |
| `plots/top5_c{N}.png` | Top-5 modelos por R² agregado |
| `plots/scatter_c{N}.png` | Scatter do melhor modelo na validação |
| `plots/train_distribution_c{N}.png` | Distribuição do alvo no treino |

### 4.4 MLP sklearn com Extreme Weighting (`cluster_mlp`)
Pipeline dedicada ao MLPRegressor com foco explícito nos extremos:
- `sample_weight ∝ daily_wind_gust_max ^ extreme_power` via oversampling
- Métricas de cauda: Bias@P90, RMSE@P90
- Comparação com baseline ERA5 bruto
- Arquitetura configurável via `--hidden-layers`

**Limitação identificada:** MSE como loss ignora a cauda da distribuição.
Treino R²≈0.85 vs Validação R²≈0.36 indicam overfitting, parcialmente
causado pela falta de eventos extremos no período de treino.

### 4.5 Estado atual da pilha de treino
Hoje o código expõe três famílias principais de treino por cluster:
- `cluster_lstm`: LSTM de saída única em Keras/TensorFlow (`model.name: cluster_lstm`, config `experiment_cluster_lstm_modal.yaml`).
- `cluster_mlp`: `MLPRegressor` do scikit-learn com reamostragem/weighting dos extremos e alvo em razão `INMET/ERA5`.
- `cluster_lazy`: benchmark automático com LazyPredict, que seleciona o melhor regressor por cluster e salva o campeão para inferência espacial.

---

## 5. Pipeline Atual

### `cluster_lstm`

Arquitetura principal em produção:
- `Input(T, F) → LSTM(96) → Dropout(0.3) → Dense(1)`, Huber + Adam, rajada em m/s.
- Fonte diária ou horária (`data.resolution`), interpolação `nearest`/`bilinear`.
- Split por blocos de mês com purga de janela.
- Artefatos da antiga LSTM dual-head/TR são recusados pela inferência — retreinar.
- A feature engineering é obrigatória na pipeline; o configurável são os grupos de features (`feature_groups`: `original`, `era5_basin`, `new_features`). Nada é imputado — ver `documentation/02_dados_e_preprocessamento.md`.

### `cluster_mlp`

Pipeline de baseline supervisionado com `MLPRegressor`:
- Pré-processamento com `RobustScaler` (sem imputação).
- Alvo treinado como razão `INMET/ERA5`.
- Reamostragem dos eventos extremos por `extreme_power`.
- Inferência espacial salva o artefato com `target_kind: ratio`.

### `cluster_lazy`

Pipeline de benchmark e seleção do melhor modelo por cluster:
- `LazyRegressor` com filtragem dos estimadores muito lentos.
- Escolha do melhor modelo por métrica no conjunto de validação.
- Salvamento do campeão por cluster para a etapa de inferência espacial.

---

## 6. Estudo de Features por Grupo

`src/feature_study/` responde quais grupos de variáveis ERA5 (1–4 do spec) acrescentam informação à
correção da rajada, e quais variáveis vale manter: arms COM e SEM cada grupo, bootstrap pareado em
blocos (ano, mês), controles negativos (ruído e estáticas permutadas entre estações) e a mesma
partição por blocos de mês. Resumo, resultados e comandos em `documentation/09_estudo_de_features.md`;
notebooks em `notebooks/feature_study_grupos/`.

---

## 7. Como Rodar

Todos os pipelines são invocados através de um único `main.py` com subcomandos:

```bash
python main.py --help
# lista: cluster_lstm, cluster_mlp, cluster_lazy, corrected_grid
```

### LSTM por cluster (`cluster_lstm`)

```bash
python main.py cluster_lstm \
  --config config/experiment_cluster_lstm_modal.yaml
```

### MLP sklearn com Extreme Weighting (`cluster_mlp`)

```bash
# Local
python main.py cluster_mlp \
    --output-dir artifacts/mlp_clusters \
    --hidden-layers 128,64,32 \
    --alpha 0.01 \
    --max-iter 1000

# Na nuvem (Modal) com download automático
modal run src/modal/cluster_mlp.py \
    --hidden-layers 128,64,32 \
    --alpha 0.01 \
    --max-iter 1000

# Só baixar artefatos de run anterior
modal run src/modal/cluster_mlp.py --only-download
```

### Lazy benchmark (`cluster_lazy`)

```bash
python main.py cluster_lazy \
  --output-dir artifacts/lazy_clusters \
  --n-neighbor-clusters 1 \
  --eval-window monthly
```

### LSTM por Cluster (`cluster_lstm`)

```bash
# Cache do merge com interpolação bilinear (uma vez; a config v2 usa bilinear)
modal run src/modal/cluster_lazy.py --build-cache --interp-method bilinear

# Local
python main.py cluster_lstm \
    --config config/experiment_cluster_lstm_modal.yaml

# Na nuvem (Modal)
modal run src/modal/cluster_lstm.py \
    --config experiment_cluster_lstm_modal.yaml
```

### LazyPredict por cluster (`cluster_lazy`) na nuvem

```bash
# LazyPredict com vizinhos e avaliação de deploy
modal run --detach src/modal/cluster_lazy.py \
    --n-neighbor-clusters 1 \
    --eval-window monthly \
    --exp-name full_v1
```

**Variantes úteis:**

```bash
# Sem vizinhos (baseline isolado por cluster)
modal run --detach src/modal/cluster_lazy.py \
    --n-neighbor-clusters 0 \
    --exp-name no_neighbors

# Com 2 vizinhos + avaliação quinzenal
modal run --detach src/modal/cluster_lazy.py \
    --n-neighbor-clusters 2 \
    --eval-window biweekly \
    --exp-name neighbors2_biweekly

# Estratificado por trimestre + vizinhos
modal run --detach src/modal/cluster_lazy.py \
    --n-neighbor-clusters 1 \
    --stratify-seasons \
    --exp-name seasonal_v1

# Testar um cluster localmente antes de subir tudo
python main.py cluster_lazy --exp-name local_test --cluster-id 3 \
    --n-neighbor-clusters 1 --eval-window monthly

# Só baixar artefatos de run anterior
modal run src/modal/cluster_lazy.py --only-download
```

**Artefatos gerados em `artifacts/lazy_clusters/expN/`:**
- `predictions/predictions_c{N}.csv` — predições de todos os modelos por split
- `rolling_r2/rolling_r2_c{N}.csv` — R² por janela de deploy (top-5 modelos)
- `plots/rolling_r2_c{N}.png` — série temporal de R² por mês/quinzena
- `plots/scatter_all_c{N}.pdf` — scatter por modelo: treino + validação
- `lazy_cluster_results.csv` — consolidado com `R2_deploy_mean`, `R2_deploy_std`, `R2_deploy_min`

| Flag | Default | Descrição |
|------|---------|-----------|
| `--n-neighbor-clusters` | `1` | Nº de clusters vizinhos cujos dados entram no treino |
| `--eval-window` | `monthly` | Janela de avaliação de deploy: `monthly` ou `biweekly` |
| `--stratify-seasons` | `False` | Treinar 1 modelo por cluster × trimestre (DJF/MAM/JJA/SON) |
| `--exp-name` | auto (`expN`) | Nome do experimento |
| `--cluster-id` | `None` | Rodar só um cluster (ex: `--cluster-id 3`) |
| `--aggregate-only` | `False` | Só reagregar resultados já calculados |

### Dashboard

```bash
cd irc_vendaval_dashboard
streamlit run app.py
```

---

## 8. Estrutura do Projeto

```
irc_vendaval/
│
├── main.py                         # Dispatcher único (subcomandos abaixo)
│
├── src/
│   ├── modal/                      # Entrypoints Modal (1 por pipeline)
│   │   ├── cluster_lstm.py         # modal run src/modal/cluster_lstm.py
│   │   ├── cluster_mlp.py          # modal run src/modal/cluster_mlp.py
│   │   ├── cluster_lazy.py         # modal run src/modal/cluster_lazy.py (fan-out)
│   │   └── corrected_grid.py       # modal run src/modal/corrected_grid.py
│   ├── pipelines/                  # Lógica de cada pipeline
│   │   ├── common.py               # Constantes, splits e utils compartilhados
│   │   ├── cluster_lstm.py         # Subcomando: cluster_lstm
│   │   ├── cluster_mlp.py          # Subcomando: cluster_mlp
│   │   ├── cluster_lazy.py         # Subcomando: cluster_lazy
│   │   └── corrected_grid.py       # Subcomando: corrected_grid
│   ├── data/
│   │   ├── netcdf_loader.py        # Carrega e agrega INMET + ERA5 para diário
│   │   ├── cluster_assigner.py     # Spatial join estação → cluster
│   │   ├── climatology.py          # Climatologia por dia-do-ano e harmônica (sem leakage)
│   │   ├── interp.py               # Grade→ponto: nearest / bilinear
│   │   └── static_features.py      # Features estáticas por estação
│   ├── feature_study/              # Estudo de features por grupo (core/, data/, selection/, diagnostics/)
│   ├── models/
│   │   └── cluster_lstm_builder.py # LSTM(96) → Dropout → Dense(1), Huber
│   ├── pipeline/                   # Módulos LSTM / EMOS (baseline)
│   │   ├── data/
│   │   │   ├── cluster_preprocessor.py  # ClusterDataBatch + fonte diária
│   │   │   ├── hourly_builder.py        # fonte horária (24 h do dia-alvo)
│   │   │   ├── lstm_sources.py          # validação do YAML + despacho por resolução
│   │   │   ├── splits.py                # split por blocos de mês, purga, anulação de lags
│   │   │   └── windowing.py             # convenção única de janelas
│   │   ├── training/
│   │   │   └── cluster_trainer.py
│   │   └── validation/
│   │       └── cluster_metrics.py
│   └── visualization/
│       └── cluster_plots.py
│
├── config/                         # Toda a configuração
│   ├── experiment_cluster_lstm_*.yaml  # Experimentos da LSTM (schema v2)
│   ├── selected_features_val12.json    # Seleção congelada de variáveis (estudo de features)
│   └── base/                       # Kedro: catalog.yml e parameters.yml
│
├── dataset/
│   ├── raw/                        # NetCDF: INMET + ERA5
│   └── shp/                        # Shapefiles dos 14 clusters
│
├── artifacts/
│   ├── mlp_clusters/               # Saída do cluster_mlp
│   ├── lazy_clusters/              # Saída do cluster_lazy
│   ├── feature_study/              # Saída do estudo de features por grupo
│   └── corrected_grid/             # NetCDFs corrigidos
│
├── irc_vendaval_dashboard/         # Dashboard Streamlit de comparação
└── test/                           # Testes unitários (pytest)
```

---

## 9. Setup Local

### Pré-requisitos

```bash
# Instalar uv (gerenciador de pacotes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependências
uv sync
```

### Modal (execução na nuvem)

```bash
pip install modal
modal setup   # autenticar com token
```

### Variáveis de ambiente

Copie `.env.example` para `.env` e preencha as credenciais necessárias.

### Rodar os testes

```bash
uv run pytest test/ -q
```

---

## 10. Referências

- *TRWindBC* — Ouarda & Houndekindo (Energy 2025): LSTM dual-branch com features estáticas por estação para correção de ERA5 horário em estações canadenses
- *GFS-NWP-Wind-Correction* — Kuang (2025): 3D-UNet com loss física para correção de campos GFS
- *Improving Probabilistic Forecasting in the Netherlands* — motivação metodológica EMOS
- *Adapting a deep convolutional RNN model with imbalanced regression loss for improved spatio-temporal forecasting of extreme wind speed events in the short to medium range*
- DeepLearning.AI GANs Specialization (Coursera) — arquitetura cWGAN-GP
- Koenker & Bassett (1978) — Quantile Regression / Pinball Loss
