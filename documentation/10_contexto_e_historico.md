# Contexto, estrutura e histórico

Texto movido do README, que agora só traz o passo a passo de execução. As partes sobre features e
alvo descrevem o desenho anterior (alvo em razão `INMET/ERA5`, 27 features diárias); o desenho atual
está em [02](02_dados_e_preprocessamento.md), [04](04_cluster_lstm.md) e
[09](09_estudo_de_features.md). O split por blocos de mês, único em todo o repositório, está em
[02](02_dados_e_preprocessamento.md). O GAN/difusão foi extraído para o repositório `irc_vendaval_gan`.

## Contexto e objetivo

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

## Dados e features

| Arquivo | Descrição |
|---------|-----------|
| `dataset/raw/Training_Dataset_INMET_ERA5_Paired.csv` | Fonte upstream do par INMET/ERA5 usada para gerar a base operacional do projeto |
| `dataset/raw/INMET_Stratified.nc` | Rajada observada do INMET em formato estratificado para o treino da pipeline |
| `dataset/raw/ERA5_Stratified.nc` | Reanálise ERA5 co-localizada em formato diário para as features de entrada |
| `dataset/shp/shp_vento.shp` | Polígonos dos 14 clusters espaciais (spatial join) |

O dataset é compartilhado via Modal Volume `irc-vendaval-dataset`. O upload ocorre
automaticamente na primeira execução de qualquer pipeline Modal.

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

## Clusters espaciais

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

## Evolução da metodologia

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

## Estrutura do projeto

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

## Referências

- *TRWindBC* — Ouarda & Houndekindo (Energy 2025): LSTM dual-branch com features estáticas por estação para correção de ERA5 horário em estações canadenses
- *GFS-NWP-Wind-Correction* — Kuang (2025): 3D-UNet com loss física para correção de campos GFS
- *Improving Probabilistic Forecasting in the Netherlands* — motivação metodológica EMOS
- *Adapting a deep convolutional RNN model with imbalanced regression loss for improved spatio-temporal forecasting of extreme wind speed events in the short to medium range*
- DeepLearning.AI GANs Specialization (Coursera) — arquitetura cWGAN-GP
- Koenker & Bassett (1978) — Quantile Regression / Pinball Loss
