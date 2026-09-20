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
6. [GAN para Augmentation de Extremos](#6-gan-para-augmentation-de-extremos)
7. [Como Rodar](#7-como-rodar)
8. [Estrutura do Projeto](#8-estrutura-do-projeto)
9. [Setup Local](#9-setup-local)
10. [Referências](#10-referências)

> **Fluxo principal recomendado:**
> ```
> cluster_gan  →  cluster_lazy (com --synthetic-csv --n-neighbor-clusters 1 --eval-window monthly)
> ```
> A GAN gera dados sintéticos de extremos; o LazyPredict avalia dezenas de modelos por cluster
> usando as features atuais, dados de clusters vizinhos e avaliação por janelas de deploy
> (mensal/quinzenal). Para a rede neural principal, o caminho ativo é `cluster_lstm`
> (LSTM(96) → Dropout(0.3) → Dense(1), Huber, saída em m/s).

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

| Split | Período | Uso |
|-------|---------|-----|
| **Treino** | 2008–2018 | Fit dos modelos atuais (`cluster_lstm`, `cluster_mlp`, `cluster_lazy`) |
| **Validação** | 2019 | Early stopping / seleção de hiperparâmetros, dependendo da pipeline |
| **Teste** | 2020–2025 | Avaliação final das pipelines atuais |

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

**Compartilhamento entre clusters vizinhos (`--n-neighbor-clusters N`):** dados reais e sintéticos dos N clusters geograficamente mais próximos entram no treino. A vizinhança é calculada por distância euclidiana entre centroides das estações. A validação permanece no cluster-alvo para medir a performance sem contaminação.

**Estratificação sazonal (`--stratify-seasons`):** treina um modelo separado por cluster × trimestre climático (DJF, MAM, JJA, SON) além do modelo geral.

**Avaliação orientada ao deploy (`--eval-window monthly|biweekly`):** em vez de um único R² anual, computa R² por janela mensal ou quinzenal para os top-5 modelos. Replica o cenário real de inferência e revela a variância mensal da performance. O resultado aparece como `R2_deploy_mean`, `R2_deploy_std` e `R2_deploy_min` no CSV de resultados.

**Artefatos gerados por cluster (`artifacts/lazy_clusters/expN/`):**

| Artefato | Descrição |
|----------|-----------|
| `predictions/predictions_c{N}.csv` | Predições de todos os modelos — colunas `split`, `y_true`, `source` (`real`/`synthetic`) e uma coluna por modelo |
| `rolling_r2/rolling_r2_c{N}.csv` | R² por janela de deploy (top-5 modelos × janelas mensais ou quinzenais) |
| `plots/rolling_r2_c{N}.png` | Série temporal de R² por janela — mostra estabilidade da performance ao longo do ano |
| `plots/scatter_all_c{N}.pdf` | PDF com ~40 páginas — uma por modelo, scatter treino (azul=real, laranja▲=sintético) + validação |
| `plots/synth_comparison_c{N}.png` | ΔR² por modelo (com − sem sintéticos): verde = melhora, vermelho = piora |
| `plots/top5_c{N}.png` | Top-5 modelos por R² agregado |
| `plots/scatter_c{N}.png` | Scatter do melhor modelo na validação |
| `plots/train_distribution_c{N}.png` | Distribuição treino: real vs sintético |

### 4.4 MLP sklearn com Extreme Weighting (`cluster_mlp`)
Pipeline dedicada ao MLPRegressor com foco explícito nos extremos:
- `sample_weight ∝ daily_wind_gust_max ^ extreme_power` via oversampling
- Métricas de cauda: Bias@P90, RMSE@P90
- Comparação com baseline ERA5 bruto
- Arquitetura configurável via `--hidden-layers`

**Limitação identificada:** MSE como loss ignora a cauda da distribuição.
Treino R²≈0.85 vs Validação R²≈0.36 indicam overfitting, parcialmente
causado pela falta de eventos extremos no período de treino.

### 4.5 GAN de Augmentation por Cluster (`cluster_gan`) — cWGAN-GP simplificado
GAN condicional treinada apenas no **alvo** (`daily_wind_gust_max`), condicionada
ao cluster via one-hot. Gera dois arquivos por experimento:

| Arquivo | Conteúdo |
|---------|---------|
| `synthetic_extremes.csv` | Alvo sintético acima do P90 por cluster |
| `synthetic_augment.csv` | Features ERA5 + alvo, prontos para treino — features atribuídas via vizinho mais próximo (nearest-neighbor no espaço do alvo por cluster) |

**Busca de hiperparâmetros (Optuna):**
```bash
modal run src/modal/gan_hparam.py --n-trials 30 --extreme  # avalia só na cauda P90+
```
Melhor resultado encontrado: Wasserstein = 0.0201 com `z_dim=16`, `hidden_dim=128`,
`c_lambda=5.0`, `crit_repeats=5`, `lr=5e-5`.

### 4.6 GAN de Augmentation completo (`cluster_gan`)
Essa pipeline segue ativa como geradora de dados sintéticos usados pelos modelos
subsequentes. Ela produz o vetor completo `[features ERA5 + target]`:
- Treina na distribuição completa (features ERA5 + target INMET)
- Gera apenas amostras acima do P90 via rejection sampling
- `n_per_cluster_ratio`: número de sintéticos proporcional ao tamanho real
  do cluster — evita dominância de dados sintéticos em clusters pequenos
- Sanity check automático: compara P90/P95 real × sintético por cluster

### 4.6 Estado atual da pilha de treino
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

## 6. GAN para Augmentation de Extremos

A cWGAN-GP (Conditional Wasserstein GAN with Gradient Penalty) gera amostras
sintéticas de `daily_wind_gust_max` condicionadas ao cluster. Apenas amostras
com rajada ≥ P90 são retidas (rejection sampling), enriquecendo a cauda da
distribuição de treino. As features ERA5 correspondentes são atribuídas por
**nearest-neighbor**: para cada alvo sintético `y_s`, busca-se o sample real
do mesmo cluster com `y_real` mais próximo e usa-se suas features ERA5.

### Arquitetura (src/gan/)

| Módulo | Conteúdo |
|--------|---------|
| `models.py` | Generator e Critic — 2 camadas ocultas, sem BatchNorm |
| `losses.py` | `crit_loss` (com GP inline) e `gen_loss` |
| `conditioning.py` | One-hot labels + combine_vectors |
| `train.py` | Loop WGAN-GP: early stopping por distância Wasserstein, sem gradient clipping (GP garante Lipschitz) |
| `sampling.py` | `sample()`, `sample_extremes()` com restrição física `y ≥ 0` |
| `evaluate.py` | Fréchet 1D, Wasserstein, KS, comparação de quantis |
| `hparam_search.py` | Optuna HPO com opção `eval_extremes` (avalia só na cauda P90+) |

### Decisões de design

- **Sem gradient clipping no critic** — conflita com o gradient penalty (ambos
  enforçam Lipschitz). Apenas GP é usado.
- **Early stopping por Wasserstein** — salva o checkpoint com menor distância
  Wasserstein na validação, não o da última época.
- **Restrição física** — velocidade de vento ≥ 0 forçada após `inverse_transform`.

### Sanity check automático

Após gerar os sintéticos, o pipeline compara por cluster:
- `frechet_1d`, `wasserstein`, `ks_stat` — qualidade da distribuição sintética
- `delta_P90`, `delta_P95`, `delta_P99` — alinhamento da cauda

---

## 7. Como Rodar

Todos os pipelines são invocados através de um único `main.py` com subcomandos:

```bash
python main.py --help
# lista: cluster_lstm, cluster_mlp, cluster_lazy,
#        cluster_gan, corrected_grid
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

### GAN de augmentation (`cluster_gan`)

```bash
python main.py cluster_gan \
  --output-dir artifacts/gan_clusters \
  --epochs 300 \
  --extreme-percentile 90.0
```

| Flag | Default | Descrição |
|------|---------|-----------|
| `--epochs` | `300` | Épocas de treino da GAN |
| `--extreme-percentile` | `90.0` | Percentil para definir os extremos |
| `--hidden-dim` | `128` | Largura das camadas do Generator e Critic |
| `--crit-repeats` | `5` | Passos do Critic por passo do Generator (WGAN-GP) |
| `--c-lambda` | `10.0` | Peso do gradient penalty |
| `--lr` | `2e-4` | Learning rate Adam (ou use `--lr-gen` / `--lr-crit` separados) |
| `--batch-size` | `128` | Tamanho do batch |
| `--extreme-percentile` | `0.90` | Limiar do rejection sampling |
| `--n-per-cluster` | `3000` | Amostras fixas por cluster |
| `--n-per-cluster-ratio` | `None` | Fração proporcional ao real, ex: `0.3` = 30% do treino |
| `--exp-name` | auto (`expN`) | Nome do experimento |

> **Importante:** use sempre `--n-per-cluster-ratio` em vez de `--n-per-cluster`.
> A versão fixa pode fazer clusters pequenos serem dominados
> por sintéticos, degradando o treino.

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

# Com augmentation via YAML
modal run src/modal/cluster_lstm.py \
    --config experiment_cluster_lstm_modal.yaml \
    --augmentation-method extreme_gan
```

### GAN por cluster + LazyPredict com comparação (`cluster_gan` → `cluster_lazy`)

Fluxo completo recomendado — gera sintéticos e avalia todos os modelos com e sem augmentation:

```bash
# Passo 1: treinar a GAN e gerar os sintéticos (com as 27 features atuais)
modal run src/modal/cluster_gan.py \
    --epochs 10000 \
    --hparams-json '{"z_dim": 32, "hidden_dim": 64, "crit_repeats": 5, "c_lambda": 10.0, "lr": 5e-5}' \
    --exp-name convergence_test15
# → gera artifacts/gan_clusters/convergence_test15/synthetic_augment.csv

# Passo 2: LazyPredict com augmentation, vizinhos e avaliação de deploy
modal run --detach src/modal/cluster_lazy.py \
    --synthetic-csv artifacts/gan_clusters/convergence_test15/synthetic_augment.csv \
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
    --synthetic-csv artifacts/gan_clusters/convergence_test15/synthetic_augment.csv \
    --n-neighbor-clusters 2 \
    --eval-window biweekly \
    --exp-name neighbors2_biweekly

# Estratificado por trimestre + vizinhos
modal run --detach src/modal/cluster_lazy.py \
    --synthetic-csv artifacts/gan_clusters/convergence_test15/synthetic_augment.csv \
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
- `predictions/predictions_c{N}.csv` — predições de todos os modelos com `source` (real/sintético)
- `rolling_r2/rolling_r2_c{N}.csv` — R² por janela de deploy (top-5 modelos)
- `plots/rolling_r2_c{N}.png` — série temporal de R² por mês/quinzena
- `plots/scatter_all_c{N}.pdf` — scatter por modelo: treino (azul=real, laranja▲=sintético) + validação
- `plots/synth_comparison_c{N}.png` — ΔR² com vs sem sintéticos (verde=melhora, vermelho=piora)
- `lazy_cluster_results.csv` — consolidado com `R2_deploy_mean`, `R2_deploy_std`, `R2_deploy_min`

| Flag | Default | Descrição |
|------|---------|-----------|
| `--synthetic-csv` | `None` | CSV gerado por `cluster_gan` com features + alvo |
| `--n-neighbor-clusters` | `1` | Nº de clusters vizinhos cujos dados entram no treino |
| `--eval-window` | `monthly` | Janela de avaliação de deploy: `monthly` ou `biweekly` |
| `--stratify-seasons` | `False` | Treinar 1 modelo por cluster × trimestre (DJF/MAM/JJA/SON) |
| `--synth-n-above` | `None` (todos) | Máx. de sintéticos acima do P90 por cluster |
| `--synth-n-below` | `0` | Nº de sintéticos abaixo do P90 por cluster |
| `--extreme-percentile` | `0.90` | Limiar extremo/normal |
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
│   │   ├── cluster_gan.py          # modal run src/modal/cluster_gan.py
│   │   └── corrected_grid.py       # modal run src/modal/corrected_grid.py
│   ├── pipelines/                  # Lógica de cada pipeline
│   │   ├── common.py               # Constantes, splits e utils compartilhados
│   │   ├── cluster_lstm.py         # Subcomando: cluster_lstm
│   │   ├── cluster_mlp.py          # Subcomando: cluster_mlp
│   │   ├── cluster_lazy.py         # Subcomando: cluster_lazy
│   │   ├── cluster_gan.py          # Subcomando: cluster_gan
│   │   └── corrected_grid.py       # Subcomando: corrected_grid
│   ├── data/
│   │   ├── netcdf_loader.py        # Carrega e agrega INMET + ERA5 para diário
│   │   ├── cluster_assigner.py     # Spatial join estação → cluster
│   │   ├── climatology.py          # Climatologia por dia-do-ano e harmônica (sem leakage)
│   │   ├── interp.py               # Grade→ponto: nearest / bilinear
│   │   └── static_features.py      # Features estáticas por estação
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
│   ├── gan/
│   │   ├── models.py               # Generator + Critic (cWGAN-GP)
│   │   ├── losses.py               # Gradient penalty + losses WGAN
│   │   ├── conditioning.py         # One-hot labels + combine_vectors
│   │   ├── train.py                # Loop de treino WGAN-GP
│   │   ├── sampling.py             # sample() + sample_extremes()
│   │   └── evaluate.py             # Fréchet 1D, Wasserstein, KS
│   └── visualization/
│       └── cluster_plots.py
│
├── config/                         # YAMLs de experimento (cluster_lstm, schema v2)
│
├── dataset/
│   ├── raw/                        # NetCDF: INMET + ERA5
│   └── shp/                        # Shapefiles dos 14 clusters
│
├── artifacts/
│   ├── mlp_clusters/               # Saída do cluster_mlp
│   ├── lazy_clusters/              # Saída do cluster_lazy
│   ├── gan_clusters/               # Saída do cluster_gan
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
