# IRC Vendaval — Correção de Viés de Rajadas de Vento Extremo

Bias correction de `daily_wind_gust_max` do ERA5 em relação a observações INMET,
estratificado em **6 clusters espaciais**. Foco em eventos extremos de vento (vendavais).

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
> A GAN gera dados sintéticos de extremos; o LazyPredict avalia 30+ modelos por cluster
> usando 27 features, dados de clusters vizinhos e avaliação por janelas de deploy (mensal/quinzenal).

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
feature (apenas como target no treino).

---

## 2. Dados

| Arquivo | Descrição |
|---------|-----------|
| `dataset/raw/INMET_Stratified.nc` | Rajada máxima diária observada — 30 estações, 2000–2023 |
| `dataset/raw/ERA5_Stratified.nc` | Reanálise ERA5 co-localizada nas estações — agregação diária |
| `dataset/shp/shp_vento.shp` | Polígonos dos 6 clusters espaciais (spatial join) |

O dataset é compartilhado via Modal Volume `irc-vendaval-dataset`. O upload ocorre
automaticamente na primeira execução de qualquer pipeline Modal.

### Split temporal

| Split | Período | Uso |
|-------|---------|-----|
| **Treino** | 2000–2022 | Fit do modelo (`cluster_lazy`) / 2000–2020 (`mlp_pytorch`) |
| **Validação** | 2023 | Avaliação do `cluster_lazy` — por janelas mensais/quinzenais |
| **Val / Teste** | 2021–2022 / 2023 | Early stopping e teste final do `mlp_pytorch` |

> No `cluster_lazy` a validação é avaliada por **janelas de deploy** (mensal ou quinzenal)
> com `--eval-window`, refletindo o cenário real de inferência mensal/quinzenal.

### Features utilizadas (27 no total)

| Grupo | Features |
|-------|---------|
| Vento | `10m_u/v_component_of_wind`, `wind_mag`, `wind_mag_max`, `wind_mag_min`, `wind_mag_std` |
| Direção do vento | `wind_dir_sin`, `wind_dir_cos` (arctan2 do vetor U/V codificado ciclicamente) |
| Convecção | `gust_factor` (pico/média intradiária — proxy convectivo) |
| Persistência | `lag1_wind_mag_max`, `lag2_wind_mag_max`, `lag3_wind_mag_max`, `lag7_wind_mag_max` |
| Tendência | `rolling7d_wind_mag_max` (média móvel 7 dias — evolução sinótica) |
| Termodinâmica | `2m_temperature`, `2m_dewpoint_temperature`, `relative_humidity`, `t2m_range` |
| Pressão | `surface_pressure`, `pressure_tendency` |
| Precipitação | `total_precipitation` |
| Sazonalidade | `day_sin`, `day_cos` |
| Climatologia ERA5 | `era5_clim_wind` (média histórica do `wind_mag_max` por dia-do-ano) |
| Localização | `latitude`, `longitude` (coordenadas da estação) |
| Climatologia INMET* | `gust_P50` (mediana observada por estação, calculada só no treino — sem leakage) |

\* `gust_P50` é calculado em runtime a partir do split de treino; amostras sintéticas recebem a mediana do cluster.

---

## 3. Clusters Espaciais

6 regiões climáticas definidas por shapefile, com regimes distintos de vento:

| Cluster | Estações | Característica |
|---------|----------|----------------|
| C1 | 5 | Litoral sul |
| C2 | 8 | Planalto gaúcho |
| C3 | 3 | Serra catarinense |
| C4 | 10 | Interior RS/SC |
| C5 | 3 | Planalto central SC |
| C6 | 1 | Região de transição |

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

### 4.2 LSTM Dual-Head por Cluster (`cluster_lstm`)
Pipeline sequencial usando dados NetCDF brutos. Arquitetura dual-head:
cabeça de valor + cabeça de incerteza por cluster espacial. Suporta
também a variante **TRWindBC** com condicionamento por features estáticas
de estação (lat, lon, percentis de rajada).

### 4.3 Screening LazyPredict (`cluster_lazy`)
30+ modelos sklearn avaliados por cluster, com as seguintes capacidades:

**Feature engineering completo (27 features):** o pipeline usa o conjunto expandido de features descrito na seção 2, incluindo direção do vento, lags longos, rolling stats e a climatologia observada por estação (`gust_P50`).

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

### 4.6 GAN de Augmentation completo (`gan_augment`) — vetor features + target
GAN condicional por cluster para gerar o vetor completo `[features ERA5 + target]`:
- Treina na distribuição completa (features ERA5 + target INMET)
- Gera apenas amostras acima do P90 via rejection sampling
- `n_per_cluster_ratio`: número de sintéticos proporcional ao tamanho real
  do cluster — evita dominância de dados sintéticos em clusters pequenos
- Sanity check automático: compara P90/P95 real × sintético por cluster

### 4.6 MLP PyTorch com Pinball Loss (`mlp_pytorch`) — atual
Migração de sklearn para PyTorch para suportar loss customizada:
- **Pinball loss** no quantil τ: penaliza `τ × (sub-estimativa)` e
  `(1-τ) × (sobre-estimativa)` — com τ=0.9 sub-estimativas custam 9× mais
- **BatchNorm + Dropout** por camada — regularização mais eficaz que L2 puro
- **ReduceLROnPlateau** — learning rate adaptativo quando validação estagna
- **Early stopping** com restauração do melhor estado
- Pesos salvos por cluster (`model_cluster_N.pt`) para reutilização

---

## 5. Pipeline Atual — MLP PyTorch

### Arquitetura

```
Features ERA5 (20)
       │
  [Linear → BatchNorm → ReLU → Dropout] × N camadas
       │
  [Linear → scalar]
       │
  razão predita × ERA5_wind_mag_max = ŷ_corrigido (m/s)
```

### Loss: Pinball (Quantile Loss)

```
L(ŷ, y; τ) = mean[ τ·max(y−ŷ, 0)  +  (1−τ)·max(ŷ−y, 0) ]
```

Com τ=0.9: sub-estimativas pesam 9× mais que sobre-estimativas.
Isso direciona o modelo a não ignorar os extremos.

### Artefatos gerados por experimento

```
artifacts/mlp_pytorch/expN/
  plots/
    per_cluster/
      loss_curve_{C}.png          # curva treino vs validação
      scatter_obs_pred_{C}.png    # scatter treino | validação | teste
    summary/
      metrics_R2.png              # R² por cluster × split
      metrics_rmse_p90.png
      metrics_bias_p90.png
  csv/
    mlp_pytorch_results.csv       # métricas de todos os clusters e splits
  model_cluster_{C}.pt            # pesos PyTorch por cluster
  run_meta.json                   # hiperparâmetros e metadados do experimento
```

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
# lista: lstm_baseline, cluster_lstm, cluster_mlp, cluster_lazy,
#        cluster_gan, gan_augment, mlp_pytorch
```

### MLP PyTorch (`mlp_pytorch`)

```bash
# Sem augmentation
python main.py mlp_pytorch \
    --output-dir artifacts/mlp_pytorch \
    --hidden-layers 128,64,32 \
    --tau 0.9 \
    --dropout 0.2 \
    --weight-decay 0.01 \
    --max-epochs 500

# Com augmentation GAN
python main.py mlp_pytorch \
    --output-dir artifacts/mlp_pytorch \
    --hidden-layers 128,64,32 \
    --tau 0.9 \
    --synthetic-csv artifacts/gan_augment/expN/synthetic_augment.csv
```

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--hidden-layers` | `128,64,32` | Camadas ocultas separadas por vírgula |
| `--tau` | `0.9` | Quantil da pinball loss (0.5=mediana, 0.9=extremos) |
| `--dropout` | `0.2` | Dropout por camada |
| `--weight-decay` | `0.01` | L2 no otimizador Adam |
| `--lr` | `1e-3` | Learning rate inicial |
| `--max-epochs` | `500` | Máximo de épocas |
| `--patience` | `30` | Épocas sem melhora antes do early stopping |
| `--batch-size` | `256` | Tamanho do batch |
| `--synthetic-csv` | `None` | CSV de augmentation gerado pela GAN |
| `--exp-name` | auto (`expN`) | Nome do experimento |

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

### GAN Augmentation (`gan_augment`)

```bash
# Local (CPU, para testar)
python main.py gan_augment \
    --output-dir artifacts/gan_augment \
    --epochs 300 \
    --n-per-cluster-ratio 0.3 \
    --exp-name exp_local

# Na nuvem com GPU T4 (recomendado)
modal run src/modal/gan_augment.py \
    --epochs 300 \
    --n-per-cluster-ratio 0.3

# Só baixar artefatos de run anterior
modal run src/modal/gan_augment.py --only-download \
    --local-dir artifacts/gan_augment_modal
```

| Flag | Default | Descrição |
|------|---------|-----------|
| `--epochs` | `300` | Épocas de treino da GAN |
| `--z-dim` | `32` | Dimensão do ruído latente do Generator |
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
> A versão fixa pode fazer clusters pequenos (ex: C6 com 1 estação) serem dominados
> por sintéticos, degradando o treino.

### LSTM Dual-Head por Cluster (`cluster_lstm`)

```bash
# Local
python main.py cluster_lstm \
    --config config/experiment_cluster_tr_lstm_modal.yaml

# Na nuvem (Modal)
modal run src/modal/cluster_lstm.py \
    --config experiment_cluster_tr_lstm_modal.yaml

# Com augmentation via YAML
modal run src/modal/cluster_lstm.py \
    --config experiment_cluster_tr_lstm_modal.yaml \
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
│   │   ├── lstm_baseline.py        # modal run src/modal/lstm_baseline.py
│   │   ├── cluster_lstm.py         # modal run src/modal/cluster_lstm.py
│   │   ├── cluster_mlp.py          # modal run src/modal/cluster_mlp.py
│   │   ├── cluster_lazy.py         # modal run src/modal/cluster_lazy.py (fan-out)
│   │   ├── cluster_gan.py          # modal run src/modal/cluster_gan.py
│   │   ├── gan_augment.py          # modal run src/modal/gan_augment.py (GPU T4)
│   │   └── mlp_pytorch.py          # modal run src/modal/mlp_pytorch.py (GPU T4)
│   ├── pipelines/                  # Lógica de cada pipeline
│   │   ├── common.py               # Constantes, splits e utils compartilhados
│   │   ├── lstm_baseline.py        # Subcomando: lstm_baseline
│   │   ├── cluster_lstm.py         # Subcomando: cluster_lstm
│   │   ├── cluster_mlp.py          # Subcomando: cluster_mlp
│   │   ├── cluster_lazy.py         # Subcomando: cluster_lazy
│   │   ├── cluster_gan.py          # Subcomando: cluster_gan
│   │   ├── gan_augment.py          # Subcomando: gan_augment (GANConfig aqui)
│   │   └── mlp_pytorch.py          # Subcomando: mlp_pytorch (WindGustMLP aqui)
│   ├── data/
│   │   ├── netcdf_loader.py        # Carrega e agrega INMET + ERA5 para diário
│   │   ├── cluster_assigner.py     # Spatial join estação → cluster
│   │   ├── climatology.py          # Média ERA5 por dia-do-ano (sem leakage)
│   │   └── static_features.py      # Features estáticas por estação (TRWindBC)
│   ├── models/
│   │   └── cluster_tr_lstm_builder.py  # LSTM dual-branch com condicionamento estático
│   ├── pipeline/                   # Módulos LSTM / EMOS (baseline)
│   │   ├── data/
│   │   │   └── cluster_preprocessor.py
│   │   ├── training/
│   │   │   ├── cluster_trainer.py
│   │   │   └── cluster_tr_trainer.py  # Variante TRWindBC
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
├── config/                         # YAMLs de experimento (cluster_lstm / lstm_baseline)
│
├── dataset/
│   ├── raw/                        # NetCDF: INMET + ERA5
│   └── shp/                        # Shapefiles dos 6 clusters
│
├── artifacts/
│   ├── mlp_pytorch/                # Saída do mlp_pytorch
│   ├── mlp_clusters/               # Saída do cluster_mlp
│   ├── lazy_clusters/              # Saída do cluster_lazy
│   ├── gan_clusters/               # Saída do cluster_gan
│   └── gan_augment/                # Dados sintéticos da cWGAN-GP
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
