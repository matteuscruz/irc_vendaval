<p align="center">
  <img src="figs/extreme_winds.png" alt="IRC Vendaval" width="100%">
</p>

# IRC Vendaval — Correção de Viés de Rajadas de Vento Extremo

Sistema de *bias correction* de rajadas máximas diárias de vento (`daily_wind_gust_max`) do ERA5 em relação a observações INMET, estratificado por **14 clusters espaciais** e com foco em eventos extremos (vendavais).

---

## Problema

A reanálise ERA5 (ECMWF) subestima sistematicamente rajadas máximas observadas por estações INMET, especialmente durante eventos extremos (vendavais, frentes frias, linhas de instabilidade). Este projeto treina modelos de **correção de viés multiplicativo** — dado apenas o ERA5, o sistema prediz a razão `INMET / ERA5` e reconstrói a rajada corrigida:

$$\hat{y}_{\text{corrigido}} = \hat{\text{Ratio}} \times \text{ERA5}_{\text{wind\_mag\_max}}$$

> **Sem vazamento de dados:** todas as features de entrada vêm exclusivamente do ERA5. Nenhuma observação INMET entra como feature — apenas como target no treino.

---

## Pipelines

O projeto opera através de quatro pipelines de treino e uma de inferência, todas executáveis via `main.py` ou na nuvem via [Modal](https://modal.com):

| Pipeline | Descrição | Documentação |
|:---------|:----------|:-------------|
| `cluster_lstm` | LSTM dual-head (Multi-Task Learning) com cabeças especialistas: normal (Huber) e extrema (`robust_extreme_loss`) | [04_cluster_lstm.md](documentation/04_cluster_lstm.md) |
| `cluster_mlp` | `MLPRegressor` (scikit-learn) com extreme-weighting via oversampling ponderado | [05_cluster_mlp.md](documentation/05_cluster_mlp.md) |
| `cluster_lazy` | Screening automático de 36 regressores via `LazyRegressor` para seleção do campeão por cluster | [06_cluster_lazy.md](documentation/06_cluster_lazy.md) |
| `cluster_gan` | cWGAN-GP condicional + EVT/GPD para geração de dados sintéticos de extremos | [07_cluster_gan.md](documentation/07_cluster_gan.md) |
| `corrected_grid` | Inferência espacial: aplica os modelos treinados sobre a grade ERA5 completa e gera NetCDFs corrigidos | [08_inferencia_e_artefatos.md](documentation/08_inferencia_e_artefatos.md) |

> **Fluxo recomendado:**
> ```
> cluster_gan  →  cluster_lazy (--synthetic-csv --n-neighbor-clusters 1 --eval-window monthly)
> ```
> A GAN gera dados sintéticos de extremos; o LazyPredict avalia dezenas de modelos usando
> augmentação, dados de clusters vizinhos e avaliação por janelas de deploy. Para a rede
> neural principal, o caminho ativo é `cluster_lstm` com a variante `cluster_dual_head_lstm`.

---

## Dados e Splits

| Fonte | Descrição |
|:------|:----------|
| `dataset/raw/INMET_Stratified.nc` | Rajada máxima diária observada (target) — 271 estações, 2000–2025 |
| `dataset/raw/ERA5_Stratified.nc` | Reanálise ERA5 co-localizada (features de entrada) |
| `dataset/shp/shp_vento.shp` | Polígonos dos 14 clusters espaciais |

| Split | Período | Uso |
|:------|:--------|:----|
| **Treino** | 2008–2018 | Ajuste dos modelos |
| **Validação** | 2019 | Early stopping / seleção de hiperparâmetros |
| **Teste** | 2020–2025 | Avaliação final fora da amostra (6 anos) |

> Para mais detalhes sobre features, fontes de dados e pré-processamento, veja [02_dados_e_preprocessamento.md](documentation/02_dados_e_preprocessamento.md).
> Para a distribuição de estações por cluster, veja [03_clusters_espaciais.md](documentation/03_clusters_espaciais.md).

---

## Como Rodar

### Dispatcher unificado (`main.py`)

```bash
python main.py --help
# Subcomandos: cluster_lstm, cluster_mlp, cluster_lazy, cluster_gan, corrected_grid
```

### Execução local

```bash
# LSTM dual-head
python main.py cluster_lstm --config config/experiment_cluster_lstm_modal.yaml

# MLP com extreme-weighting
python main.py cluster_mlp --output-dir artifacts/mlp_clusters

# LazyPredict benchmark
python main.py cluster_lazy --output-dir artifacts/lazy_clusters \
    --n-neighbor-clusters 1 --eval-window monthly

# GAN augmentation
python main.py cluster_gan --output-dir artifacts/gan_clusters --epochs 300
```

### Execução na nuvem (Modal)

```bash
# Cada pipeline tem seu entrypoint Modal correspondente
modal run src/modal/cluster_lstm.py
modal run src/modal/cluster_mlp.py
modal run src/modal/cluster_lazy.py
modal run src/modal/cluster_gan.py

# Só baixar artefatos de uma run anterior
modal run src/modal/cluster_lazy.py --only-download --local-dir artifacts/lazy_modal

# Script unificado para rodar todas as pipelines
./run_modal.sh
```

> Para a lista completa de flags e exemplos de uso avançado de cada pipeline, consulte a
> documentação específica em [`documentation/`](documentation/).

---

## Estrutura do Projeto

```
irc_vendaval/
│
├── main.py                           # Dispatcher unificado (subcomandos)
├── run_modal.sh                      # Script para rodar todas as pipelines Modal
├── dashboard.py                      # Dashboard Streamlit de comparação
│
├── src/
│   ├── modal/                        # Entrypoints Modal (1 por pipeline)
│   │   ├── cluster_lstm.py
│   │   ├── cluster_mlp.py
│   │   ├── cluster_lazy.py
│   │   ├── cluster_gan.py
│   │   └── corrected_grid.py
│   ├── pipelines/                    # Lógica de cada pipeline
│   │   ├── common.py                 # Constantes, splits, features e utils compartilhados
│   │   ├── cluster_lstm.py
│   │   ├── cluster_mlp.py
│   │   ├── cluster_lazy.py
│   │   ├── cluster_gan.py
│   │   └── metrics_schema.py         # Schema unificado de resultados e predições
│   ├── data/                         # Carregamento e processamento de dados
│   │   ├── netcdf_loader.py          # Carrega e agrega INMET + ERA5 para diário
│   │   ├── cluster_assigner.py       # Spatial join estação → cluster
│   │   ├── climatology.py            # Média ERA5 por dia-do-ano
│   │   ├── era5_18utc_loader.py      # Features ERA5 snapshot 18UTC (Paraná)
│   │   ├── era5_basin_loader.py      # Features ERA5-Basin agregadas
│   │   ├── bt55_loader.py            # Temperatura de brilho ≤55°C (convecção profunda)
│   │   └── static_features.py        # Features estáticas por estação (TRWindBC)
│   ├── pipeline/                     # Módulos LSTM / augmentação / treino
│   │   ├── augmentation/             # Augmenters: ExGAN, TabularGAN, Diffusion
│   │   ├── data/                     # ClusterPreprocessor (dados sequenciais)
│   │   ├── loss/                     # Funções de perda customizadas (Huber, extreme)
│   │   ├── training/                 # Trainers (Keras/TF)
│   │   └── validation/              # Métricas e holdout espacial
│   ├── inference/                    # Inferência espacial → grade corrigida
│   │   ├── spatial_correction.py     # Modelos sklearn (MLP/Lazy)
│   │   └── spatial_correction_dl.py  # Modelos Keras (LSTM)
│   ├── models/                       # Builders de arquitetura (LSTM dual-head)
│   ├── utils/                        # ArtifactManager, seeds, heartbeat
│   └── visualization/               # Plots por cluster, espaciais
│
├── config/                           # YAMLs de experimento (LSTM)
├── scripts/                          # Ablation studies, comparações, build de dataset
│
├── dataset/
│   ├── raw/                          # NetCDFs e CSVs fonte
│   └── shp/                          # Shapefiles dos 14 clusters
│
├── artifacts/                        # Saídas de todas as pipelines (por experimento)
│
├── documentation/                    # Documentação técnica detalhada por etapa
│   ├── DOCUMENTATION.md              # Índice da documentação
│   ├── 01_visao_geral.md
│   ├── 02_dados_e_preprocessamento.md
│   ├── 03_clusters_espaciais.md
│   ├── 04_cluster_lstm.md
│   ├── 05_cluster_mlp.md
│   ├── 06_cluster_lazy.md
│   ├── 07_cluster_gan.md
│   └── 08_inferencia_e_artefatos.md
│
├── test/                             # Testes unitários (pytest)
├── notebooks/                        # Notebooks exploratórios
├── research/                         # Experimentos e análises
├── reports/                          # Relatórios gerados
└── ref/                              # Referências bibliográficas
```

---

## Setup Local

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

### Testes

```bash
uv run pytest test/ -q
```

---

## Documentação Detalhada

Toda a documentação técnica de arquitetura, metodologia e artefatos está em [`documentation/`](documentation/DOCUMENTATION.md). Consulte o índice:

| Documento | Conteúdo |
|:----------|:---------|
| [01 — Visão Geral](documentation/01_visao_geral.md) | Fluxo geral do projeto e princípios de design |
| [02 — Dados e Pré-processamento](documentation/02_dados_e_preprocessamento.md) | Fontes de dados, features (40+), splits temporais, regras de imputação |
| [03 — Clusters Espaciais](documentation/03_clusters_espaciais.md) | Distribuição das 271 estações em 14 clusters e motivação |
| [04 — Pipeline LSTM](documentation/04_cluster_lstm.md) | Arquitetura dual-head, Multi-Task Learning, funções de perda |
| [05 — Pipeline MLP](documentation/05_cluster_mlp.md) | Formulação do ratio, extreme-weighting, SHAP, artefatos |
| [06 — Pipeline Lazy](documentation/06_cluster_lazy.md) | 36 modelos avaliados, famílias algorítmicas, modelos filtrados |
| [07 — Pipeline GAN](documentation/07_cluster_gan.md) | Metodologia ExGAN, EVT/GPD, Distribution Shifting, WGAN-GP |
| [08 — Inferência e Artefatos](documentation/08_inferencia_e_artefatos.md) | Correção espacial, grade NetCDF, artefatos comuns |

---

## Referências

- *TRWindBC* — Ouarda & Houndekindo (Energy 2025): LSTM dual-branch com features estáticas por estação para correção de ERA5
- *GFS-NWP-Wind-Correction* — Kuang (2025): 3D-UNet com loss física para correção de campos GFS
- *Improving Probabilistic Forecasting in the Netherlands* — motivação metodológica EMOS
- *Adapting a deep convolutional RNN model with imbalanced regression loss* — forecasting spatio-temporal de eventos extremos
- DeepLearning.AI GANs Specialization (Coursera) — arquitetura cWGAN-GP
- Koenker & Bassett (1978) — Quantile Regression / Pinball Loss
