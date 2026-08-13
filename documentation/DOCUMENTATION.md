# IRC Vendaval — Documentação Técnica

Este diretório contém a documentação detalhada de cada etapa da pipeline de correção de viés de rajadas de vento extremo. Cada arquivo é focado em um componente específico, com arquitetura, metodologia e artefatos.

---

## Índice

| # | Documento | Conteúdo |
|:--|:----------|:---------|
| 01 | [Visão Geral](01_visao_geral.md) | Fluxo geral do projeto, pipelines ativas e princípios de design |
| 02 | [Dados e Pré-processamento](02_dados_e_preprocessamento.md) | Fontes de dados, features (40+), splits temporais, regras de imputação e escalonamento |
| 03 | [Clusters Espaciais](03_clusters_espaciais.md) | Distribuição das 271 estações em 14 clusters e motivação da especialização espacial |
| 04 | [Pipeline `cluster_lstm`](04_cluster_lstm.md) | Arquitetura dual-head Multi-Task Learning, funções de perda e configuração YAML |
| 05 | [Pipeline `cluster_mlp`](05_cluster_mlp.md) | MLPRegressor com formulação de ratio multiplicativo, extreme-weighting, SHAP e artefatos |
| 06 | [Pipeline `cluster_lazy`](06_cluster_lazy.md) | Screening de 36 regressores do scikit-learn, famílias algorítmicas e modelos filtrados |
| 07 | [Pipeline `cluster_gan`](07_cluster_gan.md) | Metodologia ExGAN: EVT/GPD, Distribution Shifting, Conditional WGAN-GP e Nearest-Neighbor |
| 08 | [Inferência e Artefatos](08_inferencia_e_artefatos.md) | Correção espacial (sklearn + Keras), grade NetCDF corrigida e artefatos comuns |

---

## Como usar esta pasta

- Use os arquivos numerados acima como **documentação principal** de cada componente.
- Use o [README](../README.md) para a **visão executiva**, estrutura do projeto e comandos de execução.
- Para detalhes de configuração e hiperparâmetros, consulte o docstring do respectivo script em `src/modal/` ou `src/pipelines/`.
