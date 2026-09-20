# Visão Geral

O projeto corrige o viés de rajadas máximas diárias do ERA5 em relação ao INMET.
O alvo principal é `daily_wind_gust_max`, modelado como razão `INMET / ERA5` em
boa parte das pipelines.

## Fluxo atual

1. Carregamento dos dados do INMET (y) e do ERA5 (x) usados na pipeline.
2. Construção de features e janelas temporais.
3. Treino por cluster e, quando aplicável, por trimestre climático.
4. Geração de artefatos e avaliação em validação/teste.
5. Inferência espacial e geração da grade corrigida.

## Pipelines ativas

- `cluster_lstm`: LSTM de saída única em Keras/TensorFlow (diária ou horária, split por blocos de mês).
- `cluster_mlp`: `MLPRegressor` do scikit-learn com foco em extremos.
- `cluster_lazy`: benchmark com LazyPredict e seleção do melhor modelo por cluster.
- `cluster_gan`: geração de amostras sintéticas para augmentation.
- `corrected_grid`: reconstrução da grade corrigida a partir dos modelos salvos.

## Ideia central

A pipeline é modular, mas a estratégia é consistente:
- usar apenas features derivadas do ERA5 e metadados da estação;
- preservar separação temporal de treino/validação/teste;
- explorar especialização espacial por cluster;
- reforçar eventos extremos com augmentation ou weighting.