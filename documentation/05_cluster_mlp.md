# Pipeline `cluster_mlp`

Esta pipeline usa `MLPRegressor` do scikit-learn como baseline supervisionado por cluster.

## Características

- alvo treinado como razão `INMET / ERA5`;
- pré-processamento com imputação por média e `RobustScaler`;
- reamostragem dos extremos via `extreme_power`;
- avaliação com métricas gerais e de cauda.

## Fluxo

1. Monta o DataFrame tabular por cluster.
2. Aplica feature engineering e pré-processamento.
3. Oversampling ponderado dos eventos extremos.
4. Treina o `MLPRegressor`.
5. Salva o artefato com metadados para inferência espacial.

## Artefatos

- `.joblib` por cluster com modelo + imputer + scaler + features.
- CSVs de resultados por cluster e consolidados.
- gráficos de dispersão, distribuição e importância por permutação.

## Quando faz sentido

É a opção mais simples para testar hipóteses tabulares e comparar a força de
regularização implícita de árvores e boosting contra a rede neural.