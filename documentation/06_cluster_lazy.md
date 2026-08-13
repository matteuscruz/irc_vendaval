# Pipeline `cluster_lazy`

Esta pipeline faz o screening automático de modelos com LazyPredict por cluster.

## O que ela faz

- roda `LazyRegressor` sobre dezenas de regressors sklearn;
- filtra modelos muito lentos como `SVR`, `NuSVR`, `KernelRidge` e `GaussianProcessRegressor`;
- avalia o desempenho por cluster e, opcionalmente, por trimestre;
- gera métricas de deploy por janela temporal.

## O que costuma ganhar

Historicamente, árvores e boosting costumam dominar o ranking, como:

- `HistGradientBoostingRegressor`;
- `GradientBoostingRegressor`;
- `ExtraTreesRegressor`;
- às vezes `XGBRegressor`, `LGBMRegressor` e `CatBoostRegressor`.

## Modos importantes

- `--synthetic-csv`: injeta dados sintéticos do GAN no treino.
- `--n-neighbor-clusters`: adiciona dados de clusters vizinhos.
- `--stratify-seasons`: separa o treino por trimestre climático.
- `--eval-window`: mede estabilidade em janelas mensais ou quinzenais.

## Artefatos

- `predictions_c{N}.csv` com todas as previsões;
- `rolling_r2_c{N}.csv` com R² por janela;
- `lazy_cluster_results.csv` consolidado;
- modelo campeão salvo para inferência espacial.