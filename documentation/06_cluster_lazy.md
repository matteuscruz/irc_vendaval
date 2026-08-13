# Pipeline `cluster_lazy`

Esta pipeline faz o screening automático de modelos com LazyPredict por cluster.

## O que ela faz

- Executa triagem e benchmarking automático usando `LazyRegressor` sobre a suite de regressores do Scikit-Learn;
- Filtra modelos com complexidade assintótica $O(n^2 - n^3)$ que seriam proibitivos em datasets volumosos ou com agregação de vizinhos (`SVR`, `NuSVR`, `KernelRidge`, `GaussianProcessRegressor`);
- Avalia o desempenho por cluster geográfico e, opcionalmente, por trimestre (`DJF`, `MAM`, `JJA`, `SON`);
- Gera métricas de deploy por janela temporal deslizante (`monthly` ou `biweekly`).

## Modelos Avaliados (LazyRegressor)

O pipeline avalia **36 modelos ativos** divididos por famílias algorítmicas:

### 1. Ensembles, Árvores & Boosting (Geralmente os campeões)
- `HistGradientBoostingRegressor` (Scikit-Learn / LightGBM-like)
- `GradientBoostingRegressor`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `AdaBoostRegressor`
- `BaggingRegressor`
- `DecisionTreeRegressor`
- `ExtraTreeRegressor`

### 2. Modelos Lineares, Regularizados e Robustos
- `LinearRegression` (MQO padrão)
- `Ridge` e `RidgeCV` (Regularização $L_2$)
- `Lasso` e `LassoCV` (Regularização $L_1$)
- `ElasticNet` e `ElasticNetCV` (Combinação $L_1 + L_2$)
- `HuberRegressor` (Robusto a outliers)
- `RANSACRegressor` (Ajuste linear robusto por consenso amostral)
- `PassiveAggressiveRegressor` (Otimização online para grandes volumes)
- `SGDRegressor` (Gradiente descendente estocástico)
- `LinearSVR` (Support Vector Regression linear rápida)
- `QuantileRegressor` (Regressão por quantis via otimização linear)
- `BayesianRidge` (Regressão bayesiana com priores gaussianos)
- `Lars`, `LarsCV` (Least Angle Regression)
- `LassoLars`, `LassoLarsCV`, `LassoLarsIC` (Lasso via LARS com critério de informação)
- `OrthogonalMatchingPursuit`, `OrthogonalMatchingPursuitCV` (OMP esparso)

### 3. Modelos Lineares Generalizados (GLMs)
- `PoissonRegressor` (Distribuição de Poisson)
- `GammaRegressor` (Distribuição Gamma, útil para targets estritamente positivos)
- `TweedieRegressor` (Distribuição de Poisson-Gamma mista)

### 4. Redes Neurais, Vizinhos & Outros
- `MLPRegressor` (Multi-Layer Perceptron simples do Scikit-Learn)
- `KNeighborsRegressor` (K-ésimos Vizinhos mais Próximos)
- `TransformedTargetRegressor` (Meta-estimador para transformação de target)
- `DummyRegressor` (Baseline constante para comparação de referência)

### 5. Modelos Filtrados / Desativados (`_SLOW_MODELS`)
Filtrados preventivamente devido ao custo temporal e consumo de memória excessivo:
- `SVR` (Kernel RBF/Poly)
- `NuSVR`
- `KernelRidge`
- `GaussianProcessRegressor`

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