# Pipeline `cluster_mlp`

Esta pipeline treina redes neurais densas (`MLPRegressor`) especializadas por cluster geográfico, atuando como um modelo supervisionado tabular com correção de razão multiplicativa e ponderação de extremos.

---

## 1. Arquitetura do Modelo

O modelo é baseado na implementação do `scikit-learn`:

- **Topologia de Camadas**: Configuração padrão `hidden_layer_sizes=(128, 64)` com função de ativação `ReLU`.
- **Otimizador**: `adam` com taxa de aprendizado adaptativa (`learning_rate="adaptive"`).
- **Regularização**:
  - Regularização $L_2$ (`alpha=0.001` por padrão);
  - `early_stopping=True` reservando $10\%$ do conjunto de treino interno;
  - Tolerância de 30 épocas sem redução da perda (`n_iter_no_change=30`);
  - Limite máximo de iterações do solver: `max_iter=500`.

```text
Input (n_features)
    │
    ▼
Dense (128 units, ReLU)
    │
    ▼
Dense (64 units, ReLU)
    │
    ▼
Dense (1 unit, Linear) -> Prediz Ratio (INMET / ERA5)
```

---

## 2. Formulação do Alvo (Multiplicative Bias Correction)

Diferente de regredir o valor absoluto da rajada diretamente em $m/s$, o modelo aprende um **fator de correção multiplicativo**:

$$\text{Ratio} = \frac{y_{\text{INMET}}}{\max(\text{ERA5}_{\text{gust}}, 0.1)}$$

### Reconstrução na Inferência:
$$\hat{y}_{\text{final}} = \hat{\text{Ratio}} \times \max(\text{ERA5}_{\text{gust}}, 0.1)$$

**Por que essa abordagem?**
- Permite que a rede atue como um operador de calibração sobre a física de vento já prevista pela reanálise ERA5;
- Se a rede prever $\approx 1.0$, o modelo respeita a magnitude do ERA5; desvios acima de $1.0$ indicam subestimação sistemática da reanálise.

---

## 3. Ponderação e Oversampling de Extremos (`extreme_power`)

Devido à distribuição de cauda longa (eventos de vendaval são raros), o pipeline aplica uma ponderação não-linear com oversampling durante o treino:

1. **Cálculo dos Pesos Amostrais**:
   $$w_i = \left( \frac{y_i}{\max(y)} \right)^{\text{extreme\_power}}$$
   $$w_i \leftarrow \frac{w_i}{\bar{w}}$$
   *(onde $\text{extreme\_power} = 2.0$ por padrão)*

2. **Oversampling Probabilístico**:
   O número de repetições de cada amostra é dado por $\text{round}(w_i)$, permitindo que rajadas severas tenham peso equivalente a amostras frequentes no cálculo do gradiente sem quebrar a API padrão do `scikit-learn`.

---

## 4. Pré-processamento e Pipeline de Dados

1. **Imputação e Escalonamento**:
   - `SimpleImputer(strategy="mean")` para preenchimento de dados faltantes;
   - `RobustScaler()` para centralização pela mediana e escalonamento por IQR (robusto a outliers).
2. **Separação Temporal** (definido em `src/pipelines/common.py`):
   - **Treino**: 2008-01-01 a 2018-12-31 (com suporte a augmentação via `--synthetic-csv`);
   - **Validação**: 2019-01-01 a 2019-12-31 (utilizado para seleção de hiperparâmetros e métricas operacionais);
   - **Teste**: 2020-01-01 a 2025-12-31 (avaliação cega de generalização fora da amostra, 6 anos).
3. **Validação Espacial Cruzada (Opcional)**:
   - Suporte a `--validation-mode spatial-kfold` e `loocv` via `iter_holdout_folds` para testar capacidade de interpolação em estações não vistas.

---

## 5. Explicabilidade (XAI) & Diagnósticos

A cada execução por cluster, o pipeline gera:
- **SHAP (`shap.KernelExplainer`)**:
  - Gráficos *Beeswarm* de impacto e direção;
  - Gráfico de barras de $|SHAP|$ médio global;
  - Gráficos de dependência (`dependence_plot`) da feature mais relevante.
- **Permutation Importance**:
  - Mede a queda em $R^2$ ao permutar cada variável no conjunto de validação.
- **Curva de Resposta de Correção**:
  - Relação entre a intensidade do ERA5 e a razão de correção predita.

---

## 6. Artefatos Gerados

- **Modelos Treinados (`fitted_models/best_model_c{N}.joblib`)**:
  - Dicionário com `model`, `model_name`, `imputer`, `scaler`, `features`, `cluster_id` e flag `"target_kind": "ratio"`;
  - Utilizado diretamente por `src.inference.spatial_correction` para gerar mapas NetCDF corrigidos em grade contínua.
- **Tabelas de Resultados**:
  - `mlp_cluster_results.csv`: Métricas ($R^2$, RMSE, Bias, Bias@P90) para treino, validação e teste;
  - `predictions_by_station.csv`: Séries temporais observadas vs preditas por estação individual;
  - `feature_importance.csv`: Ranking de variáveis por permutação.
- **Visualizações**:
  - `cluster_report.pdf`: Relatório unificado com resumo executivo e todos os gráficos por cluster.

---

## 7. Quando Utilizar

- Excelente baseline tabular para comparar com modelos baseados em árvores (`cluster_lazy`) e sequenciais (`cluster_lstm`);
- Quando a relação entre variáveis atmosféricas e a razão de viés do ERA5 for não-linear, contínua e suave;
- Para auditoria detalhada de explicabilidade com SHAP e curvas de aprendizado por cluster.