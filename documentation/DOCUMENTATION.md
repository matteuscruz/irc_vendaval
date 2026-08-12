# IRC Vendaval — Documentação de Técnicas Testadas

Registro cronológico das abordagens experimentadas para melhorar a correção de viés de
rajadas de vento (`daily_wind_gust_max` ERA5 → INMET) nas 30 estações do sul do Brasil.

---

## Sumário de Resultados

| Técnica | R² validação (referência C3) | Observação |
|---|---|---|
| ERA5 bruto (sem correção) | ~0.30 | Baseline |
| EMOS probabilístico (CNN/LSTM) | ~0.45 | Overfitting severo |
| MLP sklearn + extreme weighting | ~0.36 | Treino R²≈0.85 — overfitting |
| LSTM dual-head por cluster | ~0.50 | Melhora com clustering espacial |
| LazyPredict (9 features) | ~0.55 | Gradient Boosting domina |
| LazyPredict (26 features) | ~0.67 | +12 pp com features completas |
| + GAN augmentation | ~0.67–0.70 | Ganho variável por cluster |
| + Clusters vizinhos | em avaliação | Aumenta volume de treino |
| + Lags observação INMET | em avaliação | Maior ganho esperado |
| + Estratificação por trimestre | em avaliação | Modelos mais específicos |

---

## 1. Baseline: ERA5 Bruto

O `wind_mag_max` do ERA5 sem qualquer correção subestima sistematicamente as rajadas
INMET, especialmente na cauda (eventos acima do P90). Usado como referência inferior
em todos os experimentos.

**Limitação principal:** ERA5 é uma reanálise de grade ~31 km; estações capturam
efeitos locais (orografia, rugosidade) que o modelo global não resolve.

---

## 2. EMOS Probabilístico — Baseline Neural (Primeiro Approach)

**Modelos testados:**

| Modelo | Arquitetura | Input |
|---|---|---|
| `NNModel` | Dense 1-D | Features meteorológicas tabulares |
| `NNConvModel` | CNN + dense | Grades espaciais de velocidade |
| `CNNModel` | Autoencoder 3-D CNN (PyTorch) | Volumétrico |
| `CGRU` / `CLSTM` | Células convolucionais recorrentes | Sequências temporais |

**Loss:** Threshold-weighted CRPS (twCRPS) — penaliza mais erros acima do limiar
de evento extremo. Abordagem probabilística: saída é uma distribuição, não um escalar.

**Resultado:** R² moderado (~0.45), instabilidade no treino. Complexidade arquitetural
alta para o volume de dados disponível (30 estações × 23 anos).

**Por que foi descartado como frente principal:** custo computacional elevado,
difícil de iterar rapidamente, vantagem do output probabilístico não se traduziu
em ganhos práticos de calibração sobre os extremos.

---

## 3. Clustering Espacial

**Técnica:** Spatial join das 30 estações INMET com polígonos de 6 regiões climáticas
(`shp/shp_vento.shp`). Cada pipeline treina um modelo independente por cluster.

| Cluster | Estações | Regime |
|---|---|---|
| C1 | 5 | Litoral sul |
| C2 | 8 | Planalto gaúcho |
| C3 | 3 | Serra catarinense |
| C4 | 10 | Interior RS/SC |
| C5 | 3 | Planalto central SC |
| C6 | 1 | Região de transição |

**Ganho:** +5–8 pp de R² em relação a modelos globais (sem separação espacial).

**Mecanismo:** cada cluster tem regime meteorológico distinto — a Serra Catarinense
(C3) é dominada por jatos de sul e eventos orográficos; o Litoral (C1) por sistemas
frontais oceânicos. Misturar esses regimes aumenta o ruído não-explicável por ERA5.

---

## 4. LSTM Dual-Head por Cluster (`cluster_lstm`)

**Arquitetura:** LSTM com duas cabeças — valor esperado + incerteza (heteroscedastic).
Variante **TRWindBC** (inspirada em Ouarda & Houndekindo, 2025): adiciona features
estáticas de estação (lat, lon, percentis históricos de rajada) como condicionamento.

**Split:** dados sequenciais em janelas temporais — preserva ordem temporal no batch.

**Resultado:** ~0.50 de R² médio. Melhor que EMOS, mas limitado pelo volume de dados:
o cluster com menos estações (C6: 1 estação) tem ~8 000 amostras de treino, insuficiente
para LSTM convergir bem.

**Por que foi parcialmente substituído:** modelos tabulares (sklearn) com features bem
construídas superam LSTM quando o dataset é pequeno e não há longas dependências
temporais explícitas no input.

---

## 5. MLP sklearn com Extreme Weighting (`cluster_mlp`)

**Ideia:** `MLPRegressor` (sklearn) com `sample_weight ∝ daily_wind_gust_max^extreme_power`
— amostras de evento extremo pesam mais no gradiente.

**Resultado:** treino R²≈0.85, validação R²≈0.36. Overfitting expressivo.

**Diagnóstico:**
- Loss MSE ignora a assimetria — sub e sobre-estimativas penalizadas igualmente
- O weighting aumenta o overfitting (pontos raros de cauda são memorizados)
- Sem regularização temporal — cada ponto é tratado de forma i.i.d.

**Legado:** motivou a migração para Pinball Loss (ver seção 8) e o screening via
LazyPredict para identificar famílias de modelos mais adequadas.

---

## 6. LazyPredict — Screening de 30+ Modelos (`cluster_lazy`)

**Técnica:** avalia em paralelo todos os regressores do scikit-learn via `LazyRegressor`
para cada cluster. Identifica qual família de modelo funciona melhor sem ajuste manual.

**Descoberta:** árvores com Gradient Boosting (HistGradientBoostingRegressor,
ExtraTreesRegressor, GradientBoostingRegressor) consistentemente dominam o ranking,
superando LSTM e MLP em todos os clusters.

**Por que funciona bem aqui:**
- Invariante à escala — não precisa de normalização cuidadosa
- Lida bem com features correlacionadas (ex: lag1 e lag2 de wind_mag_max)
- Menos sensível a dados faltantes / NaN (relevante com features de lag)
- Treino muito mais rápido, facilita iteração

---

## 7. Feature Engineering (Evolução em 3 fases)

### Fase 1 — 9 features locais (inicial)

Apenas as features básicas de vento e termodinâmica usadas diretamente da ERA5 sem
derivações. R² ~0.55.

### Fase 2 — 26 features completas

Expansão para o conjunto completo de features derivadas do ERA5:

| Grupo | Features adicionadas |
|---|---|
| Direção do vento | `wind_dir_sin`, `wind_dir_cos` (arctan2 U/V, codificação cíclica) |
| Persistência longa | `lag7_wind_mag_max` (padrão semanal sinótico) |
| Tendência ERA5 | `rolling7d_wind_mag_max` (média móvel 7 dias) |
| Convecção | `gust_factor` = wind_mag_max / wind_mag (razão pico/média intradiária) |
| Termodinâmica | `relative_humidity`, `t2m_range`, `2m_dewpoint_temperature` |
| Sinótica | `pressure_tendency` (queda rápida precede vendavais) |
| Climatologia ERA5 | `era5_clim_wind` (média histórica do wind_mag_max por dia-do-ano, sem leakage) |
| Localização | `latitude`, `longitude` (coordenadas da estação) |

**Ganho:** +12 pp de R² (de ~0.55 para ~0.67 no C3).

**Climatologia `gust_P50`:** mediana da rajada observada por estação, calculada
exclusivamente sobre o split de treino — sem leakage. Amostras sintéticas recebem
a mediana do cluster.

### Fase 3 — 35 features: conectividade temporal

Adição de features que dão ao modelo "memória" dos dias anteriores:

| Feature | Origem | Sinal |
|---|---|---|
| `lag1/2/3_gust_obs` | INMET observado | Rajada de ontem/anteontem — componente autoregressivo do alvo |
| `rolling7d_gust_obs` | INMET observado | Intensidade média da semana anterior por estação |
| `wind_mag_max_anom` | ERA5 | Desvio do dia em relação à média dos 7 dias anteriores |
| `rolling3d_wind_mag_max` | ERA5 | Janela sinótica curta (~3 dias — passagem frontal) |
| `lag1_pressure_tendency` | ERA5 | Tendência de pressão de ontem (aprofundamento de baixas) |
| `month_sin` / `month_cos` | Calendário | Sazonalidade mensal cíclica (complementa `day_sin/cos`) |

**Princípio dos lags INMET:** rajadas têm alta autocorrelação — `lag1_gust_obs` é o
preditor isolado mais poderoso. Os modelos sklearn passam a ter "memória" dos dias
anteriores sem precisar de arquitetura recorrente.

**Validade em deploy:** o sistema opera em correção mensal em lote — a observação do
dia anterior sempre está disponível quando se corrige o dia atual.

---

## 8. Data Augmentation com cWGAN-GP (`cluster_gan` / `gan_augment`)

**Motivação:** clusters pequenos (C3: 3 estações, C6: 1 estação) têm poucos eventos
acima do P90 no período de treino — ~200–400 amostras de extremo para 23 anos.

**Arquitetura:** Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP):
- Generator: ruído latente z + one-hot do cluster → rajada sintética
- Critic: detecta amostras reais vs sintéticas, condicionado ao cluster
- Gradient Penalty (GP) substitui clipping — garante condição Lipschitz sem
  conflito com o regularizador

**Rejection sampling:** apenas amostras sintéticas acima do P90 são retidas,
enriquecendo a cauda da distribuição de treino.

**Atribuição de features ERA5:** para cada alvo sintético `y_s`, o nearest-neighbor
no espaço do alvo dentro do mesmo cluster determina quais features ERA5 usar —
preserva a correlação feature/target sem inventar valores de ERA5.

**HPO via Optuna:** busca de hiperparâmetros com avaliação específica na cauda P90+
(Wasserstein 1D entre real e sintético apenas nos extremos). Melhor configuração:
`z_dim=32`, `hidden_dim=64`, `c_lambda=10.0`, `lr=5e-5`.

**Resultado:** ganho variável por cluster (0–+4 pp de R²). Maior benefício em
clusters pequenos. Risco de degradação em clusters grandes se `n_per_cluster`
for fixo (sintéticos podem dominar o treino) — usar `--n-per-cluster-ratio`.

---

## 9. Compartilhamento entre Clusters Vizinhos (`--n-neighbor-clusters`)

**Motivação:** clusters com poucas estações têm volume de treino limitado. Clusters
geograficamente próximos têm regimes meteorológicos similares e podem complementar.

**Técnica:** distância euclidiana entre centroides de estações (média lat/lon por
cluster). Os N clusters mais próximos têm seus dados reais e sintéticos incluídos no
treino do cluster-alvo.

**Validação permanece isolada:** apenas dados do cluster-alvo entram na avaliação,
preservando a medição de performance sem contaminação.

**Configuração padrão:** `--n-neighbor-clusters 1` (um vizinho mais próximo).

---

## 10. Estratificação por Trimestre Climático (`--stratify-seasons`)

**Motivação:** um único modelo por cluster não captura diferenças sazonais —
os mecanismos de vendaval no inverno (jatos polares, sistemas frontais intensos)
diferem fundamentalmente do verão (convecção, linhas de instabilidade).

**Técnica:** treina um modelo LazyPredict separado para cada combinação
(cluster, trimestre). Com 6 clusters × 4 trimestres = **24 modelos independentes**.

| Trimestre | Meses | Hemisfério Sul |
|---|---|---|
| DJF | Dez, Jan, Fev | Verão — convecção, linhas de instabilidade |
| MAM | Mar, Abr, Mai | Outono — transição, primeiros sistemas frontais |
| JJA | Jun, Jul, Ago | Inverno — jatos polares, frentes frias intensas |
| SON | Set, Out, Nov | Primavera — instabilidade convectiva crescente |

**Padrão atual:** ativado por padrão (`stratify_seasons=True`). Para desabilitar:
`--no-stratify-seasons`.

---

## 11. Avaliação Orientada ao Deploy (`--eval-window`)

**Motivação:** o R² anual agrega toda a variação de 12 meses, mascarando meses
onde o modelo falha. Em produção, o sistema opera em lotes mensais ou quinzenais.

**Técnica:** R² calculado por janela temporal (mensal ou quinzenal) para os top-5
modelos do LazyPredict. Métricas reportadas:
- `R2_deploy_mean` — média do R² entre janelas
- `R2_deploy_std` — variância entre meses (estabilidade da performance)
- `R2_deploy_min` — pior mês (confiabilidade mínima garantida)

**Artefatos:** `rolling_r2/rolling_r2_c{N}.csv` e `plots/rolling_r2_c{N}.png`
(série temporal de R² por janela).

---

## 12. MLP PyTorch com Pinball Loss (`mlp_pytorch`)

**Motivação:** resolver a limitação da MSE (seção 5) com uma loss assimétrica
que penaliza mais sub-estimativas de eventos extremos.

**Pinball Loss** com quantil τ:
```
L(ŷ, y; τ) = mean[ τ · max(y − ŷ, 0) + (1−τ) · max(ŷ − y, 0) ]
```
Com τ=0.9: sub-estimativas custam 9× mais que sobre-estimativas.

**Regularização:** BatchNorm + Dropout por camada + ReduceLROnPlateau + early stopping
com restauração do melhor checkpoint.

**Resultado:** reduz o gap treino/validação em relação ao MLP sklearn, mas ainda
inferior ao Gradient Boosting via LazyPredict em R² absoluto. Vantagem na cauda:
Bias@P90 e RMSE@P90 melhores que MSE.

---

## 13. Plots e Diagnóstico

| Artefato | Conteúdo |
|---|---|
| `scatter_all_{slug}.pdf` | Scatter por modelo (treino real/sintético + validação) |
| `seasonal_scatter_{slug}.png` | Grid 2×2 por trimestre climático para o melhor modelo |
| `synth_comparison_{slug}.png` | ΔR² com vs sem sintéticos (verde=melhora, vermelho=piora) |
| `rolling_r2_{slug}.png` | R² por janela mensal/quinzenal — top-5 modelos |
| `top5_{slug}.png` | Bar plot com os 5 melhores modelos e métricas |

---

## Próximos Experimentos Sugeridos

1. **Stacking dos top-3 por trimestre:** usar os 3 melhores modelos LazyPredict
   de cada trimestre como meta-features de um modelo de nível 2.

2. **Features espaciais entre estações:** média/max do vento das outras estações
   do mesmo cluster no mesmo dia — sinal espacial coerente de sistemas sinóticos.

3. **Calibração dos quantis:** após escolher o modelo, calibrar o quantil τ da
   Pinball Loss por cluster usando validação cruzada temporal.

4. **Retreino periódico:** avaliar ganho de incluir 2023 (validação atual) no
   treino e validar em 2024 — aumenta volume de extremos observados.
