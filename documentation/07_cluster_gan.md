# Pipeline `cluster_gan` (ExGAN & Augmentação Extrema)

Esta pipeline implementa a metodologia **ExGAN** (*Extreme Generative Adversarial Network*) adaptada para dados meteorológicos tabulares e sequenciais, com foco em gerar amostras sintéticas realistas de **rajadas severas de vento (vendavais)**.

---

## 1. Motivação e Desafios

Modelos supervisionados tradicionais de Machine Learning e Deep Learning tendem a falhar na previsão de eventos raros e extremos:

1. **Escassez de Cauda**: Rajadas destrutivas ocupam o topo da distribuição de probabilidade ($P \ge 90\%$), representando uma fração mínima do histórico observacional.
2. **Subestimação Sistemática (Tail Regression)**: Funções de perda baseadas em MSE/MAE forçam o modelo a aprender a média climatológica, "achatando" os picos extremos.
3. **Limitação de GANs Clássicas**: GANs convencionais sofrem de *mode collapse* e ignoram regiões de densidade amostral muito baixa, gerando amostras concentradas apenas no regime padrão.

Para solucionar esses gargalos, este projeto utiliza a teoria de **Extreme Value Theory (EVT)** acoplada a uma **Conditional WGAN-GP** com **Distribution Shifting**.

---

## 2. Fundamentação Teórica: EVT & Generalized Pareto Distribution (GPD)

Com base no **Teorema de Pickands-Balkema-de Haan**, as excedências de uma variável contínua acima de um limiar suficientemente alto $u$ convergem assintoticamente para uma **Distribuição de Pareto Generalizada (GPD)**:

$$P(Y - u \le y \mid Y > u) \approx G(y; \xi, \sigma) = 1 - \left( 1 + \frac{\xi y}{\sigma} \right)^{-1/\xi}$$

onde:
- $u$: Limiar de extremo (definido pelo quantil `--extreme-percentile`, default $P90$);
- $\xi$ (*shape* / forma): Determina a espessura da cauda (se $\xi > 0$, cauda pesada do tipo Fréchet);
- $\sigma$ (*scale* / escala): Escala de dispersão das excedências.

### 2.1. Amostragem de Alvos Extremos via GPD
Durante o treino e inferência, alvos extremos sintéticos $e_{\text{target}}$ são amostrados analiticamente via inversa da CDF da GPD para $U \sim \text{Uniform}(0, 1)$:

$$e_{\text{target}} = \begin{cases} 
u + \frac{\sigma}{\xi} \left( (1 - U)^{-\xi} - 1 \right), & \text{se } \xi \neq 0 \\
u - \sigma \ln(1 - U), & \text{se } \xi = 0 
\end{cases}$$

### 2.2. Normalização da Condição de Extremo
O valor extremo amostrado é normalizado para compor o vetor de condicionamento da rede neural:

$$e_{\text{norm}} = \max\left(0, \frac{e_{\text{target}} - u}{\max(\sigma, 10^{-6})}\right)$$

---

## 3. Algoritmo de *Distribution Shifting* (Iterativo em $k$ Passos)

Em dados meteorológicos reais, mais de $90\%$ das amostras possuem $e_{\text{norm}} = 0$. Se a rede fosse treinada diretamente nesse conjunto, o gerador aprenderia a ignorar $e_{\text{norm}}$, colapsando para valores constantes médios.

Para evitar isso, executa-se o **Distribution Shifting** antes do treinamento final:

```text
[Dataset Original Real]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Iteração de Shifting (k_shift = 3 rounds):                 │
│  1. Ordena o conjunto e remove os c% menos extremos (c=0.3);│
│  2. Treina uma GAN auxiliar no subconjunto restante;        │
│  3. Gera n_gen candidatos com e_norm = 1.0;                 │
│  4. Filtra e concatena os candidatos mais extremos;         │
│  5. O pool resultante fica progressivamente mais extremo.   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
[Dataset Deslocado (Shifted Dataset)]
       │
       ▼
[Treinamento da Conditional WGAN-GP Final]
```

---

## 4. Arquitetura da Rede Neural (Conditional WGAN-GP)

A arquitetura é construída com **TensorFlow / Keras** e utiliza a formulação de **Wasserstein GAN com Gradient Penalty (WGAN-GP)**.

### 4.1. Vetor de Condição ($c$)
O vetor de condição concatena informações geográficas, sazonais e de severidade:

$$c = \big[ \underbrace{\mathbf{c}_{\text{cluster}}}_{\text{One-Hot }(N_{\text{clusters}})}, \, \underbrace{\mathbf{s}_{\text{season}}}_{\text{One-Hot }(4)\text{ [Opcional]}}, \, \underbrace{e_{\text{norm}}}_{1\text{ valor escalar}} \big]$$

### 4.2. Topologia do Gerador ($G$) e Crítico ($D$)

> **Nota:** O diagrama abaixo refere-se ao **`TabularExtremeGANAugmenter`** (alvo-only, usado pelo `cluster_gan.py`). O `ExGANAugmenter` (usado pelo `cluster_lstm.py`) possui topologia diferente: `hidden_units=256`, e o gerador produz `(lookback × n_features + 1)` saídas (sequência temporal completa + alvo), não apenas o escalar.

```text
               GERADOR (G) — Tabular                       CRÍTICO (D) — Tabular
    ┌───────────────────────────────┐          ┌───────────────────────────────┐
    │ Ruído z ~ N(0, I_32)  +  Cond c│          │  Alvo y (1)  +  Condição c    │
    └───────────────┬───────────────┘          └───────────────┬───────────────┘
                    │                                          │
                    ▼                                          ▼
       Dense(128, activation='elu')               Dense(128, activation='elu')
                    │                                          │
                    ▼                                          ▼
       Dense(128, activation='elu')               Dense(64, activation='elu')
                    │                                          │
                    ▼                                          ▼
       Dense(1, linear) -> y_synth                 Dense(1, linear) -> Score WGAN
```

### 4.3. Funções de Perda e Otimização

1. **Perda do Crítico ($\mathcal{L}_D$)**:
   $$\mathcal{L}_D = \mathbb{E}_{\tilde{y} \sim \mathbb{P}_g}[D(\tilde{y}, c_{\text{fake}})] - \mathbb{E}_{y \sim \mathbb{P}_r}[D(y, c_{\text{real}})] + \lambda_{\text{GP}} \cdot \mathbb{E}_{\hat{y}}\left[ \left( \|\nabla_{\hat{y}} D(\hat{y}, c_{\text{real}})\|_2 - 1 \right)^2 \right]$$
   *(com $\lambda_{\text{GP}} = 10.0$ e $n_{\text{critic}} = 5$ iterações do crítico por passo do gerador)*

2. **Perda do Gerador com Penalidade Extrema ($\mathcal{L}_G$)**:
   $$\mathcal{L}_G = -\mathbb{E}_{z, c_g}[D(G(z, c_g), c_{\text{real}})] + \lambda_y \cdot \mathcal{L}_{\text{ext}}$$
   onde $\mathcal{L}_{\text{ext}}$ é a perda percentual relativa do extremo requisitado:
   $$\mathcal{L}_{\text{ext}} = \mathbb{E} \left[ \frac{|G(z, c_g) - e_{\text{target}}|}{|e_{\text{target}}| + 10^{-8}} \right]$$

   > **Valores de $\lambda_y$ por implementação:**
   > - `TabularExtremeGANAugmenter` (cluster_gan): $\lambda_y = 5.0$ (penalidade forte — alvo escalar é a única saída).
   > - `ExGANAugmenter` (cluster_lstm): $\lambda_y = 0.5$ (penalidade moderada — o gerador também produz features temporais).

---

## 5. Pareamento Físico de Features (*Nearest-Neighbor Assignment*)

### Por que gerar apenas o alvo escalar?
Redes gerativas que tentam modelar simultaneamente 15+ variáveis meteorológicas tabulares (temperatura, pressão ao nível do mar, umidade, gradientes de vento $u/v$, cisalhamento) frequentemente geram combinações termodinamicamente impossíveis (ex.: gradientes térmicos inconsistentes com a pressão).

### Metodologia Híbrida:
1. A GAN gera com precisão estatística o valor escalar do alvo extremo $\hat{y}_{\text{synth}}$ condicionado a cada cluster e estação climática;
2. A função `_nearest_neighbor_assign(y_real, x_real, y_synth)` localiza no histórico observacional real a amostra de maior similaridade de intensidade ($y_{\text{real}} \approx y_{\text{synth}}$);
3. O vetor completo de covariáveis meteorológicas do evento real mais próximo é associado à amostra sintética, garantindo **$100\%$ de coerência física** nos dados tabulares.

---

## 6. Implementações Disponíveis no Projeto

| Classe / Módulo | Pipeline | Tipo de Dado | Defaults Divergentes | Características Principais |
| :--- | :--- | :--- | :--- | :--- |
| **`TabularExtremeGANAugmenter`**<br>`src/pipeline/augmentation/tabular_gan_augmenter.py` | `cluster_gan.py` | Tabular (2D) | `hidden_units=128`, `epochs=300`, `lambda_y=5.0` | Alvo-only + Nearest-Neighbor assignment. Produz `synthetic_augment.csv` para consumo no `cluster_lazy` e `cluster_mlp`. |
| **`ExGANAugmenter`**<br>`src/pipeline/augmentation/extreme_gan_augmenter.py` | `cluster_lstm.py` | Sequencial (3D)<br>`(batch, lookback, features)` | `hidden_units=256`, `epochs=200`, `lambda_y=0.5` | Gera sequência temporal completa + alvo. Correção de viés QDM (`ERA5BiasCorrector`). Suporta `tau` para GPD inverse targeting. |

---

## 7. Principais Parâmetros e Modos de Execução

| Parâmetro / Flag | Padrão | Descrição |
| :--- | :---: | :--- |
| `--epochs` | `300` | Número de épocas de treinamento da GAN final. |
| `--extreme-percentile` | `90.0` | Percentil que define o limiar $u$ de início da cauda da GPD. |
| `--n-per-cluster-ratio` | `0.3` | Fração proporcional de dados sintéticos em relação ao total de treino de cada cluster. |
| `--n-per-cluster` | `None` | Número absoluto fixo de amostras sintéticas por cluster (sobrepõe a razão). |
| `--include-season` | `False` | Condiciona a geração também pelo trimestre climático (`DJF`, `MAM`, `JJA`, `SON`). |
| `--cluster-merge` | `None` | Permite fundir clusters geograficamente correlacionados antes do treino. |

---

## 8. Artefatos e Consumo Downstream

A execução do pipeline (`modal run src/modal/cluster_gan.py` ou `python -m src.pipelines.cluster_gan`) gera os seguintes artefatos em `artifacts/gan_clusters/<exp_name>/`:

1. **`synthetic_augment.csv`**:
   - Tabela consolidada contendo todas as variáveis base (`wind_mag_max`, pressões, temperaturas, etc.), as flags `cluster_id`, `is_extreme=True`, `season` e o alvo sintético gerado em `TARGET_VAR`.
2. **`run_meta.json`**:
   - Metadados completos com os parâmetros ajustados da GPD ($\xi, \sigma, u$), estatísticas descritivas reais vs sintéticas e percentis comparativos.
3. **`synthetic_vs_real_distribution.png`**:
   - Gráfico de densidade sobrepondo as distribuições real e sintética com linhas verticais em P90, P95 e P99.

### Como Utilizar nos Pipelines de Downstream:
```bash
# Treinar pipeline de screening Lazy com os sintéticos da GAN
modal run src/modal/cluster_lazy.py --synthetic-csv artifacts/gan_clusters/exp1/synthetic_augment.csv

# Treinar pipeline MLP com extreme-weighting + augmentação GAN
modal run src/modal/cluster_mlp.py --synthetic-csv artifacts/gan_clusters/exp1/synthetic_augment.csv

# Treinar pipeline LSTM com augmentação ExGAN embutida
modal run src/modal/cluster_lstm.py --augmentation-method extreme_gan
```