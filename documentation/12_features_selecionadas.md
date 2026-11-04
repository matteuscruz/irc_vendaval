# Features selecionadas: nomes e significado

As colunas que o modelo usa, com o nome por extenso de cada variável. São **58 colunas** de **12 famílias**,
selecionadas na validação do estudo de features (arm `sel__val12`, `config/features_maping.json`). O
**Corrigido ML** usa **49**: as mesmas, sem as 9 colunas `anor_*` (arm `sel__val12_sem_anor`), porque no
cluster 3 o `anor` funciona como identificador de estação e quebra a generalização para outras regiões.

Fonte dos nomes e das definições: dicionários `DICIONARIO_features_grupo{1,2,3}.md` e
`README_features_grupo4.md` (M3, `/home/publico/vendaval/data/out/`), que seguem o
`ERA5_INMET_Gust_BiasCorrection_ML_Technical_Spec.pdf` v1.0 (Tabelas 1–4, seções 7 e 8).

## As 12 variáveis selecionadas

| # | nome no código | nome verdadeiro | variável ERA5 de origem | grupo | colunas | no Corrigido ML |
|---|---|---|---|---|---|---|
| 1 | `gust10fg` | Rajada máxima de vento a 10 m (no ponto e nas horas anteriores) | `10fg`, paramId 49 | 1 | 5 | sim |
| 2 | `gust` | Rajada máxima de vento a 10 m na vizinhança (raios de 75 e 250 km) | `10fg`, paramId 49 | 1 | 6 | sim |
| 3 | `w10` | Velocidade do vento a 10 m, √(u10² + v10²) | `10u`/`10v`, paramId 165/166 | 1 | 10 | sim |
| 4 | `w100` | Velocidade do vento a 100 m, √(u100² + v100²) | `100u`/`100v`, paramId 228246/228247 | 1 | 1 | sim |
| 5 | `v10` | Componente norte-sul (meridional) do vento a 10 m | `10v`, paramId 166 | 1 | 1 | sim |
| 6 | `v100` | Componente norte-sul (meridional) do vento a 100 m | `100v`, paramId 228247 | 1 | 1 | sim |
| 7 | `hora_solar_sin` | Hora solar média (ciclo diurno), seno | hora UTC + longitude | 1 | 1 | sim |
| 8 | `cape` | Energia potencial convectiva disponível (CAPE) | `cape`, paramId 59 | 2 | 11 | sim |
| 9 | `mcpr` | Taxa média de precipitação convectiva | `mcpr`, paramId 235030 | 2 | 11 | sim |
| 10 | `grad_mslp_hpa_100km` | Gradiente horizontal da pressão ao nível do mar, no ponto | `msl`, paramId 151 | 3 | 1 | sim |
| 11 | `grad_mslp` | Gradiente horizontal da pressão ao nível do mar, mediana em 250 km | `msl`, paramId 151 | 3 | 1 | sim |
| 12 | `anor` | Ângulo da orografia subgrade (orientação do relevo) | `anor`, paramId 162, estático | 4 | 9 | **não** |
| | | | | | **58** | **49** |

As famílias `gust10fg` e `gust` são a mesma variável física (a rajada ERA5). A seleção as separa porque a
primeira reúne o valor no ponto e as defasagens, e a segunda as estatísticas de vizinhança. O mesmo vale para
`grad_mslp_hpa_100km` (no ponto) e `grad_mslp` (vizinhança).

## Como ler os nomes das colunas

Cada coluna é uma **variável ERA5** (às vezes calculada a partir de outras) mais, opcionalmente, um **operador**:

| sufixo | significado |
|---|---|
| (nenhum) | valor no ponto (célula ERA5 da estação), na hora do pico |
| `_ponto` | valor na célula ERA5 mais próxima (campos estáticos) |
| `_r75_<estat>`, `_r250_<estat>` | estatística sobre todas as células com centro a até 75 / 250 km (distância geodésica, haversine). `median` mediana, `max` máximo, `min` mínimo, `mean` média, `std` desvio-padrão, `fracpos` fração das células com valor > 0 |
| `_c75_250` | contraste de escala: mediana(75 km) − mediana(250 km) |
| `_l75_250` | contraste de escala: máximo(75 km) − mediana(250 km) |
| `_lag1h` | valor 1 hora antes (t−1) |
| `_max_prev3h` | máximo nas 3 horas anteriores (t−3 a t−1) |
| `_std_prev3h` | desvio-padrão nas 3 horas anteriores (t−3 a t−1) |
| `_delta3h` | variação em 3 horas: valor em t menos valor em t−3 |

**Tempo de referência (t):** cada linha é um par (estação ou célula, dia), tomado na **hora de maior rajada ERA5
do dia** (UTC). O alvo é a rajada máxima diária do INMET.

## Grupo 1: vento local e tempo (25 colunas)

| coluna | variável | operador | unidade |
|---|---|---|---|
| `gust10fg` | Rajada máxima ERA5 a 10 m desde o pós-processamento anterior (`10fg`, paramId 49) | ponto | m/s |
| `gust10fg_lag1h` | Rajada máxima ERA5 a 10 m | t−1 | m/s |
| `gust10fg_max_prev3h` | Rajada máxima ERA5 a 10 m | máximo de t−3 a t−1 | m/s |
| `gust10fg_std_prev3h` | Rajada máxima ERA5 a 10 m | desvio-padrão de t−3 a t−1 | m/s |
| `gust10fg_delta3h` | Rajada máxima ERA5 a 10 m | t − (t−3) | m/s |
| `gust_r75_median` | Rajada máxima ERA5 a 10 m | mediana em 75 km | m/s |
| `gust_r75_max` | Rajada máxima ERA5 a 10 m | máximo em 75 km | m/s |
| `gust_r75_std` | Rajada máxima ERA5 a 10 m | desvio-padrão em 75 km | m/s |
| `gust_r250_median` | Rajada máxima ERA5 a 10 m | mediana em 250 km | m/s |
| `gust_r250_max` | Rajada máxima ERA5 a 10 m | máximo em 250 km | m/s |
| `gust_r250_std` | Rajada máxima ERA5 a 10 m | desvio-padrão em 250 km | m/s |
| `v10` | Componente norte-sul (meridional) do vento a 10 m (`10v`, paramId 166) | ponto | m/s |
| `v100` | Componente norte-sul (meridional) do vento a 100 m (`100v`, paramId 228247) | ponto | m/s |
| `w10` | Velocidade do vento a 10 m, √(u10² + v10²) | ponto | m/s |
| `w10_lag1h` | Velocidade do vento a 10 m | t−1 | m/s |
| `w10_max_prev3h` | Velocidade do vento a 10 m | máximo de t−3 a t−1 | m/s |
| `w10_std_prev3h` | Velocidade do vento a 10 m | desvio-padrão de t−3 a t−1 | m/s |
| `w10_delta3h` | Velocidade do vento a 10 m | t − (t−3) | m/s |
| `w10_r75_median` | Velocidade do vento a 10 m | mediana em 75 km | m/s |
| `w10_r75_max` | Velocidade do vento a 10 m | máximo em 75 km | m/s |
| `w10_r75_std` | Velocidade do vento a 10 m | desvio-padrão em 75 km | m/s |
| `w10_r250_median` | Velocidade do vento a 10 m | mediana em 250 km | m/s |
| `w10_r250_std` | Velocidade do vento a 10 m | desvio-padrão em 250 km | m/s |
| `w100` | Velocidade do vento a 100 m, √(u100² + v100²) | ponto | m/s |
| `hora_solar_sin` | Hora solar média, h = (hora UTC + longitude/15) mod 24, codificada como sen(2π·h/24) | — | adimensional |

## Grupo 2: convectivo / mesoescala (22 colunas)

| coluna | variável | operador | unidade |
|---|---|---|---|
| `cape` | Energia potencial convectiva disponível, CAPE (`cape`, paramId 59) | ponto | J/kg |
| `cape_lag1h` | CAPE | t−1 | J/kg |
| `cape_max_prev3h` | CAPE | máximo de t−3 a t−1 | J/kg |
| `cape_std_prev3h` | CAPE | desvio-padrão de t−3 a t−1 | J/kg |
| `cape_delta3h` | CAPE | t − (t−3) | J/kg |
| `cape_r75_median` | CAPE | mediana em 75 km | J/kg |
| `cape_r75_max` | CAPE | máximo em 75 km | J/kg |
| `cape_r75_std` | CAPE | desvio-padrão em 75 km | J/kg |
| `cape_r250_median` | CAPE | mediana em 250 km | J/kg |
| `cape_c75_250` | CAPE | contraste: mediana(75) − mediana(250) | J/kg |
| `cape_l75_250` | CAPE | contraste: máximo(75) − mediana(250) | J/kg |
| `mcpr` | Taxa média de precipitação convectiva (`mcpr`, paramId 235030) | ponto | kg m⁻² s⁻¹ |
| `mcpr_lag1h` | Precipitação convectiva | t−1 | kg m⁻² s⁻¹ |
| `mcpr_max_prev3h` | Precipitação convectiva | máximo de t−3 a t−1 | kg m⁻² s⁻¹ |
| `mcpr_std_prev3h` | Precipitação convectiva | desvio-padrão de t−3 a t−1 | kg m⁻² s⁻¹ |
| `mcpr_delta3h` | Precipitação convectiva | t − (t−3) | kg m⁻² s⁻¹ |
| `mcpr_r75_max` | Precipitação convectiva | máximo em 75 km | kg m⁻² s⁻¹ |
| `mcpr_r75_std` | Precipitação convectiva | desvio-padrão em 75 km | kg m⁻² s⁻¹ |
| `mcpr_r75_fracpos` | Precipitação convectiva | fração das células em 75 km com chuva convectiva > 0 | 0–1 |
| `mcpr_r250_max` | Precipitação convectiva | máximo em 250 km | kg m⁻² s⁻¹ |
| `mcpr_r250_median` | Precipitação convectiva | mediana em 250 km | kg m⁻² s⁻¹ |
| `mcpr_l75_250` | Precipitação convectiva | contraste: máximo(75) − mediana(250) | kg m⁻² s⁻¹ |

## Grupo 3: sinótico (2 colunas)

| coluna | variável | operador | unidade |
|---|---|---|---|
| `grad_mslp_hpa_100km` | Magnitude do gradiente horizontal da pressão ao nível do mar (MSLP; `msl`, paramId 151) | ponto | hPa/100 km |
| `grad_mslp_r250_median` | Magnitude do gradiente horizontal da MSLP | mediana em 250 km | hPa/100 km |

## Grupo 4: orografia (9 colunas, **fora do Corrigido ML**)

| coluna | variável | operador | unidade |
|---|---|---|---|
| `anor_ponto` | Ângulo principal da orografia subgrade, orientação do relevo (`anor`, paramId 162, estático) | célula mais próxima | rad |
| `anor_r75_mean` | Ângulo da orografia subgrade | média em 75 km | rad |
| `anor_r75_std` | Ângulo da orografia subgrade | desvio-padrão em 75 km | rad |
| `anor_r75_min` | Ângulo da orografia subgrade | mínimo em 75 km | rad |
| `anor_r75_max` | Ângulo da orografia subgrade | máximo em 75 km | rad |
| `anor_r250_mean` | Ângulo da orografia subgrade | média em 250 km | rad |
| `anor_r250_std` | Ângulo da orografia subgrade | desvio-padrão em 250 km | rad |
| `anor_r250_min` | Ângulo da orografia subgrade | mínimo em 250 km | rad |
| `anor_r250_max` | Ângulo da orografia subgrade | máximo em 250 km | rad |

Por que o `anor` fica fora do Corrigido ML:
- é estático: no cluster 3 há só 9 valores por coluna, um por estação, e `anor_r250_min`/`max` têm um único
  valor. O modelo o usa como **código de estação**, não como física;
- é um **ângulo**: −π/2 e +π/2 são a mesma orientação, então média e desvio-padrão na vizinhança misturam
  direções iguais como se fossem opostas;
- em região nova os valores saem da faixa do treino, e o MLP do DJF extrapola até ~+12 m/s acima do INMET
  (cluster 12).

## Resumo

| grupo | famílias | colunas | no Corrigido ML |
|---|---|---|---|
| 1, vento local e tempo | `gust10fg`, `gust`, `w10`, `v10`, `v100`, `w100`, `hora_solar_sin` | 25 | sim |
| 2, convectivo | `cape`, `mcpr` | 22 | sim |
| 3, sinótico | `grad_mslp`, `grad_mslp_hpa_100km` | 2 | sim |
| 4, orografia | `anor` | 9 | **não** |
| **total** | **12** | **58** | **49** |

Para outro cluster, as colunas são recalculadas do ERA5 bruto por `src/feature_study/data/era5_features.py`,
com as mesmas regras (conferidas contra o cluster 3). `mcpr_r75_fracpos`, `mcpr_l75_250` e `grad_mslp_*`
seguem a definição acima, mas não puderam ser conferidas contra o cluster 3, porque o bruto de `mcpr` e
`msl` dele não está nesta máquina.
