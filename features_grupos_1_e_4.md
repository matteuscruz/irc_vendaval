# Features dos Grupos 1 e 4

Referência: `ERA5_INMET_Gust_BiasCorrection_ML_Technical_Spec.pdf`, v1.0 de 26/08/2026, Tabelas 1 e 4.

A coluna "Engenheirada" diz se a linha é uma variável baixada direto do ERA5 (Não) ou um valor calculado a partir de outras (Sim). Quando é calculada, a coluna "Entradas" lista de quais variáveis ela sai.

## Grupo 1: vento local, camada limite, tempo e coordenadas

| Feature | Engenheirada | Entradas | Descrição | Unidade | Prioridade |
|---|---|---|---|---|---|
| Rajada ERA5 | Não | `10fg` (paramId 49) | Rajada máxima a 10 m desde o pós-processamento anterior. É o ponto de partida da correção | m/s | Essential |
| u10, v10 | Não | `10u` (165), `10v` (166) | Componentes leste-oeste e norte-sul do vento a 10 m | m/s | Essential |
| W10 | **Sim** | u10, v10 | Velocidade do vento a 10 m: raiz de (u10² + v10²) | m/s | Essential |
| u100, v100 | Não | `100u` (228246), `100v` (228247) | Componentes do vento a 100 m | m/s | High |
| W100 | **Sim** | u100, v100 | Velocidade do vento a 100 m, mesma fórmula do W10 | m/s | High |
| BWD 10-100 m | **Sim** | u10, v10, u100, v100 | Cisalhamento na camada de 10 a 100 m: módulo da diferença vetorial, sem dividir pela espessura. Mede o acoplamento vertical | m/s | High |
| Velocidade de fricção (u*) | Não | `zust` (228003) | Turbulência na camada superficial. A própria ECMWF avisa que o ERA5 pode subestimar essa variável, então ela entra como candidata e o peso dela precisa ser medido | m/s | High |
| Altura da camada limite | Não | `blh` (159) | Espessura da camada de mistura | m | High |
| Dissipação média na camada limite | Não | `mbld` (235032) | Média sobre a hora anterior | W/m² | High |
| Fluxo de calor sensível médio | Não | `msshf` (235033) | Média sobre a hora anterior. Sinal de mistura convectiva seca | W/m² | High |
| T2m | Não | `2t` (167) | Temperatura a 2 m | K | Medium |
| Td 2m | Não | `2d` (168) | Ponto de orvalho a 2 m | K | Medium |
| Depressão do ponto de orvalho | **Sim** | T2m, Td 2m | T2m menos Td 2m, nas mesmas unidades. Medida compacta de secura perto da superfície | K | High |
| Pressão de superfície | Não | `sp` (134) | Usada também para cortar os níveis de pressão abaixo do solo | Pa | Essential |
| Latitude e longitude | Não | coordenadas do ponto | Metadado do ponto de previsão, não é variável física do ERA5 | graus | Medium |
| Hora solar média, sin e cos | **Sim** | hora UTC + longitude | h_solar = (h_utc + lon/15) mod 24, codificada em seno e cosseno de 2π·h/24. Captura o ciclo diurno da camada limite sem depender de fuso horário | adimensional | High |
| Dia do ano, sin e cos | **Sim** | data | Seno e cosseno de 2π·DOY/365,2425 | adimensional | High |

## Grupo 4: geografia, superfície e orografia do ERA5

Não há modelo de elevação externo na versão 1. Tudo sai do próprio ERA5.

| Feature | Engenheirada | Entradas | Descrição | Unidade | Prioridade |
|---|---|---|---|---|---|
| Altura da superfície | **Sim** | `z` (129), geopotencial de superfície | Conversão de geopotencial para altura: z = Φ·Re / (g·Re − Φ). A aproximação Φ/g dá quase o mesmo em terreno comum, mas a recomendação é usar a fórmula completa | m | Essential |
| sdor | Não | `sdor` (160) | Desvio-padrão da orografia subgrade dentro da célula, escalas de ~5 km até a grade | m | High |
| sdfor | Não | `sdfor` (74) | Mesma ideia, versão filtrada, escalas de ~3 a 22 km | m | High |
| isor | Não | `isor` (161) | Anisotropia do relevo subgrade: separa serra alongada de morro espalhado | adimensional | Medium |
| anor | Não | `anor` (162) | Ângulo principal do relevo subgrade alongado | rad | Medium |
| slor | Não | `slor` (163) | Declividade subgrade. Não é grau: plano vale 0 e encosta de 45° vale 0,5 | adimensional | High |
| Máscara terra-mar | Não | `lsm` (172) | Fração de terra da célula | 0 a 1 | High |
| Distância à costa | **Sim** | `lsm` + coordenadas do ponto | Célula é terra quando LSM ≥ 0,5 e mar quando é menor. Célula de costa é a que tem terra e mar na vizinhança de 8. A feature é a distância geodésica do ponto ao centro da célula de costa mais próxima. O limiar de 0,5 fica em configuração, para poder ser testado | km | High |
| Rugosidade de superfície | Não | `fsr` (244) | Comprimento de rugosidade aerodinâmica z0. Sobre terra depende de vegetação e neve, sobre oceano depende das ondas. É a única do Grupo 4 que varia no tempo, as outras são invariantes e se baixa uma vez só | m | High |

## Operadores espaciais e temporais

Os dois blocos acima são as features base. Por cima delas o spec manda aplicar operadores de vizinhança e de defasagem, mas **só em algumas variáveis**, não em todas.

Do Grupo 1, entram nos operadores espaciais (raios de 75 km e 250 km, distância geodésica): rajada ERA5 (mediana, máximo, desvio nos dois raios) e W10 (mediana, máximo e desvio em 75 km; mediana e desvio em 250 km).

Do Grupo 1, entram nas defasagens de 1 a 3 h: rajada ERA5, W10 e altura da camada limite. Para a rajada, os descritores pedidos são o valor em t−1, o máximo de t−1 a t−3 e o desvio-padrão dessas três horas. Para a camada limite, a diferença contra t−1 e contra t−3.

O Grupo 4 não recebe nem operador espacial nem defasagem: os campos são invariantes no tempo, então uma diferença temporal sobre eles seria zero por construção. A exceção conceitual é o `fsr`, que varia no tempo, mas o spec não o coloca na lista de defasagem.

## O que já está gerado hoje

As features estáticas do Grupo 4 estão calculadas em `features_grupo4_estatico_cluster3.csv` e `.nc`: 21 estações INMET dentro do polígono do Cluster 3, 112 colunas de feature, sem nenhum valor faltante. O padrão de nome é `<campo>_ponto` para o valor da célula mais próxima e `<campo>_r{raio}_{mean,std,min,max,n}` para a estatística de vizinhança.

Duas ressalvas sobre esse arquivo:

1. Os raios usados nele são 25, 50 e 75 km, escolhidos a partir da grade do ERA5 (0,25° dá cerca de 25 km nesta latitude). O spec recomenda **75 km e 250 km**. Se o desenho oficial for o do spec, é trocar a lista de raios e rodar de novo.
2. A distância à costa da Tabela 4 ainda não está nesse arquivo. As outras oito linhas do Grupo 4 estão.

Do Grupo 1, todas as variáveis brutas já estão em disco para 2000 a 2025, ano a ano. As features calculadas (W10, W100, BWD, depressão do ponto de orvalho, hora solar e dia do ano) ainda não foram geradas.
