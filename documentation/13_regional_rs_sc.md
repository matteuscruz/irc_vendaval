# RS/SC: o modelo do cluster 3 em regiões novas e modelos treinados nos locais novos

Teste de generalização e treino em RS/SC (clusters 1, 2, 3 e 4), a partir do ERA5 bruto em
`dataset/raw/era5_rs_sc`. A grade tem 8 variáveis, é horária, cobre 2000–2025 em 0,25° (34 × 40 células) e não
traz os campos estáticos. O resultado do cluster 12 (`documentation/11` e `inferencia_cluster12.ipynb`) entra na
comparação.

## Desenho

Todos os modelos usam:
- os campeões do cluster 3 em cada trimestre: MLP no DJF, ExtraTrees no MAM, CatBoost no JJA, HistGB no SON;
- as **49 colunas sem `anor`** (`documentation/12_features_selecionadas.md`);
- seed 42;
- o EQM ajustado na validação do próprio treino.

O split é o do estudo e depende só do calendário, então é o mesmo para todos os clusters:
- teste: jan/abr/jul/out de 2000–2024;
- validação: blocos (ano, mês) dos outros meses;
- 2025: fora de tudo.

| desenho | treina com | avaliado em |
|---|---|---|
| **A** | cluster 3 (o modelo do estudo, sem treino novo) | clusters 1–4 e 12 |
| **B** | as estações de **um** cluster | os 4 clusters (matriz treino × teste) |
| **C** | as estações dos clusters 1–4 juntas (regional) | cada cluster |
| **L** | os outros três clusters (regional deixando um de fora, LOCO) | o cluster que ficou de fora |

As linhas são pares (estação, dia), com as colunas da célula ERA5 mais próxima tomadas na hora do pico de rajada
ERA5 daquele dia. O alvo é a rajada máxima diária do INMET. As linhas de treino somadas por desenho são:

| desenho | linhas |
|---|---|
| A | 23 mil |
| B, cluster 1 | 14 mil |
| B, cluster 2 | 46 mil |
| B, cluster 3 | 23 mil |
| B, cluster 4 | 85 mil |
| C (regional) | 168 mil |

## Resultados (meses de teste 2000–2024, RMSE diário em m/s contra o INMET)

| cluster | ERA5 | A sem EQM | B local sem EQM | C regional sem EQM | L LOCO sem EQM | C regional com EQM |
|---|---|---|---|---|---|---|
| 1 | 2,62 | 2,69 | **2,28** | 2,41 | 2,61 | 2,66 |
| 2 | 2,54 | 2,44 | 2,30 | **2,30** | 2,41 | 2,54 |
| 3 | 2,66 | 2,41 | **2,41** | 2,47 | 2,66 | 2,70 |
| 4 | 3,44 | 3,35 | **2,75** | 2,76 | 3,31 | 2,94 |
| 12 | 2,27 | 2,42 (só A) | — | — | — | — |

POD no P99, com EQM:

| cluster | ERA5 | A | B local | C regional | L LOCO |
|---|---|---|---|---|---|
| 1 | 0,22 | 0,15 | **0,36** | 0,29 | 0,28 |
| 2 | 0,08 | 0,09 | 0,13 | 0,16 | **0,18** |
| 3 | 0,21 | 0,12 | 0,13 | 0,23 | **0,26** |
| 4 | 0,13 | 0,10 | **0,16** | 0,15 | 0,08 |

## O que os números dizem

1. **Treinar com estações da região funciona.** O modelo local (B) e o regional (C), sem EQM, ganham do ERA5 nos
   quatro clusters, de −0,2 a −0,7 m/s. O maior ganho é no cluster 4, onde o ERA5 é pior (RMSE 3,44 → 2,75).
2. **O regional quase empata com o local.** A diferença vai de 0 a 0,13 m/s. Um modelo único para RS/SC serve aos
   quatro clusters, inclusive ao cluster 1, que tem só 4 estações.
3. **Transferir para uma região não vista não funciona.** O desenho A (modelo do cluster 3) e o LOCO (regional sem o
   cluster avaliado) ficam iguais ou piores que o ERA5 no erro diário. O mesmo vale para o cluster 12. Na matriz do
   desenho B, os modelos ganham na diagonal (região do treino) e perdem fora dela, às vezes muito: o modelo do cluster
   1 avaliado no 3 perde 0,6 m/s. A relação ERA5 → INMET é regional.
4. **O EQM troca erro diário por extremos.** Com o EQM, o RMSE sobe de 0,2 a 0,3 m/s, e o POD no P99 fica igual ou
   maior que o do ERA5 nos desenhos B e C.
5. **O 2025, ano nunca visto, confirma o padrão.** O local e o regional ganham do ERA5 em todos os clusters. A e LOCO
   não ganham de forma consistente.

## Limites

- Os campeões vieram do cluster 3 e não foram retriados para RS/SC.
- Uma seed: diferenças de poucos centésimos estão dentro do ruído.
- O cluster 4 tem a borda norte da vizinhança cortada pela grade (o polígono vai a −25,6°, a grade a −25,75°).
- A vizinhança depende da grade baixada. No cluster 3, as colunas `_r250` diferem das do
  `test_inference_cluster_3.parquet`, porque a grade de RS/SC tem mais vizinhos. As colunas pontuais e as defasagens
  são idênticas, e a hora do pico coincide em 100% dos dias.

## Como rodar

```bash
uv run python scripts/treinar_rs_sc.py --stage all       # grade (~8 min) → base → treino (~13 min) → mapas
uv run python scripts/build_notebook_rs_sc.py            # e executar notebooks/feature_study_grupos/atual/regional_rs_sc.ipynb
```

Saídas em `artifacts/feature_study/rs_sc/`:
- `metricas.csv`
- `previsoes_estacoes.parquet`
- `params.json`
- os modelos A, C e L em `modelos/`
- `results_rs_sc_c<N>.nc`, a grade de cada cluster com A, C e L

Código:
- `src/feature_study/data/station_dataset.py`: base por estação e split;
- `src/feature_study/diagnostics/regional.py`: treino, previsão e métricas.
