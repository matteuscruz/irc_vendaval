# Pipeline `cluster_lstm`

Pipeline neural principal: uma LSTM de saída única por cluster (14 modelos),
treinada com as quatro estações do ano agrupadas. Configuração:
`config/experiment_cluster_lstm_modal.yaml` (schema v2).

## Arquitetura

Metodologia de artigo (WRF + estações NOAA ISD) adaptada ao ERA5 + INMET:

```text
Input(shape=(T, n_features))
    -> LSTM(96)            # último estado oculto
    -> Dropout(0.3)
    -> Dense(1, linear)    # "gust": rajada máxima diária em m/s (escalonada por scaler_y)
loss = Huber(delta=1.0), otimizador = Adam(lr=1e-3)
```

Builder: `src/models/cluster_lstm_builder.py::ClusterLSTMRegressorBuilder`.
Trainer: `src/pipeline/training/cluster_trainer.py` (EarlyStopping em
`val_loss`, semente por cluster). A saída é invertida por `scaler_y` e
limitada a [0, 80] m/s — não há mais razão × ERA5 nem troca de cabeça.

A antiga LSTM dual-head (`head_normal`/`head_extreme`, alvo em razão
INMET/ERA5) e a variante `cluster_tr_lstm` foram removidas. Artefatos delas
são recusados pela inferência (`LegacyLSTMArtifactError`) — é preciso
retreinar.

## Fontes de dados (`data.resolution`)

Interface comum em `src/pipeline/data/lstm_sources.py::build_lstm_batch`:

| resolução | construtor | janela | onde há dado |
|---|---|---|---|
| `daily` | `ClusterPreprocessor` | os `lookback` dias até o dia-alvo, `[D-L+1 .. D]` | 271 estações |
| `hourly` | `build_hourly_batch` | as 24 horas do dia D (`day_offset_hours` desloca o dia) | hoje só o cluster 3 (`test_cluster_3_hourly.nc`) |

A convenção de janela vive só em `src/pipeline/data/windowing.py`, usada
pelo treino e pelas duas inferências. A janela diária exige calendário
contínuo; dias sem linha no merge viram rótulo `out` e são purgados.

A fonte horária repete nas 24 horas o contexto diário do dia D
(sazonalidade). Não existe ERA5 horário em grade,
então modelos horários não geram grade corrigida.

## Interpolação grade → estação (`data.interp_method`)

`nearest` ou `bilinear` para ERA5-Basin e features novas (`nf_*`).
Bilinear usa os 4 vizinhos e cai no vizinho mais próximo onde algum é NaN.
Cada método tem o próprio cache: `era5_merged_cache.nc` (nearest) e
`era5_merged_cache_bilinear.nc`. Gere o bilinear com
`modal run src/modal/cluster_lazy.py --build-cache --interp-method bilinear`
(ou `scripts/build_synced_dataset.py --interp-method bilinear`).

## Split por blocos de mês (`data.split`)

- **Teste** = Jan/Abr/Jul/Out de **todos** os anos (uma amostra de cada
  estação do ano). **Treino** = os demais meses.
- **Validação** = blocos (ano, mês) sorteados dentro dos meses de treino,
  estratificados por mês (`val_fraction`, semente), sempre para todas as
  estações ao mesmo tempo. Os blocos sorteados ficam salvos no metadata.
- **Purga**: uma janela só entra se todos os seus dias têm o rótulo do
  dia-alvo. Com L=7, os primeiros 6 dias de cada bloco saem.
- **Sem features de observação**: o INMET é só o alvo. Os antigos
  `lag1/2/3_gust_obs`/`rolling7d_gust_obs` foram removidos (não existem na
  grade ERA5); as chaves `exclude_observation_features` e
  `blank_cross_split_observations` são rejeitadas pelo validador do YAML.
- Climatologia ERA5 (`era5_clim_wind`): **harmônica** (3 harmônicos),
  ajustada só nos dias de treino. Uma média por dia-do-ano deixaria os meses
  de teste sem valor.
- `scaler_x` e `scaler_y` são ajustados só em linhas de treino com alvo. Sem
  imputer: dias com feature ausente são descartados (`select_complete_rows`) e
  a purga remove as janelas que os tocam.

## Saída

- `fitted_models/best_model_c{id}.keras` (um por cluster);
- `fitted_models/dl_metadata.joblib` — schema v2: scalers,
  features, `WindowSpec`, split com `val_units`, climatologia,
  `interp_method`, `target_kind="absolute"`;
- `histories.json`, `results.csv` (por cluster × estação do ano, teste),
  `predictions/predictions_by_station.csv`, `run_meta.json`.

## Comparação com lazy/mlp

O teste da LSTM (Jan/Abr/Jul/Out de todos os anos) é outro recorte, diferente
do de lazy/mlp (2020–2024 inteiro). O seletor de vencedores
(`best_model_selector.build_winner_table`, `eval_set="common"`) recalcula as
métricas de todos no conjunto comum: Jan/Abr/Jul/Out de 2020–2024, só nas
datas presentes em todos os combos (`split="common_test"`).

## Quando usar

É o benchmark neural principal. Ele não vence em todos os clusters: o
vencedor por cluster × trimestre sai do seletor.
