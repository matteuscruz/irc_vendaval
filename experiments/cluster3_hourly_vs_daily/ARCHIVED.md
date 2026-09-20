# Experimento arquivado — diário vs. horário (cluster 3)

Este experimento isolado foi feito para a LSTM dual-head: alvo em razão
INMET/ERA5, janela de 168 h que excluía o dia-alvo, e split por anos. Ele
**não roda mais** contra o código atual. `run_experiment.py` chama
`ClusterTrainer`/`ClusterMetricsEvaluator` com assinaturas antigas, e
`hourly_windower.py` monta um `ClusterDataBatch` com campos que foram
removidos (`era5_*`, `x_static_*`).

A parte horária foi promovida para a pipeline de produção:

- `src/pipeline/data/hourly_builder.py::build_hourly_batch` — janela = as
  24 h do dia-alvo, contexto diário repetido nas 24 h, split por blocos de
  mês;
- selecionada com `data.resolution: hourly` em
  `config/experiment_cluster_lstm_modal.yaml`.

Os arquivos ficam aqui como registro do resultado anterior (horário pior que
diário no cluster 3).
