# Inferência e Artefatos

Depois do treino, o sistema produz artefatos diferentes conforme a pipeline.

## Inferência espacial

- `src/inference/spatial_correction.py`: usa os melhores modelos salvos pelo
  `cluster_lazy` para corrigir estações e interpolar para a grade ERA5.
- `src/inference/spatial_correction_dl.py`: versão DL para os modelos Keras do
  `cluster_lstm`.
- `corrected_grid`: gera a grade final corrigida em NetCDF.

## Artefatos comuns

- `results.csv` / `predictions.csv` consolidados;
- modelos salvos por cluster;
- métricas de validação e teste;
- gráficos por cluster e por janela de deploy.

## Leitura recomendada

- Use o README para comandos.
- Use esta pasta para entender o fluxo da etapa e o que cada saída significa.