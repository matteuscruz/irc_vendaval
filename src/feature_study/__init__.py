"""Estudo de informação de features por grupo (cluster 3), isolado das pipelines.

Treina modelos COM e SEM cada grupo de variáveis ERA5 sobre o treino completo e mede,
com bootstrap pareado em blocos, quais grupos acrescentam informação. Não altera nenhuma
pipeline de treino; ver `documentation/09_estudo_de_features.md`.

  core/         o estudo: config, arms, prepare, worker, analysis (+ artifacts do campeão)
  data/         entrada: groups_source (4 parquets), base_only (BASE antiga), purge (gap temporal)
  selection/    seleção de variáveis e arms `sel__*`
  diagnostics/  análises sobre os resultados: robustness, shap_groups, group_importance

Estágios: `prepare` (dados → parquet), `fit` (um lote de arms por trimestre e seed),
`screen` (top-5 modelos), `aggregate` (efeitos pareados, IC por blocos, ranking).
"""
