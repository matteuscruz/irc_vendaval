"""Estudo de informação de features novas (cluster 3), isolado das pipelines.

Treina modelos COM e SEM cada feature `nf_*` sobre uma amostra representativa do
treino e mede quais acrescentam informação. Não altera nenhuma pipeline de
treino; ver `documentation/09_estudo_de_features.md`.

Estágios: `prepare` (dados → amostras), `fit` (um lote de arms por trimestre e
réplica), `aggregate` (efeitos pareados, IC por blocos, ranking).
"""
