# IRC Vendaval

Correção de viés das rajadas de vento do ERA5 contra as estações INMET, com um modelo por cluster
climático. Este README só diz **como executar**. O que o projeto faz, os dados, as pipelines, o split
e os resultados estão em [`documentation/`](documentation/DOCUMENTATION.md).

## 1. Pré-requisitos

- Python ≥ 3.10 e [uv](https://docs.astral.sh/uv/)
- Conta no [Modal](https://modal.com) (só para os passos marcados *Modal*)
- Os dados em `dataset/` (passo 3)

## 2. Instalar

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # se ainda não tiver o uv
uv sync --extra dev                                 # dependências + pytest e modal
uv pip install nbformat nbconvert ipykernel         # só para gerar/executar os notebooks (não estão no pyproject)
uv run pytest test/ -q                              # confere a instalação
```

Para o Modal: `modal setup` (autentica uma vez). Se você tem mais de um perfil, escolha o que tem os
volumes na própria linha do comando: `MODAL_PROFILE=<perfil> modal run ...`.

## 3. Colocar os dados

```
dataset/
├── raw/
│   ├── INMET_Stratified.nc                    alvo observado (INMET)
│   ├── ERA5_Stratified.nc                     ERA5 co-localizado
│   ├── ERA5_Features_Basin_2000_2026.nc       grade diária da bacia
│   ├── cluster_3_hourly.nc                    base horária (cluster 3)
│   ├── new_features/                          features ERA5 novas (grade)
│   └── feature_study_cluster_3/               4 parquets por grupo (estudo de features)
└── shp/shp_vento.shp                          polígonos dos 14 clusters
```

Para copiar um arquivo de outra máquina: `./scripts/fetch_from_aws.sh <user@host> <caminho/remoto>`.
No Modal, `dataset/` sobe sozinho para o volume `irc-vendaval-dataset` na primeira execução de
qualquer pipeline. Detalhes de cada arquivo: [`documentation/02`](documentation/02_dados_e_preprocessamento.md).

## 4. Treinar as pipelines

Todas partem de `main.py` (`python main.py --help`) e usam o mesmo split por blocos de mês.

```bash
# LazyPredict por cluster (benchmark e escolha do melhor modelo)
uv run python main.py cluster_lazy --output-dir artifacts/lazy_clusters \
    --n-neighbor-clusters 1 --eval-window monthly

# MLP com extreme weighting
uv run python main.py cluster_mlp --output-dir artifacts/mlp_clusters

# LSTM v2 (precisa do cache bilinear, gerado uma vez no Modal)
modal run src/modal/cluster_lazy.py --build-cache --interp-method bilinear
uv run python main.py cluster_lstm --config config/experiment_cluster_lstm_modal.yaml
```

Na nuvem (Modal), uma por vez; `--only-download` só baixa o resultado de uma run anterior:

```bash
modal run --detach src/modal/cluster_lazy.py --exp-name <nome>
modal run src/modal/cluster_mlp.py --exp-name <nome>
modal run src/modal/cluster_lstm.py --config experiment_cluster_lstm_modal.yaml --exp-name <nome>
modal run src/modal/cluster_lazy.py --only-download         # (ou cluster_mlp / cluster_lstm)
```

Matriz de ablation (original / newfeatures / basin), uma pipeline por script:

```bash
uv run python scripts/run_ablation_lazy.py     # também run_ablation.py (mlp) e run_ablation_lstm.py
uv run python scripts/compare_ablation_all.py  # compara as três
```

## 5. Gerar a grade corrigida

Depois de treinar, escolhe o melhor modelo por cluster × trimestre e gera os NetCDF:

```bash
uv run python main.py corrected_grid --winners-only   # só mostra os vencedores
uv run python main.py corrected_grid                  # gera a grade (--help para as opções)
modal run src/modal/corrected_grid.py                 # mesmo passo na nuvem
```

## 6. Estudo de features por grupo

**Local** (sequencial, idempotente; `plan` mostra o que vai rodar e quanto custa):

```bash
uv run python scripts/run_feature_study_local.py plan
uv run python scripts/run_feature_study_local.py run --stage all --seeds 42
uv run python scripts/run_feature_study_local.py status
```

**Modal** (as 5 seeds do estudo de referência; `--study` isola o resultado no volume):

```bash
modal run src/modal/feature_study.py --stage prepare --study <nome> --arm-sets anchors,groups,controls
modal run src/modal/feature_study.py --stage fit --arms base --models all --triage --study <nome>
modal run src/modal/feature_study.py --stage screen --study <nome>            # congela os 5 melhores modelos
modal run src/modal/feature_study.py --stage fit --models top5 --study <nome>
modal run src/modal/feature_study.py --stage aggregate --study <nome>         # baixa o resumo
```

Depois: `uv run python scripts/analise_robustez.py --study <nome>` (análises de robustez, sem custo de
nuvem). Selecionar variáveis e treinar o conjunto reduzido:

```bash
uv run python scripts/congelar_selecao_features.py
modal run src/modal/feature_study.py --stage add-arms --study <nome>
modal run src/modal/feature_study.py --stage fit --arms sel__val12 --models top5 --study <nome>
```

**LSTM no mesmo estudo** (mesmos arms e linhas; fora do `all`; ver `documentation/09`):

```bash
uv run python scripts/run_feature_study_local.py run --stage lstm --seeds 42            # ~18 min
uv run python scripts/run_feature_study_local.py run --stage aggregate-lstm --seeds 42
# Modal (GPU): piloto com 2 arms antes das 5 seeds
modal run src/modal/feature_study.py --stage fit-lstm --study <nome> --seeds 42 --arms base,full
modal run src/modal/feature_study.py --stage aggregate --lstm --study <nome>
```

Os resultados são lidos pelos notebooks de `notebooks/feature_study_grupos/atual/`. Para regenerá-los
a partir dos scripts:

```bash
uv run python scripts/build_notebook_resultados_finais.py
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/feature_study_grupos/atual/resultados_finais_grupos.ipynb
```

Passos de nuvem são pagos: confira o `plan` antes e use `MODAL_PROFILE` como no passo 2.

## 7. Onde ler mais

| Para saber… | Leia |
|---|---|
| o que o projeto faz e por quê | [`documentation/01`](documentation/01_visao_geral.md) e [`10`](documentation/10_contexto_e_historico.md) |
| dados, features e o split por blocos de mês | [`documentation/02`](documentation/02_dados_e_preprocessamento.md) |
| clusters | [`documentation/03`](documentation/03_clusters_espaciais.md) |
| cada pipeline | [`04` LSTM](documentation/04_cluster_lstm.md), [`05` MLP](documentation/05_cluster_mlp.md), [`06` lazy](documentation/06_cluster_lazy.md) |
| inferência e artefatos | [`documentation/08`](documentation/08_inferencia_e_artefatos.md) |
| estudo de features (método, resultados, comandos) | [`documentation/09`](documentation/09_estudo_de_features.md) |
| os notebooks | [`notebooks/README.md`](notebooks/README.md) |
