# Dados e Pré-processamento

Esta etapa é obrigatória em todas as pipelines. O que varia entre elas é o conjunto
de features habilitado (`feature_groups`).

## Fontes de dados

- `Training_Dataset_INMET_ERA5_Paired.csv`: fonte upstream do par INMET/ERA5 usado para constrói a base operacional do projeto.
- `INMET_Stratified.nc`: alvo observado (`y`) e metadados das estações, gerado a partir do CSV paired.
- `ERA5_Stratified.nc`: variáveis meteorológicas de entrada (`x`), gerado a partir do CSV paired.
- `ERA5_Features_Basin_2000_2026.nc`: grade diária de 0.25° da bacia — fonte das features ERA5 (`original`, `era5_basin`).
- `new_features/<região>/{sl,pl,static}/`: features ERA5 novas em teste (grupo `new_features`), lidas automaticamente — ver abaixo.
- `shp/shp_vento.shp`: insumo espacial auxiliar para atribuição dos clusters, não uma fonte principal de dados meteorológicos.

## Split temporal atual — blocos de mês

As três pipelines de modelagem (lazy, mlp e LSTM) usam a **mesma** partição
(`src/pipeline/data/splits.py::MonthBlockSplit`):

- **Teste**: janeiro, abril, julho e outubro de todos os anos — um mês por
  trimestre climático, cobrindo as quatro estações do ano.
- **Treino**: os oito meses restantes.
- **Validação**: blocos (ano, mês) sorteados dentro dos meses de treino,
  estratificados por mês e aplicados a todas as estações ao mesmo tempo, de
  modo que um mesmo dia nunca caia em dois splits por estações diferentes. Os
  blocos sorteados vão para o metadado do experimento (`split.val_units`),
  tornando a partição exatamente reprodutível.

Em lazy e mlp a partição é resolvida **uma vez** sobre todo o eixo de tempo, no
loader, e viaja com o dado na coluna `_split` (`assign_split_labels` /
`split_part` em `src/pipelines/common.py`). Recalculá-la por cluster ou por
fold devolveria outros blocos de validação, e a climatologia — ajustada só nos
dias de treino — ficaria incoerente com o treino.

A LSTM aplica ainda a **purga de janela**: como consome janelas de 7 dias, uma
amostra só entra no split do seu dia-alvo se todos os dias da janela têm o
mesmo rótulo. Em lazy e mlp não há purga: a amostra é um único dia.

> **Consequência obrigatória:** sob blocos de mês, os dias-do-ano dos meses de
> teste não têm nenhuma amostra de treino. Uma climatologia por média de
> dia-do-ano ficaria indefinida em 120 dos 126 dias-do-ano avaliados e, como
> linhas sem climatologia são descartadas, apagaria ~95% do teste. Por isso as
> três pipelines usam a **climatologia harmônica** (ver abaixo).

O `cluster_gan` é a exceção: segue no split por blocos de ano
(`TRAIN_SLICE`/`VAL_SLICE`/`TEST_SLICE`), o que significa que seu pool de
treino inclui os meses de teste das demais pipelines. Como as amostras
sintéticas alimentam o treino de lazy/mlp, há aí um vazamento indireto
conhecido e ainda não resolvido — relevante apenas para os braços que usam
dados sintéticos.

## Features

Só entram features derivadas do ERA5 (mais calendário e coordenadas da estação):

- **ERA5**: variáveis do ERA5-Basin e derivadas (lags, rolling, climatologia ERA5).
- **Sazonalidade e localização**: `day/month sin/cos`, `latitude`, `longitude`.
- **Features novas** (`nf_*`): de `dataset/raw/new_features`.

Grupos (`feature_groups`): `original` (padrão), `era5_basin`, `new_features`, ou `all`.
`new_features_static` e `new_features_dynamic` recortam o grupo `new_features`
por `KNOWN_STATIC_FEATURES` (`src/data/new_features.py`) — hoje
`nf_orog_height`, `nf_lsm`, `nf_sdor`, `nf_isor`, `nf_anor`, `nf_slor`,
`nf_sdfor` (estáticas, relevo/superfície) vs. o restante (`nf_fg10_*`,
`nf_cape_*`, `nf_ws10_*` etc., agregadas por dia).

## Features novas (`dataset/raw/new_features`)

Qualquer arquivo em `new_features/<região>/sl/<nome>_<AAAA>.nc` (horário),
`pl/<nome>_<AAAA>.nc` (horário por nível) ou `static/<nome>.nc` é lido sem mudar código
(`src/data/new_features.py`):

- A grade tem de ser um recorte da grade do ERA5-Basin (mesmos nós). Grade
  diferente é erro. Células fora da máscara da bacia ficam NaN, como na base antiga.
- Agregação horária → diária por dia UTC (mesma convenção do ERA5-Basin):
  `fg10`/`cape` → max e mean; `u10,v10` → `ws10_max/mean/std`; `u100,v100` →
  `ws100_max/mean` (+ `shear_100_10_mean` com u10/v10); `sp` → mean, min, diff;
  `fsr` → mean; demais → mean e max; `pl` → mean por nível; estáticas constantes
  (`z` vira `orog_height` em m).
- Variável com período incompleto em relação às demais da região (ano faltando,
  hora faltando, NaN na bacia) é descartada inteira, com aviso.
- Resultado em cache (`new_features/_daily_cache.nc`), refeito quando os arquivos mudam.
  Fica fora do `era5_merged_cache*.nc` e é sempre relido.
- No Modal, arquivos novos dessa pasta sobem para o volume automaticamente.

## Regras do preprocessor (sem imputação)

- `select_complete_rows` (`src/pipelines/common.py`) é a regra única de todas as pipelines:
  - feature pedida que não existe é erro;
  - com `new_features`, só treinam os clusters com **todas** as estações dentro
    da cobertura das features novas (cluster parcialmente coberto sai inteiro);
  - linha com NaN em qualquer feature ativa é descartada.
- `RobustScaler` é ajustado no treino. Não há imputer: NaN que chegue ao modelo é erro.
- Na inferência, linha/janela/célula com feature ausente fica sem predição (NaN).

## Climatologia (`era5_clim_wind`)

Série harmônica (3 harmônicos) do proxy ERA5 `wind_mag_max`, ajustada por
mínimos quadrados **só nos dias de treino**, independente por ponto
(`src/data/climatology.py::get_harmonic_climatology`). É contínua no ano e fica
definida nos 366 dias, inclusive nos meses de teste — o que a média por
dia-do-ano não garante sob blocos de mês (ver acima).

Na validação espacial, a mesma definição vale para a climatologia
**populacional** por fold (`src/pipeline/validation/climatology_population.py`):
média sobre as demais estações do fold, ajustada por harmônicos. Com poucos
dias distintos o ajuste degenera para a média constante do fold — que continua
definida nos 366 dias, que é a propriedade da qual o fold depende.

A inferência precisa reproduzir **exatamente** a mesma climatologia: os
artefatos de lazy/mlp gravam `split` e `climatology`, e
`src/inference/grid_direct_predict.py` reajusta a harmônica por célula a partir
deles. Artefato que use `era5_clim_wind` sem registrar a partição é recusado —
adivinhar produziria feature fora da distribuição de treino, em silêncio.

## Observações importantes

- O INMET entra só como alvo: nenhuma feature (lags, climatologia, mediana por
  estação) é derivada da observação, pois ela não existe na grade ERA5.
- Na LSTM, dia descartado vira rótulo "out" e a purga remove as janelas que o tocam.
- A feature engineering é parte fixa do fluxo; o configurável é o subconjunto usado.