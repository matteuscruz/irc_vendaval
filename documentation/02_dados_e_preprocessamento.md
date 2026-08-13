# Dados e Pré-processamento

Esta etapa é obrigatória em todas as pipelines. O que varia entre elas é o conjunto
de features habilitado e o recorte de cobertura.

## Fontes de dados

- `Training_Dataset_INMET_ERA5_Paired.csv`: fonte upstream do par INMET/ERA5 usado para constrói a base operacional do projeto.
- `INMET_Stratified.nc`: alvo observado (`y`) e metadados das estações, gerado a partir do CSV paired.
- `ERA5_Stratified.nc`: variáveis meteorológicas de entrada (`x`), gerado a partir do CSV paired.
- `shp/shp_vento.shp`: insumo espacial auxiliar para atribuição dos clusters, não uma fonte principal de dados meteorológicos.

## Splits temporais atuais

- Treino: 2008–2018.
- Validação: 2019.
- Teste: 2020–2025.

O corte em 2008 existe porque a cobertura antes disso é muito esparsa para o treino
estável do conjunto atual.

## Features

As features são divididas em três grupos:

- **ERA5 bruto**: variáveis que já vêm do NetCDF da reanálise.
- **Derivadas na pipeline**: lags, rolling, sazonalidade e climatologia.
- **Meta da estação**: latitude, longitude e climatologias derivadas do INMET.

## Regras do preprocessor

- `feature_groups` controla quais grupos entram no treino.
- `restrict_coverage` filtra estações quando há features com cobertura regional.
- `SimpleImputer(strategy="mean")` é ajustado no treino.
- `RobustScaler` é ajustado no treino.

## Observações importantes

- `gust_P50` é derivado no runtime a partir do treino.
- Em pipelines temporais, as janelas podem ser descartadas se a cobertura gerar
  NaN incompatível com o recorte configurado.
- A feature engineering é parte fixa do fluxo; o configurável é o subconjunto usado.