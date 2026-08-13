# Pipeline `cluster_lstm`

Esta é a pipeline neural principal em produção.

## Modelo

- LSTM com duas cabeças de saída.
- Cabeça normal com Huber loss.
- Cabeça extrema com `robust_extreme_loss`.
- Variante `cluster_tr_lstm` existe para entrada dupla, mas não é o caminho padrão.

## Entrada


## Pré-processamento

## Saída

# Pipeline `cluster_lstm`

Esta é a pipeline neural principal em produção.

## Arquitetura do modelo

O modelo treinado por esta pipeline é um LSTM com backbone compartilhado e duas saídas paralelas:

- Entrada: tensor com formato `(lookback, n_features)`.
- Backbone: `LSTM(units=64)` seguido de `BatchNormalization()`.
- Cabeça intermediária: `Dense(16, activation="elu")`.
- Dropout de regularização: `Dropout(dropout)`.
- Saída 1 (`head_normal`): `Dense(1)`, otimizada com Huber loss.
- Saída 2 (`head_extreme`): `Dense(1)`, otimizada com `robust_extreme_loss`.

A lógica de multi-task é explícita no builder:

- o mesmo alvo é passado para as duas saídas;
- a cabeça normal foca na previsão estável da média;
- a cabeça extrema enfatiza eventos raros e intensos;
- no inferência, quando a previsão da cabeça normal excede o limiar de extremo, a saída final é substituída pela previsão da cabeça extrema.

Em resumo, a arquitetura em código é:

```text
Input(shape=(lookback, n_features))
	-> LSTM(64)
	-> BatchNormalization()
	-> Dense(16, activation='elu')
	-> Dropout(dropout)
	-> Dense(1)  # head_normal
	-> Dense(1)  # head_extreme
```

A variante de LSTM simples também existe no código, com um `Sequential` mais direto:

```text
Input(shape=(lookback, n_features))
	-> LSTM(units)
	-> BatchNormalization()
	-> Dense(16, activation='elu')
	-> Dropout(dropout)
	-> Dense(1)
```

## Saída

- modelos `.keras` por cluster;
- `histories.json` com a evolução do treino;
- resultados consolidados para validação/teste;
- artefatos usados pela inferência espacial DL.

## Quando usar

Use quando a prioridade for uma arquitetura recorrente com cabeças especializadas
para evento normal e extremo. Na prática, ela serve como benchmark neural principal,
mas não é a melhor solução em todos os clusters.

## Quando usar

Use quando a prioridade for uma arquitetura recorrente com cabeças especializadas
para evento normal e extremo. Na prática, ela serve como benchmark neural principal,
mas não é a melhor solução em todos os clusters.