# Pipeline `cluster_gan`

Esta pipeline gera dados sintéticos para augmentation.

## Objetivo

Produzir amostras de rajada extrema por cluster para compensar a escassez de
eventos raros no treino.

## Modelo

É uma cWGAN-GP condicional:

- Generator: ruído latente + one-hot do cluster.
- Critic: distingue real vs sintético condicionado ao cluster.
- Gradient penalty: estabiliza o treino.

## Saídas

- `synthetic_extremes.csv` com alvos sintéticos;
- `synthetic_augment.csv` com features ERA5 + alvo para uso downstream.

## Uso downstream

- `cluster_lazy` pode consumir o CSV sintético via `--synthetic-csv`.
- `cluster_mlp` também pode usar augmentation quando configurado.

## Observação prática

O ganho costuma ser maior em clusters menores. Em clusters grandes, é melhor usar
fração proporcional do treino (`--n-per-cluster-ratio`) para evitar dominância de
sintéticos.