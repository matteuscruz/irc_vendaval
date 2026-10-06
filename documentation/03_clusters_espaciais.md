# Clusters Espaciais

O shapefile atual separa as estações em 14 clusters espaciais. A motivação é
reduzir a mistura de regimes meteorológicos distintos.

## Distribuição atual

| Cluster | Estações |
|---------|----------|
| C1 | 9 |
| C2 | 16 |
| C3 | 21 |
| C4 | 31 |
| C5 | 22 |
| C6 | 16 |
| C7 | 9 |
| C8 | 15 |
| C9 | 44 |
| C10 | 32 |
| C11 | 16 |
| C12 | 19 |
| C13 | 10 |
| C14 | 11 |

## Por que clusterizar

- cada região combina relevo, rugosidade e dinâmica atmosférica diferentes;
- modelos globais tendem a misturar regimes pouco comparáveis;
- clusters menores podem se beneficiar mais dos dados de clusters vizinhos.

## Uso na pipeline

- `cluster_lstm` treina um modelo por cluster.
- `cluster_mlp` também treina por cluster, mas com input tabular.
- `cluster_lazy` avalia modelos por cluster e, opcionalmente, por trimestre.