# Dados essenciais (Git LFS)

Copie para `dataset/` antes de rodar (`git lfs pull` baixa os arquivos):

```bash
git lfs pull
mkdir -p dataset && cp -r dataset_lfs/raw dataset_lfs/shp dataset/
```

| Arquivo | Compressão |
|---|---|
| `raw/feature_study_cluster_3/features_grupo{1..4}_cluster3.parquet` | parquet, ZSTD |
| `raw/INMET_Stratified.nc`, `raw/cluster_3_hourly.nc` | NetCDF, zlib nível 5 (valores idênticos aos originais) |
| `shp/shp_vento.*` | shapefile dos 14 clusters |

`ERA5_Features_Basin_2000_2026.nc` (6 GB) é grande demais para o LFS e fica fora: baixe do repositório externo indicado na documentação.
