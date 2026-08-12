"""Loader para ERA5_Features_Basin_2000_2026.nc — grade regional pré-agregada.

22 features diárias já pré-computadas (média/pico/desvio/percentil/lags) numa
grade regular de 0.25° cobrindo o Sul/Sudeste do Brasil (bacia hidrográfica,
não co-localizada nas estações INMET, diferente do ERA5_Stratified.nc). Este
loader:
  1. Abre o arquivo (lazy, backend netCDF4 — não precisa de dask, é um único
     arquivo, sem open_mfdataset)
  2. Interpola nearest-neighbor para as coordenadas das estações INMET
  3. Renomeia variáveis com sufixo _basin (evita colisão com features ERA5 base)
  4. Retorna xr.Dataset com dims (time, estacao) — compatível com NetCDFLoader
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from src.utils.heartbeat import Heartbeat


class ERA5BasinLoader:
    """Carrega ERA5_Features_Basin (grade da bacia) e interpola pras estações INMET."""

    FILENAME = "ERA5_Features_Basin_2000_2026.nc"

    def __init__(self, raw_dir: str) -> None:
        self.path = Path(raw_dir) / self.FILENAME

    @staticmethod
    def _interp_to_stations(
        ds: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        station_ids: np.ndarray,
    ) -> xr.Dataset:
        """Interpolação nearest do grid pra coordenadas das estações.

        Vetorizado: uma única chamada `.sel()` com indexadores compartilhando
        a dim "estacao" faz as N buscas nearest-neighbor de uma vez (indexação
        "pointwise" do xarray), em vez de um loop Python de N chamadas `.sel()`
        sequenciais — evita releitura repetida do arquivo (crítico pra grades
        grandes/lazy; um loop de 243 estações chegou a levar ~2h localmente).

        Estações fora do bounding box do grid recebem NaN (`.sel(nearest)`
        não gera NaN sozinho fora dos limites, por isso a máscara manual);
        estações dentro do bounding box mas fora da máscara da bacia
        hidrográfica também acabam com NaN (a célula mais próxima já é NaN
        nesse caso).
        """
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
        in_bounds = (
            (lats >= lat_min) & (lats <= lat_max)
            & (lons >= lon_min) & (lons <= lon_max)
        )

        lat_da = xr.DataArray(lats, dims="estacao", coords={"estacao": station_ids})
        lon_da = xr.DataArray(lons, dims="estacao", coords={"estacao": station_ids})
        ds_sta = ds.sel(latitude=lat_da, longitude=lon_da, method="nearest")
        ds_sta = ds_sta.drop_vars(
            [c for c in ("latitude", "longitude") if c in ds_sta.coords]
        )

        if not in_bounds.all():
            mask = xr.DataArray(in_bounds, dims="estacao", coords={"estacao": station_ids})
            ds_sta = ds_sta.where(mask)

        return ds_sta

    def load(self, ds_inmet: xr.Dataset) -> xr.Dataset:
        """Carrega ERA5 Basin e retorna Dataset alinhado com (time, estacao).

        Parameters
        ----------
        ds_inmet : xr.Dataset
            Dataset INMET com coordenadas latitude/longitude por estação.

        Returns
        -------
        xr.Dataset
            22 variáveis com sufixo `_basin`, dims (time, estacao). Estações
            fora do bounding box ou fora da máscara da bacia recebem NaN.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"ERA5 Basin não encontrado: {self.path}")

        lats = ds_inmet.latitude.values.astype(float)
        lons = ds_inmet.longitude.values.astype(float)
        station_ids = ds_inmet.estacao.values

        import time as _time

        print(f"[ERA5-Basin] Abrindo {self.path.name}...")
        _t_open = _time.time()
        ds = xr.open_dataset(self.path)
        print(f"[ERA5-Basin] Arquivo aberto em {_time.time() - _t_open:.0f}s "
              f"(lazy — só metadados, ainda não leu dado do disco).")

        print(f"[ERA5-Basin] Montando indexadores nearest-neighbor pra "
              f"{len(station_ids)} estações...")
        _t_sel = _time.time()
        ds_sta = self._interp_to_stations(ds, lats, lons, station_ids)
        print(f"[ERA5-Basin] .sel(nearest) montado em {_time.time() - _t_sel:.0f}s "
              f"(ainda lazy — o .sel() em si não força leitura de disco).")

        # Ponto crítico: .sel(nearest) num Dataset lazy só MONTA os índices,
        # não lê nada do arquivo ainda — a leitura de verdade (potencialmente
        # cara: 271 pontos espalhados num grid de 6GB SEM chunking interno
        # — confirmado via encoding: contiguous=True, chunksizes=None — o
        # que força varredura ampla do arquivo por ponto, agravado pelo
        # volume do Modal ser um filesystem de rede) só acontece na PRIMEIRA
        # vez que algo pede os valores.
        #
        # Testado ler variável-por-variável (22 chamadas .load()) pra
        # visibilidade — ficou ~3.3x MAIS LENTO (cada .load() isolado paga
        # overhead fixo alto nesse arquivo sem chunking). Aqui divide por
        # LOTES DE ESTAÇÕES em vez de variáveis (poucas chamadas — ~6-7 — em
        # vez de 22), com checkpoint em disco por lote: se a task cair no
        # meio, a próxima tentativa pula os lotes já lidos em vez de reler
        # tudo do zero. Ainda corre o mesmo risco de overhead por chamada,
        # só que numa escala bem menor.
        batch_dir = self.path.parent / "_era5_basin_batches"
        batch_dir.mkdir(exist_ok=True)
        batch_size = 40
        n_stations = len(station_ids)
        n_batches = (n_stations + batch_size - 1) // batch_size
        print(f"[ERA5-Basin] Lendo do disco em {n_batches} lote(s) de até "
              f"{batch_size} estações ({len(ds_sta.data_vars)} variáveis × "
              f"{ds_sta.sizes.get('time', '?')} timesteps cada), checkpoint "
              f"por lote em {batch_dir}...")
        _t_load = _time.time()
        batch_datasets = []
        for b in range(n_batches):
            start, end = b * batch_size, min((b + 1) * batch_size, n_stations)
            expected_ids = list(station_ids[start:end])
            batch_path = batch_dir / f"batch_{b:03d}.nc"

            if batch_path.exists():
                cached = xr.open_dataset(batch_path).load()
                if list(cached.estacao.values) == expected_ids:
                    print(f"[ERA5-Basin] Lote {b + 1}/{n_batches} já em "
                          f"cache ({batch_path.name}), pulando.")
                    batch_datasets.append(cached)
                    continue
                print(f"[ERA5-Basin] Lote {b + 1}/{n_batches} em cache não "
                      f"bate com as estações esperadas — recalculando.")
                cached.close()

            _t_batch = _time.time()
            with Heartbeat(f"lote {b + 1}/{n_batches}", interval=30):
                ds_batch = ds_sta.isel(estacao=slice(start, end)).load()
            tmp_batch_path = batch_path.with_suffix(".nc.tmp")
            ds_batch.to_netcdf(tmp_batch_path)
            tmp_batch_path.rename(batch_path)
            print(f"[ERA5-Basin] Lote {b + 1}/{n_batches} ({end - start} "
                  f"estações) lido e salvo em "
                  f"{_time.time() - _t_batch:.0f}s (acumulado: "
                  f"{_time.time() - _t_load:.0f}s).")
            batch_datasets.append(ds_batch)

        ds_sta = xr.concat(batch_datasets, dim="estacao")
        print(f"[ERA5-Basin] Leitura do disco concluída em "
              f"{_time.time() - _t_load:.0f}s.")

        for v in list(ds_sta.data_vars):
            ds_sta = ds_sta.rename({v: f"{v}_basin"})

        # Alinhar ao período do INMET
        t_min = ds_inmet.time.min().values
        t_max = ds_inmet.time.max().values
        ds_sta = ds_sta.sel(time=slice(t_min, t_max))

        # dim="time" (não axis posicional) evita depender da ordem das dims
        # após o concat (aqui fica "estacao" primeiro, diferente do que a
        # posição do axis=0 sugeriria). Já materializado (.load() acima), sem
        # custo extra aqui.
        first_var = ds_sta[next(iter(ds_sta.data_vars))]
        n_valid = int(first_var.notnull().any(dim="time").sum())
        print(f"[ERA5-Basin] Concluído: {len(ds_sta.data_vars)} variáveis, "
              f"{n_valid}/{len(station_ids)} estações com dados válidos.")

        ds.close()
        return ds_sta
