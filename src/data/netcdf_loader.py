from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from src.utils.heartbeat import Heartbeat


class NetCDFLoader:
    """Carrega o alvo observado do INMET (y) e as variáveis de entrada do ERA5 (x), alinhando-os temporalmente."""

    ERA5_FEATURES = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure",
        "2m_dewpoint_temperature",
        "total_precipitation",
    ]

    def __init__(self, raw_dir: str) -> None:
        self.raw_dir = Path(raw_dir)

    def load(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Retorna (ds_inmet, ds_era5) alinhados em dimensão (time_daily, estacao)."""
        ds_inmet = xr.open_dataset(self.raw_dir / "INMET_Stratified.nc")
        ds_era5_raw = xr.open_dataset(self.raw_dir / "ERA5_Stratified.nc")

        # Calcular wind_mag horário antes do resample
        u_h = ds_era5_raw["10m_u_component_of_wind"]
        v_h = ds_era5_raw["10m_v_component_of_wind"]
        ds_era5_raw = ds_era5_raw.assign(wind_mag=np.sqrt(u_h**2 + v_h**2))

        # Resample horário → diário
        base_vars = self.ERA5_FEATURES + ["wind_mag"]
        ds_mean = ds_era5_raw[base_vars].resample(time="1D").mean()
        ds_max = ds_era5_raw[["wind_mag"]].resample(time="1D").max().rename(
            {"wind_mag": "wind_mag_max"}
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice",
                category=RuntimeWarning,
            )
            ds_std = ds_era5_raw[["wind_mag"]].resample(time="1D").std(ddof=0).rename(
                {"wind_mag": "wind_mag_std"}
            )
        ds_min = ds_era5_raw[["wind_mag"]].resample(time="1D").min().rename(
            {"wind_mag": "wind_mag_min"}
        )
        ds_t2m_max = (
            ds_era5_raw[["2m_temperature"]].resample(time="1D").max()
            .rename({"2m_temperature": "t2m_max"})
        )
        ds_t2m_min = (
            ds_era5_raw[["2m_temperature"]].resample(time="1D").min()
            .rename({"2m_temperature": "t2m_min"})
        )

        ds_era5_daily = xr.merge(
            [ds_mean, ds_max, ds_std, ds_min, ds_t2m_max, ds_t2m_min],
            compat="override",
        )

        # Features derivadas
        wind_mean_safe = ds_era5_daily["wind_mag"].clip(0.1)
        T_c = ds_era5_daily["2m_temperature"] - 273.15
        Td_c = ds_era5_daily["2m_dewpoint_temperature"] - 273.15
        rh = (
            100
            * np.exp(17.625 * Td_c / (243.04 + Td_c))
            / np.exp(17.625 * T_c / (243.04 + T_c))
        ).clip(0, 100)

        u_d = ds_era5_daily["10m_u_component_of_wind"]
        v_d = ds_era5_daily["10m_v_component_of_wind"]
        wind_angle = np.arctan2(v_d, u_d)

        ds_era5_daily = ds_era5_daily.assign(
            # Razão pico/média intradiária — proxy do caráter convectivo do dia
            gust_factor=ds_era5_daily["wind_mag_max"] / wind_mean_safe,
            # Amplitude térmica diária — proxy de mistura vertical
            t2m_range=ds_era5_daily["t2m_max"] - ds_era5_daily["t2m_min"],
            # Umidade relativa pela fórmula de Magnus
            relative_humidity=rh,
            # Tendência de pressão — queda rápida precede vendavais
            pressure_tendency=ds_era5_daily["surface_pressure"].diff("time"),
            # Persistência temporal — lags de 1 a 3 dias
            lag1_wind_mag_max=ds_era5_daily["wind_mag_max"].shift(time=1),
            lag2_wind_mag_max=ds_era5_daily["wind_mag_max"].shift(time=2),
            lag3_wind_mag_max=ds_era5_daily["wind_mag_max"].shift(time=3),
            # Lag semanal — padrão sinótico de 7 dias
            lag7_wind_mag_max=ds_era5_daily["wind_mag_max"].shift(time=7),
            # Tendência de médio prazo — média móvel dos últimos 7 dias
            rolling7d_wind_mag_max=ds_era5_daily["wind_mag_max"].rolling(
                time=7, min_periods=3
            ).mean(),
            # Direção do vento codificada como sin/cos (evita descontinuidade em 360°)
            wind_dir_sin=np.sin(wind_angle),
            wind_dir_cos=np.cos(wind_angle),
        )

        # Conectividade temporal: anomalia e lags dependentes de features já calculadas
        ds_era5_daily = ds_era5_daily.assign(
            # Desvio do dia em relação à média recente — sinal de evento anômalo
            wind_mag_max_anom=(
                ds_era5_daily["wind_mag_max"] - ds_era5_daily["rolling7d_wind_mag_max"]
            ),
            # Janela sinótica curta — passagens frontais duram ~3 dias
            rolling3d_wind_mag_max=ds_era5_daily["wind_mag_max"].rolling(
                time=3, min_periods=2
            ).mean(),
            # Tendência de pressão de ontem — aprofundamento de baixas precede vendavais
            lag1_pressure_tendency=ds_era5_daily["pressure_tendency"].shift(time=1),
        )

        # Alinhar ao período do INMET
        ds_era5_aligned = ds_era5_daily.sel(
            time=slice(ds_inmet.time.min(), ds_inmet.time.max())
        )

        return ds_inmet, ds_era5_aligned

    # Cache do merge completo (caro: ~15min+ pras 243 estações) — gerado uma
    # vez por scripts/build_synced_dataset.py, reaproveitado por GAN/Lazy/MLP/
    # LSTM em vez de cada um recalcular do zero. Vive dentro de dataset/raw/,
    # sobe pro volume Modal junto com o resto (não é sentinela — ausência não
    # bloqueia _ensure_dataset(), só faz load_extended() cair no caminho lento).
    CACHE_FILENAME = "era5_merged_cache.nc"

    def load_extended(
        self, use_cache: bool = True, checkpoint_path: str | Path | None = None,
        on_checkpoint_saved: Callable[[], None] | None = None,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Carrega dados base + ERA5-18UTC + BT55 + ERA5-Basin e retorna merge
        unificado, já recortado pra janela de tempo comum entre INMET e as
        fontes ERA5 que carregaram com sucesso.

        Retorna (ds_inmet, ds_era5_extended) onde ds_era5_extended inclui:
        - Todas as variáveis do ERA5_Stratified.nc (base, reescritas por
          ERA5-Basin — ver passo 5 abaixo)
        - Variáveis ERA5-18UTC (SL + PL derivadas) — sufixo _18z
        - Variáveis BT55 (flag convectivo + rolling)
        - Variáveis ERA5-Basin — sufixo _basin

        Estações fora da cobertura dos novos datasets recebem NaN.

        Se `use_cache=True` (padrão) e existir um cache já pronto em
        `raw_dir/era5_merged_cache.nc` (gerado por
        scripts/build_synced_dataset.py), carrega direto dele — pula todo o
        recálculo caro do merge. `ds_inmet` continua sendo carregado
        normalmente (leve, não entra no cache).

        `checkpoint_path` (opcional): quando informado, salva o resultado do
        merge das 4 fontes (passos 1-4, ANTES de rebuild_original_from_basin)
        nesse arquivo assim que terminar — e, se ele já existir numa chamada
        seguinte, carrega dele em vez de refazer o merge. Não substitui o
        cache final (`CACHE_FILENAME`): protege só a parte cara e sujeita a
        timeout/preempção (ver build_dataset_cache em src/modal/cluster_lazy.py)
        contra ter que recomeçar do zero se falhar DEPOIS do merge.

        `on_checkpoint_saved` (opcional): callback sem argumento chamado logo
        depois do checkpoint ser escrito no disco local — em Modal, o volume
        precisa de `.commit()` explícito pra persistir de verdade (sobreviver
        a um restart de container), e essa função não tem acesso ao objeto
        `Volume` do chamador, só ao caminho do arquivo.
        """
        ds_inmet = xr.open_dataset(self.raw_dir / "INMET_Stratified.nc")

        cache_path = self.raw_dir / self.CACHE_FILENAME
        if use_cache and cache_path.exists():
            print(f"[load_extended] Usando cache: {cache_path.name}")
            ds_era5_base = xr.open_dataset(cache_path)
            ds_era5_base = self._backfill_station_coords(ds_era5_base, ds_inmet)
            return ds_inmet, ds_era5_base

        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_path is not None and checkpoint_path.exists():
            print(f"[load_extended] Usando checkpoint intermediário (pós-merge, "
                  f"pré-rebuild): {checkpoint_path.name}")
            ds_era5_base = xr.open_dataset(checkpoint_path)
            return self._finish_load_extended(ds_inmet, ds_era5_base)

        from src.data.era5_18utc_loader import ERA518UTCLoader
        from src.data.bt55_loader import BT55Loader
        from src.data.era5_basin_loader import ERA5BasinLoader

        # 1. Carregar base
        ds_inmet, ds_era5_base = self.load()
        print(
            f"[load_extended] Janela INMET: {str(ds_inmet.time.values[0])[:10]} "
            f"→ {str(ds_inmet.time.values[-1])[:10]}"
        )

        # 2. ERA5-18UTC
        era5_18_dir = self.raw_dir / "dados_era5_parana_18utc"
        if era5_18_dir.exists():
            print("\n[load_extended] Carregando ERA5-18UTC...")
            try:
                ds_era5_18 = ERA518UTCLoader(str(self.raw_dir)).load(ds_inmet)
                # Os timestamps 18UTC são às 18:00:00 — normalizar para meia-noite
                # para alinhar com o ERA5 base (resample diário → 00:00:00)
                import pandas as _pd
                ds_era5_18 = ds_era5_18.assign_coords(
                    time=_pd.DatetimeIndex(ds_era5_18.time.values).floor("D")
                )
                # Alinhar tempo e merge
                common_times = np.intersect1d(ds_era5_base.time.values, ds_era5_18.time.values)
                if len(common_times) > 0:
                    ds_era5_18_aligned = ds_era5_18.sel(time=common_times)
                    ds_era5_base = xr.merge(
                        [ds_era5_base.sel(time=common_times), ds_era5_18_aligned],
                        compat="override",
                    )
                    print(f"[load_extended] ERA5-18UTC merged: {len(ds_era5_18.data_vars)} variáveis, "
                          f"{len(common_times)} timesteps comuns.")
                else:
                    print("[load_extended] AVISO: sem sobreposição temporal ERA5-18UTC.")
            except Exception as e:
                print(f"[load_extended] AVISO: ERA5-18UTC falhou: {e}")
        else:
            print("[load_extended] ERA5-18UTC não encontrado, pulando.")

        # 3. BT55
        bt55_dir = self.raw_dir / "dados_temperatura_brilho_BT55"
        if bt55_dir.exists():
            print("\n[load_extended] Carregando BT55...")
            try:
                ds_bt55 = BT55Loader(str(self.raw_dir)).load(ds_inmet)
                common_times = np.intersect1d(ds_era5_base.time.values, ds_bt55.time.values)
                if len(common_times) > 0:
                    ds_bt55_aligned = ds_bt55.sel(time=common_times)
                    ds_era5_base = xr.merge(
                        [ds_era5_base.sel(time=common_times), ds_bt55_aligned],
                        compat="override",
                    )
                    print(f"[load_extended] BT55 merged: {len(ds_bt55.data_vars)} variáveis, "
                          f"{len(common_times)} timesteps comuns.")
                else:
                    print("[load_extended] AVISO: sem sobreposição temporal BT55.")
            except Exception as e:
                print(f"[load_extended] AVISO: BT55 falhou: {e}")
        else:
            print("[load_extended] BT55 não encontrado, pulando.")

        # 4. ERA5-Basin (grade regional pré-agregada)
        basin_path = self.raw_dir / "ERA5_Features_Basin_2000_2026.nc"
        if basin_path.exists():
            print("\n[load_extended] Carregando ERA5-Basin...")
            try:
                ds_basin = ERA5BasinLoader(str(self.raw_dir)).load(ds_inmet)
                common_times = np.intersect1d(ds_era5_base.time.values, ds_basin.time.values)
                if len(common_times) > 0:
                    ds_basin_aligned = ds_basin.sel(time=common_times)
                    ds_era5_base = xr.merge(
                        [ds_era5_base.sel(time=common_times), ds_basin_aligned],
                        compat="override",
                    )
                    print(f"[load_extended] ERA5-Basin merged: {len(ds_basin.data_vars)} variáveis, "
                          f"{len(common_times)} timesteps comuns.")
                else:
                    print("[load_extended] AVISO: sem sobreposição temporal ERA5-Basin.")
            except Exception as e:
                print(f"[load_extended] AVISO: ERA5-Basin falhou: {e}")
        else:
            print("[load_extended] ERA5-Basin não encontrado, pulando.")

        if checkpoint_path is not None:
            _t_ckpt = time.time()
            print(f"[load_extended] Salvando checkpoint intermediário (pós-merge) "
                  f"em {checkpoint_path} ({len(ds_era5_base.data_vars)} "
                  f"variáveis, {ds_era5_base.sizes.get('time', '?')} "
                  f"timesteps, {ds_era5_base.sizes.get('estacao', '?')} "
                  f"estações)...")
            tmp_path = checkpoint_path.with_suffix(".nc.tmp")
            with Heartbeat("checkpoint to_netcdf()", interval=30):
                ds_era5_base.to_netcdf(tmp_path)
            tmp_path.rename(checkpoint_path)
            print(f"[load_extended] Checkpoint salvo em {time.time() - _t_ckpt:.0f}s.")
            if on_checkpoint_saved is not None:
                on_checkpoint_saved()

        return self._finish_load_extended(ds_inmet, ds_era5_base)

    @staticmethod
    def _finish_load_extended(
        ds_inmet: xr.Dataset, ds_era5_base: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Passos 5-6 de load_extended() — extraído à parte pra poder rodar
        tanto no caminho normal (logo após os merges) quanto a partir de um
        `checkpoint_path` já salvo (pula os merges inteiramente). Instrumentado
        com tempo pra descobrir qual sub-passo é o gargalo real quando o passo
        inteiro estoura o timeout do Modal."""
        # 5. Reescrever ORIGINAL_FEATURES a partir do ERA5-Basin — cobre
        # 236/243 estações, vs. só 30/243 do ERA5_Stratified.nc. Ver
        # src/data/original_features_basin.py pro mapeamento completo.
        _t0 = time.time()
        from src.data.original_features_basin import (
            REQUIRED_BASIN_COLUMNS,
            rebuild_original_from_basin,
        )
        if all(c in ds_era5_base.data_vars for c in REQUIRED_BASIN_COLUMNS):
            try:
                ds_era5_base = rebuild_original_from_basin(ds_era5_base)
                print(f"[load_extended] ORIGINAL_FEATURES reescrito a partir do "
                      f"ERA5-Basin em {time.time() - _t0:.0f}s.")
            except Exception as e:
                print(f"[load_extended] AVISO: reescrita via ERA5-Basin falhou: {e}")
        else:
            print(
                "[load_extended] AVISO: ERA5-Basin ausente/incompleto — mantendo "
                "ORIGINAL_FEATURES derivado do ERA5_Stratified.nc (cobertura 30 estações)."
            )

        # 6. Sincronizar janela de tempo: recorta pra interseção INMET × ERA5
        # (start/end comuns) — descarta dias fora do período observado por
        # ambos, em vez de carregar/processar tempo que nenhuma pipeline usa.
        _t1 = time.time()
        t_start = max(ds_inmet.time.min().values, ds_era5_base.time.min().values)
        t_end = min(ds_inmet.time.max().values, ds_era5_base.time.max().values)
        print(
            f"[load_extended] Janela sincronizada INMET×ERA5: "
            f"{str(t_start)[:10]} → {str(t_end)[:10]}"
        )
        ds_inmet = ds_inmet.sel(time=slice(t_start, t_end))
        ds_era5_base = ds_era5_base.sel(time=slice(t_start, t_end))
        ds_era5_base = NetCDFLoader._backfill_station_coords(ds_era5_base, ds_inmet)
        print(f"[load_extended] sync+backfill em {time.time() - _t1:.0f}s.")

        return ds_inmet, ds_era5_base

    @staticmethod
    def _backfill_station_coords(ds_era5_base: xr.Dataset, ds_inmet: xr.Dataset) -> xr.Dataset:
        """Corrige latitude/longitude NaN nas estações novas.

        `ds_era5_base` nasce de ERA5_Stratified.nc (só 30 estações — rede
        INMET antiga). Os merges com ERA5-18UTC/BT55/ERA5-Basin (cada um já
        reindexado pras 243 estações via `.load(ds_inmet)`) fazem um outer
        join em `estacao`: as 213 estações novas ganham NaN não só nas
        variáveis originais do ERA5_Stratified, mas também nas COORDENADAS
        latitude/longitude — que `ORIGINAL_FEATURES` usa como feature de
        localização (common.py). `rebuild_original_from_basin` reescreve as
        variáveis meteorológicas a partir do ERA5-Basin (236/243 estações),
        mas não toca coordenadas — então lat/lon ficam NaN pras 213 novas
        pra sempre, mesmo depois da reescrita.

        Efeito prático: qualquer pipeline que exija TODAS as features
        finitas por linha/janela (LSTM — cluster_preprocessor.py descarta
        a janela inteira se houver 1 NaN) perde 100% das amostras dessas
        estações; pipelines que toleram/imputam NaN por coluna (MLP, lazy)
        mascaram o problema silenciosamente (treinam com lat/lon errado
        pras estações novas, sem cair — pior, não sem querer).

        Corrige na fonte: lat/lon são propriedade da ESTAÇÃO, não uma
        medição ERA5 — `ds_inmet` sempre tem valor correto pras estações
        que ele conhece.

        `ds_era5_base` nasce de ERA5_Stratified.nc, uma lista FIXA e
        antiga de 30 estações independente de qual fonte alimenta
        `ds_inmet` no momento — se essa lista e o INMET atual divergirem
        (ex: estação descontinuada/renomeada pelo INMET desde então), uma
        estação de ERA5_Stratified pode não ter contrapartida em
        `ds_inmet`. Sem alvo INMET real, essa estação não serve pra
        treinar de qualquer forma — descarta antes do backfill, senão o
        `.sel()` abaixo quebra (KeyError) tentando buscar uma estação
        ausente.
        """
        valid = ds_era5_base.estacao.isin(ds_inmet.estacao.values)
        if not bool(valid.all()):
            dropped = ds_era5_base.estacao.values[~valid.values]
            print(
                f"[_backfill_station_coords] {len(dropped)} estação(ões) de "
                f"ERA5_Stratified.nc sem contrapartida no INMET atual, "
                f"descartada(s): {list(dropped)}"
            )
            ds_era5_base = ds_era5_base.sel(estacao=valid)

        ds_era5_base = ds_era5_base.assign_coords(
            latitude=("estacao", ds_inmet["latitude"].sel(estacao=ds_era5_base.estacao).values),
            longitude=("estacao", ds_inmet["longitude"].sel(estacao=ds_era5_base.estacao).values),
        )
        return ds_era5_base
