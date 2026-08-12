"""Loader para dados de Temperatura de Brilho ≤ 55 °C (BT55).

Fonte: arquivos Parquet mensais (`bt55_pixels_YYYY_MM.parquet`) com flags
diários de convecção profunda por pixel (d01–d31) e indicadores de validade
(v01–v31).

Este loader:
  1. Concatena todos os parquets mensais
  2. Pivota flags diários → série temporal diária por pixel
  3. Interpola nearest-neighbor para as coordenadas das estações INMET
  4. Calcula features derivadas (flag diário, rolling 3d, fração mensal)
  5. Retorna xr.Dataset com dims (time, estacao) — compatível com NetCDFLoader
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


class BT55Loader:
    """Carrega BT55 (Parquet mensal) e interpola para estações INMET."""

    def __init__(self, raw_dir: str) -> None:
        self.bt_dir = Path(raw_dir) / "dados_temperatura_brilho_BT55"

    @staticmethod
    def _build_pixel_kdtree(
        sample_df: pd.DataFrame,
    ) -> tuple[cKDTree, pd.DataFrame]:
        """Constrói KDTree com coordenadas únicas de pixels BT55.

        Returns
        -------
        tree : cKDTree
        pixels : DataFrame com colunas [pixel_id, lat, lon]
        """
        pixels = (
            sample_df[["pixel_id", "lat", "lon"]]
            .drop_duplicates("pixel_id")
            .reset_index(drop=True)
        )
        tree = cKDTree(pixels[["lat", "lon"]].values)
        return tree, pixels

    @staticmethod
    def _find_nearest_pixel(
        tree: cKDTree,
        pixels: pd.DataFrame,
        lat: float,
        lon: float,
        max_dist_deg: float = 0.5,
    ) -> int | None:
        """Retorna pixel_id mais próximo, ou None se distância > max_dist_deg."""
        dist, idx = tree.query([lat, lon])
        if dist > max_dist_deg:
            return None
        return int(pixels.iloc[idx]["pixel_id"])

    def _load_all_parquets(self) -> pd.DataFrame:
        """Carrega e concatena todos os parquets mensais."""
        files = sorted(self.bt_dir.glob("bt55_pixels_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"Nenhum arquivo parquet encontrado em {self.bt_dir}"
            )

        chunks = []
        for f in files:
            df = pd.read_parquet(f)
            chunks.append(df)

        print(f"[BT55] Lidos {len(files)} arquivos parquet.")
        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _pivot_to_daily(df_all: pd.DataFrame) -> pd.DataFrame:
        """Pivota flags diários (d01–d31) para linhas diárias.

        Returns
        -------
        DataFrame com colunas [pixel_id, date, bt55_flag, valid_flag]
        """
        # Encontrar colunas de dias (d01-d31 e v01-v31)
        d_cols = sorted([c for c in df_all.columns if c.startswith("d") and c[1:].isdigit()])
        v_cols = sorted([c for c in df_all.columns if c.startswith("v") and c[1:].isdigit()])

        rows = []
        for _, row in df_all.iterrows():
            ano = int(row["ano"])
            mes = int(row["mes"])
            pid = int(row["pixel_id"])

            for d_col, v_col in zip(d_cols, v_cols):
                day = int(d_col[1:])
                try:
                    date = pd.Timestamp(year=ano, month=mes, day=day)
                except ValueError:
                    # Dia inválido para o mês (ex: 31 de fevereiro)
                    continue
                rows.append({
                    "pixel_id": pid,
                    "date": date,
                    "bt55_flag": int(row[d_col]),
                    "valid_flag": int(row[v_col]),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _pivot_to_daily_vectorized(df_all: pd.DataFrame) -> pd.DataFrame:
        """Versão vetorizada do pivot diário — muito mais rápida.

        Usa pd.melt para transformar colunas d01-d31 em linhas.
        """
        d_cols = sorted([c for c in df_all.columns if c.startswith("d") and c[1:].isdigit()])
        v_cols = sorted([c for c in df_all.columns if c.startswith("v") and c[1:].isdigit()])

        id_cols = ["pixel_id", "ano", "mes"]

        # Melt d-columns
        df_d = df_all[id_cols + d_cols].melt(
            id_vars=id_cols, var_name="day_col", value_name="bt55_flag"
        )
        df_d["day"] = df_d["day_col"].str[1:].astype(int)

        # Melt v-columns
        df_v = df_all[id_cols + v_cols].melt(
            id_vars=id_cols, var_name="day_col", value_name="valid_flag"
        )
        df_v["day"] = df_v["day_col"].str[1:].astype(int)

        # Merge
        df_daily = df_d.merge(df_v[id_cols + ["day", "valid_flag"]], on=id_cols + ["day"])

        # Construir date — filtrar dias inválidos
        df_daily["date"] = pd.to_datetime(
            df_daily[["ano", "mes", "day"]].rename(
                columns={"ano": "year", "mes": "month"}
            ),
            errors="coerce",
        )
        df_daily = df_daily.dropna(subset=["date"])
        df_daily["date"] = df_daily["date"].dt.normalize()

        return df_daily[["pixel_id", "date", "bt55_flag", "valid_flag"]].reset_index(drop=True)

    def load(
        self,
        ds_inmet: xr.Dataset,
        max_dist_deg: float = 0.5,
    ) -> xr.Dataset:
        """Carrega BT55 e retorna Dataset alinhado com (time, estacao).

        Parameters
        ----------
        ds_inmet : xr.Dataset
            Dataset INMET com coordenadas latitude/longitude por estação.
        max_dist_deg : float
            Distância máxima (em graus) para associar estação a pixel.
            Estações mais distantes recebem NaN.

        Returns
        -------
        xr.Dataset
            Dataset com variáveis BT55, dims (time, estacao).
        """
        lats = ds_inmet.latitude.values.astype(float)
        lons = ds_inmet.longitude.values.astype(float)
        station_ids = ds_inmet.estacao.values

        # 1. Carregar dados
        print("[BT55] Carregando parquets mensais...")
        df_all = self._load_all_parquets()

        # 2. Construir KDTree para match estação→pixel
        print("[BT55] Construindo mapeamento estação → pixel...")
        sample = df_all[df_all["ano"] == df_all["ano"].mode().iloc[0]]
        tree, pixels = self._build_pixel_kdtree(sample)

        station_pixel_map = {}
        n_matched = 0
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            pid = self._find_nearest_pixel(tree, pixels, lat, lon, max_dist_deg)
            station_pixel_map[station_ids[i]] = pid
            if pid is not None:
                n_matched += 1

        print(f"[BT55] Estações mapeadas: {n_matched}/{len(station_ids)}")

        # 3. Filtrar apenas pixels necessários
        needed_pixels = {p for p in station_pixel_map.values() if p is not None}
        if not needed_pixels:
            print("[BT55] AVISO: nenhuma estação dentro da cobertura BT55.")
            # Retornar dataset vazio com NaN
            return self._empty_dataset(ds_inmet)

        df_filtered = df_all[df_all["pixel_id"].isin(needed_pixels)]
        del df_all  # liberar memória

        # 4. Pivotar para diário
        print("[BT55] Pivotando flags diários (vetorizado)...")
        df_daily = self._pivot_to_daily_vectorized(df_filtered)
        del df_filtered

        # Mascarar dias sem dados válidos
        df_daily.loc[df_daily["valid_flag"] == 0, "bt55_flag"] = np.nan

        # 5. Construir séries por estação
        print("[BT55] Construindo séries por estação...")
        station_series = {}
        for sta_id in station_ids:
            pid = station_pixel_map.get(sta_id)
            if pid is None:
                station_series[sta_id] = None
                continue
            df_px = df_daily[df_daily["pixel_id"] == pid].set_index("date").sort_index()
            station_series[sta_id] = df_px["bt55_flag"]

        # 6. Construir xr.Dataset
        # Obter range de datas completo
        all_dates = pd.date_range(
            df_daily["date"].min(), df_daily["date"].max(), freq="D"
        )

        data_dict = {}
        bt55_flag_arr = np.full((len(all_dates), len(station_ids)), np.nan, dtype=np.float32)

        for j, sta_id in enumerate(station_ids):
            series = station_series.get(sta_id)
            if series is not None and len(series) > 0:
                # Reindex para todas as datas
                series = series.reindex(all_dates).astype(float)
                bt55_flag_arr[:, j] = series.values

        data_dict["bt55_flag"] = (["time", "estacao"], bt55_flag_arr)

        ds_out = xr.Dataset(
            data_dict,
            coords={
                "time": all_dates,
                "estacao": station_ids,
            },
        )

        # 7. Features derivadas
        ds_out = self._compute_derived(ds_out)

        # 8. Alinhar ao período INMET
        t_min = ds_inmet.time.min().values
        t_max = ds_inmet.time.max().values
        ds_out = ds_out.sel(time=slice(t_min, t_max))

        # dim="time" (não axis posicional) — mesmo padrão defensivo usado nos
        # outros loaders, mesmo array já sendo construído com (time, estacao).
        n_valid = int(ds_out["bt55_flag"].notnull().any(dim="time").sum())
        print(
            f"[BT55] Concluído: {len(ds_out.data_vars)} variáveis, "
            f"{n_valid}/{len(station_ids)} estações com dados válidos."
        )

        return ds_out

    @staticmethod
    def _compute_derived(ds: xr.Dataset) -> xr.Dataset:
        """Calcula features derivadas a partir do flag diário BT55."""
        bt = ds["bt55_flag"]

        # Rolling 3 dias — convecção recente (proxy de ambiente sinótico ativo)
        bt55_rolling3d = bt.rolling(time=3, min_periods=1).mean()

        # Rolling 7 dias — tendência de convecção
        bt55_rolling7d = bt.rolling(time=7, min_periods=3).mean()

        # Fração mensal — climatologia suavizada de convecção
        # (usa rolling 30d como proxy contínuo ao invés de corte mensal)
        bt55_rolling30d = bt.rolling(time=30, min_periods=15).mean()

        ds = ds.assign(
            bt55_rolling3d=bt55_rolling3d,
            bt55_rolling7d=bt55_rolling7d,
            bt55_frac_month=bt55_rolling30d,
        )

        return ds

    @staticmethod
    def _empty_dataset(ds_inmet: xr.Dataset) -> xr.Dataset:
        """Retorna dataset com NaN para todas as estações."""
        times = pd.date_range(
            ds_inmet.time.min().values, ds_inmet.time.max().values, freq="D"
        )
        station_ids = ds_inmet.estacao.values
        n_t, n_s = len(times), len(station_ids)
        nan_arr = np.full((n_t, n_s), np.nan, dtype=np.float32)

        return xr.Dataset(
            {
                "bt55_flag": (["time", "estacao"], nan_arr),
                "bt55_rolling3d": (["time", "estacao"], nan_arr.copy()),
                "bt55_rolling7d": (["time", "estacao"], nan_arr.copy()),
                "bt55_frac_month": (["time", "estacao"], nan_arr.copy()),
            },
            coords={"time": times, "estacao": station_ids},
        )
