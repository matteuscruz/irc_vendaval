"""Inferência Espacial — Correção de Rajadas ERA5 via Modelos por Cluster.

Usa os **melhores modelos salvos** pelo pipeline LazyPredict (`cluster_lazy`)
para gerar o mapa de rajadas corrigidas.

Abordagem:
  1. Carrega os artefatos joblib (modelo + imputer + scaler) salvos por cluster
  2. Monta o DataFrame flat (mesma pipeline de features do treino)
  3. Prediz rajada corrigida para cada estação INMET no período desejado
  4. Interpola (IDW) as predições pontuais para a grade regular ERA5
  5. Opcionalmente aplica suavização Gaussiana
  6. Exporta NetCDF com `rajada_max_corrigida` em grade (lat × lon × time)

Uso standalone:
    python -m src.inference.spatial_correction \\
        --raw-dir dataset/raw --shp-dir dataset/shp \\
        --models-dir artifacts/lazy_modal/lazy_clusters/exp1/fitted_models \\
        --out-dir output_corrigido_v3 --year 2023

Ou via pipeline Modal (chamado automaticamente após o treinamento).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from src.data.netcdf_loader import NetCDFLoader
from src.data.cluster_assigner import assign_station_clusters
from src.data.climatology import get_climatology
from src.pipelines.common import (
    BASE_FEATURES, TARGET_VAR, ERA5_GUST_PROXY, TRAIN_SLICE, VAL_SLICE, TEST_SLICE,
    build_flat_dataframe, RANDOM_STATE,
)

warnings.filterwarnings("ignore")


# ── IDW interpolation ───────────────────────────────────────────────────────

def _idw_interpolate(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    power: float = 2.0,
    max_neighbors: int = 8,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation from stations to grid.

    Parameters
    ----------
    station_lats, station_lons : 1-D arrays of station coordinates
    values : 1-D array of values at stations (may contain NaN)
    grid_lats, grid_lons : 1-D arrays defining the output grid
    power : IDW exponent (2 = quadratic decay)
    max_neighbors : number of nearest stations to consider

    Returns
    -------
    2-D array (len(grid_lats), len(grid_lons)) with interpolated values
    """
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return np.full((len(grid_lats), len(grid_lons)), np.nan)

    st_coords = np.column_stack([station_lats[valid], station_lons[valid]])
    st_vals = values[valid]
    tree = cKDTree(st_coords)

    LON, LAT = np.meshgrid(grid_lons, grid_lats)
    flat = np.column_stack([LAT.ravel(), LON.ravel()])

    dists, idxs = tree.query(flat, k=min(max_neighbors, len(st_vals)))

    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    # Avoid division by zero for exact station locations
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / dists**power
    weighted_vals = (weights * st_vals[idxs]).sum(axis=1) / weights.sum(axis=1)

    return weighted_vals.reshape(LON.shape)


# ── Main class ───────────────────────────────────────────────────────────────

class SpatialCorrector:
    """Carrega modelos salvos pelo LazyPredict e interpola para grade ERA5."""

    def __init__(
        self,
        raw_dir: str | Path,
        shp_dir: str | Path,
        models_dir: str | Path,
    ):
        self.raw_dir = Path(raw_dir)
        self.shp_dir = str(shp_dir)
        self.models_dir = Path(models_dir)
        self.loader = NetCDFLoader(str(self.raw_dir))

        self.cluster_models: dict[int, dict[str, Any]] = {}
        self.df_all: pd.DataFrame | None = None
        self._station_lats: np.ndarray | None = None
        self._station_lons: np.ndarray | None = None
        self._station_ids: np.ndarray | None = None

        # Grid bounds — grade real do ERA5-Basin (ver _load_era5_basin_grid)
        self._grid_lats: np.ndarray | None = None
        self._grid_lons: np.ndarray | None = None

    # ── Grade de saída ────────────────────────────────────────────────────

    def _load_era5_basin_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Grade real pra interpolação — ERA5_Features_Basin (86×77 pontos,
        0.25°, Sul/Sudeste do Brasil). Antes usávamos np.unique(latitude) do
        ERA5_Stratified.nc, que só tem coordenadas de 30 estações pontuais
        (não é uma grade — `ds.dims` nem tem latitude/longitude reais), o
        que gerava um mapa artificial e limitado. O ERA5-Basin já é a grade
        que o resto do pipeline usa pra derivar ORIGINAL_FEATURES, então é
        a fonte correta de verdade aqui também.
        """
        basin_path = self.raw_dir / "ERA5_Features_Basin_2000_2026.nc"
        ds_basin = xr.open_dataset(basin_path)
        grid_lats = np.sort(ds_basin.latitude.values)[::-1]
        grid_lons = np.sort(ds_basin.longitude.values)
        ds_basin.close()
        return grid_lats, grid_lons

    # ── Step 1: Load models + prepare data ───────────────────────────────

    def prepare(self) -> None:
        """Carrega modelos salvos e constrói DataFrame flat."""
        import joblib

        # 1. Carregar modelos salvos pelo cluster_lazy
        joblib_files = sorted(self.models_dir.glob("best_model_c*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(
                f"Nenhum modelo encontrado em {self.models_dir}. "
                "Execute o pipeline cluster_lazy primeiro."
            )

        print(f"[SpatialCorrector] Carregando {len(joblib_files)} modelo(s)...")
        for jf in joblib_files:
            try:
                artifact = joblib.load(jf)
            except Exception as e:
                print(f"  ✗ {jf.name}: falha ao carregar ({e}) — pulando cluster.")
                continue
            cid = artifact["cluster_id"]
            self.cluster_models[cid] = artifact
            r2 = artifact.get("r2")
            r2_str = f"R²={r2:.4f}, " if r2 is not None else ""
            print(
                f"  ✓ Cluster {cid}: {artifact['model_name']} "
                f"({r2_str}{len(artifact['features'])} features)"
            )

        if not self.cluster_models:
            raise RuntimeError(
                f"Nenhum modelo pôde ser carregado de {self.models_dir} "
                "(todos falharam — verifique dependências, ex: catboost/xgboost/lightgbm)."
            )

        # 2. Carregar dados
        print("\n[SpatialCorrector] Carregando dados (load_extended)...")
        ds_inmet, ds_era5 = self.loader.load_extended()

        self._station_lats = ds_inmet.latitude.values.astype(float)
        self._station_lons = ds_inmet.longitude.values.astype(float)
        self._station_ids = ds_inmet.estacao.values

        # Grid real do ERA5-Basin (não mais o pseudo-grid de estações do
        # ERA5_Stratified.nc)
        self._grid_lats, self._grid_lons = self._load_era5_basin_grid()

        # 3. Clusters + climatologia + DataFrame flat
        print("[SpatialCorrector] Atribuindo clusters...")
        station_clusters = assign_station_clusters(ds_inmet, self.shp_dir)

        print("[SpatialCorrector] Calculando climatologia...")
        ds_clim = get_climatology(ds_inmet, TARGET_VAR, slice(*TRAIN_SLICE))

        print("[SpatialCorrector] Construindo DataFrame flat...")
        self.df_all = build_flat_dataframe(ds_inmet, ds_era5, station_clusters, ds_clim)

        # self.df_all já tem tudo que precisamos (pandas puro) — fechar os
        # handles xarray/netCDF4 aqui. Sem isso, cada Corrector instanciado
        # (1 por combo pipeline×arm) deixa um handle aberto de
        # INMET_Stratified.nc pro resto do processo; múltiplos handles
        # simultâneos do MESMO arquivo (2+ combos processados em sequência)
        # colidem com uma reabertura fresca mais tarde (RuntimeError: NetCDF:
        # HDF error em _load_direction_stations).
        ds_inmet.close()
        ds_era5.close()

        # gust_P50: mesma feature climatológica por estação que cluster_lazy.py
        # usa no treino (mediana do alvo em TRAIN_SLICE, com fallback pra
        # mediana do cluster) — sem isso, um modelo lazy vencedor ficava com
        # essa feature sempre NaN aqui (imputada pela média de treino do
        # imputer, silenciosamente degradando a predição).
        train_mask = (
            (self.df_all["time"] >= TRAIN_SLICE[0])
            & (self.df_all["time"] <= TRAIN_SLICE[1])
        )
        df_train = self.df_all[train_mask]
        gust_p50_station = df_train.groupby("estacao")[TARGET_VAR].median()
        gust_p50_cluster = df_train.groupby("cluster_id")[TARGET_VAR].median()
        self.df_all["gust_P50"] = self.df_all["estacao"].map(gust_p50_station)
        na_mask = self.df_all["gust_P50"].isna()
        if na_mask.any():
            self.df_all.loc[na_mask, "gust_P50"] = (
                self.df_all.loc[na_mask, "cluster_id"].map(gust_p50_cluster)
            )
        print(f"  Shape: {self.df_all.shape}")

    # ── Step 2: Predict at stations ──────────────────────────────────────

    def predict_stations(self, time_slice: tuple[str, str]) -> pd.DataFrame:
        """Prediz rajada corrigida usando o melhor modelo salvo por cluster."""
        df_slice = self.df_all[
            (self.df_all["time"] >= time_slice[0])
            & (self.df_all["time"] <= time_slice[1])
        ].copy()

        df_slice["rajada_corrigida"] = np.nan

        for cid, artifact in self.cluster_models.items():
            mask = df_slice["cluster_id"] == cid
            if not mask.any():
                continue

            features = artifact["features"]
            imputer = artifact["imputer"]
            scaler = artifact["scaler"]
            model = artifact["model"]

            # Extrair features (mesma lista usada no treino)
            available = [f for f in features if f in df_slice.columns]
            if len(available) < len(features):
                missing = set(features) - set(available)
                print(f"  ⚠ Cluster {cid}: {len(missing)} features ausentes, preenchendo com NaN")
                for f in missing:
                    df_slice[f] = np.nan

            x = df_slice.loc[mask, features]

            # Aplicar mesma pipeline de pré-processamento do treino
            x_imp = imputer.transform(x)
            x_scaled = scaler.transform(x_imp)

            # Modelos treinados com Pandas (LazyPredict/Scikit-Learn) podem requerer
            # nomes de colunas explícitos se usaram pipelines ou ColumnTransformer
            x_scaled_df = pd.DataFrame(x_scaled, columns=features)

            # target_kind explícito (novos artifacts) com fallback pro
            # comportamento antigo (artifacts salvos antes dessa chave
            # existir) — só MLPRegressor era tratado como razão.
            target_kind = artifact.get(
                "target_kind",
                "ratio" if artifact.get("model_name") == "MLPRegressor" else "absolute",
            )
            if target_kind == "ratio":
                ratio_preds = model.predict(x_scaled_df)
                # ERA5 proxy baseline
                era5_proxy = df_slice.loc[mask, "wind_mag_max"].values
                era5_safe = np.clip(era5_proxy, 0.1, None)
                preds = ratio_preds * era5_safe
            else:
                preds = model.predict(x_scaled_df)
                
            preds = np.clip(preds, 0, 80)  # range físico
            df_slice.loc[mask, "rajada_corrigida"] = preds

        n_pred = df_slice["rajada_corrigida"].notna().sum()
        n_total = len(df_slice)
        print(f"  Predições: {n_pred}/{n_total} ({n_pred/n_total*100:.1f}%)")

        return df_slice[["time", "estacao", "latitude", "longitude",
                         "cluster_id", TARGET_VAR, ERA5_GUST_PROXY, "rajada_corrigida"]]

    # ── Step 3: IDW to grid ──────────────────────────────────────────────

    def infer_grid(
        self,
        time_slice: tuple[str, str],
        smoothing: str = "gaussian",
    ) -> xr.Dataset:
        """Produz grade corrigida para o período dado.

        Parameters
        ----------
        time_slice : (start, end) strings ISO
        smoothing : 'none' | 'gaussian'

        Returns
        -------
        xr.Dataset com var `rajada_max_corrigida` dims (time, latitude, longitude)
        """
        print(f"\n[SpatialCorrector] Predizendo estações ({time_slice})...")
        df_preds = self.predict_stations(time_slice)
        times = sorted(df_preds["time"].unique())
        n_times = len(times)
        print(f"  {n_times} dias, {df_preds['estacao'].nunique()} estações")

        grid_shape = (n_times, len(self._grid_lats), len(self._grid_lons))
        grid_arr = np.full(grid_shape, np.nan, dtype=np.float32)

        print("[SpatialCorrector] Interpolando IDW para grade...")
        for t_idx, t in enumerate(times):
            day = df_preds[df_preds["time"] == t]
            vals = np.full(len(self._station_ids), np.nan)
            for _, row in day.iterrows():
                st_idx = np.where(self._station_ids == row["estacao"])[0]
                if len(st_idx) > 0:
                    vals[st_idx[0]] = row["rajada_corrigida"]

            grid_arr[t_idx] = _idw_interpolate(
                self._station_lats, self._station_lons, vals,
                self._grid_lats, self._grid_lons,
            )
            if (t_idx + 1) % 60 == 0 or t_idx == n_times - 1:
                print(f"  IDW: {t_idx + 1}/{n_times} dias")

        if smoothing == "gaussian":
            print("[SpatialCorrector] Suavização gaussiana (σ=1)...")
            for t in range(n_times):
                field = grid_arr[t]
                if not np.all(np.isnan(field)):
                    grid_arr[t] = gaussian_filter(field, sigma=1.0)

        # Resumo dos modelos usados
        model_summary = ", ".join(
            f"C{cid}={a['model_name']}"
            for cid, a in sorted(self.cluster_models.items())
        )

        ds_out = xr.Dataset(
            {"rajada_max_corrigida": (["time", "latitude", "longitude"], grid_arr)},
            coords={
                "time": pd.DatetimeIndex(times),
                "latitude": self._grid_lats,
                "longitude": self._grid_lons,
            },
        )
        ds_out["rajada_max_corrigida"].attrs = {
            "units": "m/s",
            "long_name": f"Rajada Max Corrigida (IDW + {smoothing})",
            "source": "IRC Vendaval V3 — SpatialCorrector",
            "models": model_summary,
        }
        return ds_out

    # ── Step 4: Save ─────────────────────────────────────────────────────

    def save(self, ds: xr.Dataset, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path)
        print(f"[SpatialCorrector] Salvo: {out_path}")

        # Gerar o plot espacial (máximo histórico do período)
        try:
            self._plot_map(ds, out_path.with_suffix(".png"))
        except Exception as e:
            print(f"[SpatialCorrector] Erro ao plotar mapa: {e}")

    def _plot_map(self, ds: xr.Dataset, out_path: Path) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        print("[SpatialCorrector] Gerando plot espacial...")
        da_max = ds["rajada_max_corrigida"].max(dim="time").load()

        # matplotlib's contourf exige coordenadas monotonicamente crescentes. 
        # ERA5 tipicamente tem latitudes decrescentes, o que resulta em um mapa vazio.
        da_max = da_max.sortby(["latitude", "longitude"])

        WIND_COLORS = [
            "#d4e9f7", "#95c8e8", "#4da6d6", "#1a7fc4", "#0d5a9e", "#1a8a3c",
            "#6ab84d", "#c8d84a", "#f5d020", "#f5980a", "#e85a0a", "#c41a0a",
            "#8c0a1a", "#4a0a28"
        ]
        cmap_wind = LinearSegmentedColormap.from_list("wind_nws", WIND_COLORS, N=512)

        fig = plt.figure(figsize=(10, 8), facecolor="#07111c")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        lon_min, lon_max = float(da_max.longitude.min()), float(da_max.longitude.max())
        lat_min, lat_max = float(da_max.latitude.min()), float(da_max.latitude.max())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#0f1e2c")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#07111c")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#4a7090")
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#243040", linewidth=0.5)

        vmin, vmax = float(da_max.min()), float(da_max.max())
        if vmin == vmax:
            vmax += 0.1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 40)

        X, Y = np.meshgrid(da_max.longitude.values, da_max.latitude.values)
        cf = ax.contourf(
            X, Y, da_max.values,
            levels=levels, cmap=cmap_wind, norm=norm,
            transform=ccrs.PlateCarree(), extend='both', alpha=0.92
        )

        ax.set_title("Rajada Máxima Corrigida (Período)", color="#d0e4f0", fontsize=14, fontweight="bold", pad=15)

        gl = ax.gridlines(draw_labels=True, color='#1a2f40', alpha=0.5, linestyle=':')
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': '#6090a8'}
        gl.ylabel_style = {'size': 10, 'color': '#6090a8'}

        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08, aspect=30, shrink=0.8)
        cbar.ax.tick_params(colors='#6090a8', labelsize=10)
        cbar.set_label("Rajada Máxima (m/s)", color="#a0c4d8", size=12)
        cbar.outline.set_edgecolor("#243040")

        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Plot salvo: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Inferência Espacial V3")
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--shp-dir", default="dataset/shp")
    parser.add_argument("--models-dir", required=True,
                        help="Diretório com best_model_c*.joblib")
    parser.add_argument("--out-dir", default="output_corrigido_v3")
    parser.add_argument("--year", default="2023")
    parser.add_argument("--smoothing", choices=["none", "gaussian"], default="gaussian")
    args = parser.parse_args()

    t0 = _time.time()
    corr = SpatialCorrector(args.raw_dir, args.shp_dir, args.models_dir)
    corr.prepare()

    time_slice = (f"{args.year}-01-01", f"{args.year}-12-31")
    ds = corr.infer_grid(time_slice, smoothing=args.smoothing)

    out_path = Path(args.out_dir) / f"era5_corrigido_{args.year}_{args.smoothing}.nc"
    corr.save(ds, out_path)
    print(f"\nConcluído em {_time.time() - t0:.1f}s")
