"""Grade corrigida (magnitude + direção do vento), usando o MELHOR modelo já
treinado por (cluster, trimestre climático) — versão "inteligente" do
`ref/interpolation/04_correction_models/v2_directional/generate_grid_v2.py`:
em vez de IDW do resíduo bruto (observado − ERA5), usa IDW do resíduo do
modelo vencedor daquele cluster/trimestre (predição do modelo − ERA5).

Fluxo:
  1. `best_model_selector` escolhe (pipeline, arm) vencedor por cluster ×
     trimestre, lendo os artefatos locais já treinados.
  2. Recarrega os modelos vencedores (`SpatialCorrector`/`DLSpatialCorrector`)
     e roda inferência por estação em TODO o período 2020-2024 de uma vez
     por (pipeline, arm) — no máximo 12 combinações, cada uma cacheada.
  3. "Costura" as predições: o valor de cada estação vem do combo que venceu
     o CLUSTER daquela estação NAQUELE trimestre.
  4. Resíduo = predição do vencedor − ERA5 (wind_mag_max), interpolado por
     IDW pra grade ERA5-Basin inteira (não recortado por cluster — ver
     `VectorIDW`). Direção usa IDW vetorial (u/v) da direção OBSERVADA do
     INMET (não passa por correção de modelo).
  5. Soma o resíduo interpolado ao campo bruto do ERA5 (`ws_max`) e salva um
     NetCDF por ano.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.pipelines.common import ERA5_GUST_PROXY, TRAIN_SLICE, month_to_season
from src.dataset.creation.best_model_selector import Combo, build_winner_table

SEASONS_ORDER = ["DJF", "MAM", "JJA", "SON"]


# ── IDW vetorial pré-computado ───────────────────────────────────────────────

class VectorIDW:
    """IDW com vizinhos/pesos pré-computados uma única vez (via cKDTree) e
    reaproveitados por todos os dias do período — o `_idw_interpolate` de
    `spatial_correction.py` reconstrói a árvore a cada dia, inviável num
    range de vários anos (~1800 dias × 14 clusters).

    Também corrige um viés do script de referência (`VectorIDWInterpolator`
    em generate_grid_v2.py): lá, `nan_to_num(0.0)` trata estação SEM DADO
    naquele dia como "reportou zero", puxando o campo interpolado pra baixo
    em áreas de cobertura esparsa. Aqui os pesos são renormalizados só
    sobre os vizinhos VÁLIDOS por (dia, ponto de grade) — dias/estações
    ausentes simplesmente não contribuem, em vez de contribuir com zero.
    """

    def __init__(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
        k: int = 15,
        power: float = 2.0,
    ) -> None:
        from scipy.spatial import cKDTree

        self.grid_lats = grid_lats
        self.grid_lons = grid_lons
        self.grid_shape = (len(grid_lats), len(grid_lons))
        self.k = max(1, min(k, len(station_lats)))
        self.power = power

        st_coords = np.column_stack([station_lats, station_lons])
        tree = cKDTree(st_coords)
        LON, LAT = np.meshgrid(grid_lons, grid_lats)
        grid_pts = np.column_stack([LAT.ravel(), LON.ravel()])
        dists, idxs = tree.query(grid_pts, k=self.k)
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]
        dists = np.maximum(dists, 1e-10)
        self.weights = 1.0 / dists ** power  # (n_grid, k)
        self.indices = idxs  # (n_grid, k), índices em station_lats/lons

    def interpolate_scalar(self, values: np.ndarray) -> np.ndarray:
        """values: (n_stations,), NaN pra estação sem dado nesse dia.
        Retorna grade 2D (lat, lon) — 1 dia."""
        vals_at_neighbors = values[self.indices]  # (n_grid, k)
        valid = np.isfinite(vals_at_neighbors)
        w = np.where(valid, self.weights, 0.0)
        num = np.sum(w * np.where(valid, vals_at_neighbors, 0.0), axis=1)
        den = np.sum(w, axis=1)
        out = np.full(num.shape, np.nan)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out.reshape(self.grid_shape)

    def interpolate_direction(self, degrees: np.ndarray) -> np.ndarray:
        """degrees: (n_stations,) direção meteorológica ("de onde vem o
        vento"), NaN pra estação sem dado. Decompõe em u/v (mesma convenção
        do script de referência), interpola cada componente com a MESMA
        máscara de validade, recompõe via arctan2."""
        rad = np.deg2rad(degrees)
        u = -np.sin(rad)
        v = -np.cos(rad)
        u_grid = self.interpolate_scalar(u)
        v_grid = self.interpolate_scalar(v)
        return (np.degrees(np.arctan2(-u_grid, -v_grid)) + 360) % 360


# ── Recarrega modelos vencedores e prediz por estação ────────────────────────

def _detect_pipeline_kind(models_dir: Path) -> str:
    """Mesmo critério de scripts/run_spatial_inference.py::_detect_pipeline."""
    if (models_dir / "dl_metadata.joblib").exists() or list(models_dir.glob("best_model_c*.keras")):
        return "lstm"
    return "classic"


def _predict_combo(
    combo: Combo, raw_dir: str, shp_dir: str, time_slice: tuple[str, str],
) -> pd.DataFrame:
    """Recarrega os modelos de UM (pipeline, arm) e prediz nas estações pra
    todo o período — 1 instanciação por combo (cara: recarrega NetCDFs e
    reconstrói o DataFrame flat), não 1 por cluster."""
    kind = _detect_pipeline_kind(combo.models_dir)
    if kind == "lstm":
        from src.inference.spatial_correction_dl import DLSpatialCorrector as Corrector
    else:
        from src.inference.spatial_correction import SpatialCorrector as Corrector

    corr = Corrector(raw_dir, shp_dir, combo.models_dir)
    corr.prepare()
    return corr.predict_stations(time_slice)


def _load_direction_stations(raw_dir: str) -> tuple[pd.DataFrame, xr.DataArray]:
    """Direção observada do INMET — independente de qual modelo venceu (não
    passa por correção nenhuma, só interpolação direta do observado)."""
    ds = xr.open_dataset(Path(raw_dir) / "INMET_Stratified.nc")
    da_dir = ds["daily_wind_direction_at_gust_max"].load()
    stations = pd.DataFrame({
        "estacao": ds["estacao"].values,
        "latitude": ds["latitude"].values,
        "longitude": ds["longitude"].values,
    })
    ds.close()
    return stations, da_dir


def _load_era5_basin(raw_dir: str) -> xr.Dataset:
    ds = xr.open_dataset(Path(raw_dir) / "ERA5_Features_Basin_2000_2026.nc")
    return ds


# ── Orquestração ─────────────────────────────────────────────────────────────

def select_quarterly_winners(winner_table: pd.DataFrame) -> pd.DataFrame:
    """Filtra a tabela de vencedores pros vencedores por trimestre (não
    'ALL') que de fato têm modelo salvo — só esses entram na correção
    espacial. Extraído de `run()` sem mudança de lógica."""
    quarterly = winner_table[
        (winner_table["season"] != "ALL") & (winner_table["has_fitted_model"])
    ]
    if quarterly.empty:
        raise RuntimeError(
            "Nenhum vencedor por trimestre com modelo salvo disponível — nada a gerar."
        )
    return quarterly


def predict_all_combos(
    quarterly: pd.DataFrame, combos: list[Combo], raw_dir: str, shp_dir: str,
    time_slice: tuple[str, str],
) -> pd.DataFrame:
    """Recarrega cada (pipeline, arm) vencedor e prediz em todas as estações
    pro período pedido. Extraído de `run()` sem mudança de lógica —
    `_predict_combo` já era uma função independente, só a orquestração do
    loop estava inline."""
    combo_by_key = {(c.pipeline, c.arm): c for c in combos}
    needed = sorted({(row.pipeline, row.arm) for row in quarterly.itertuples()})
    print(f"[corrected_grid] {len(needed)} combinação(ões) (pipeline, arm) a recarregar...")

    station_frames = []
    for pipeline, arm in needed:
        combo = combo_by_key[(pipeline, arm)]
        print(f"  → {pipeline}/{arm} ({time_slice[0]}..{time_slice[1]})...")
        preds = _predict_combo(combo, raw_dir, shp_dir, time_slice)
        if preds.empty:
            print("    ⚠ Nenhuma predição retornada — combo será ignorado.")
            continue
        preds = preds.copy()
        preds["pipeline"] = pipeline
        preds["arm"] = arm
        station_frames.append(preds)

    if not station_frames:
        raise RuntimeError("Nenhuma predição por estação foi gerada — abortando.")
    all_preds = pd.concat(station_frames, ignore_index=True)
    all_preds["time"] = pd.to_datetime(all_preds["time"])
    all_preds["season"] = month_to_season(all_preds["time"].dt.month)
    return all_preds


def stitch_predictions(quarterly: pd.DataFrame, all_preds: pd.DataFrame) -> pd.DataFrame:
    """"Costura": cada estação usa o combo que venceu SEU cluster, NAQUELE
    trimestre específico (não o vencedor do ano inteiro). Extraído de
    `run()` sem mudança de lógica."""
    stitched_frames = []
    for row in quarterly.itertuples():
        sub = all_preds[
            (all_preds["pipeline"] == row.pipeline)
            & (all_preds["arm"] == row.arm)
            & (all_preds["cluster_id"] == row.cluster_id)
            & (all_preds["season"] == row.season)
        ]
        if not sub.empty:
            stitched_frames.append(sub)
    if not stitched_frames:
        raise RuntimeError("Nenhuma predição sobrou após 'costurar' vencedores — abortando.")
    stitched = pd.concat(stitched_frames, ignore_index=True)
    stitched["residual"] = stitched["rajada_corrigida"] - stitched[ERA5_GUST_PROXY]
    print(
        f"[corrected_grid] {len(stitched)} predições costuradas, "
        f"{stitched['estacao'].nunique()} estações únicas."
    )
    return stitched


def run(
    *,
    raw_dir: str = "dataset/raw",
    shp_dir: str = "dataset/shp",
    artifacts_root: str = "artifacts",
    out_dir: str = "artifacts/corrected_grid",
    exp_name: str | None = None,
    start_date: str = "2000-01-08",
    end_date: str = "2025-12-31",
    metric: str = "R2",
    prediction_source: str = "reload",
    interp_mode: str = "basin",
    idw_neighbors: int = 15,
    idw_power: float = 2.0,
    smoothing: str = "none",
    per_year: bool = True,
    clusters: str | None = None,
    winners_only: bool = False,
    precomputed_winner_table: pd.DataFrame | None = None,
    precomputed_combos: list[Combo] | None = None,
    **_,
) -> None:
    from src.utils.artifact_manager import ArtifactManager

    if interp_mode != "basin":
        raise NotImplementedError(
            "interp_mode='per-cluster' ainda não implementado — só 'basin' "
            "(IDW basin-wide, ver Fluxo item 4 no docstring do módulo)."
        )
    if prediction_source != "reload":
        raise NotImplementedError(
            "prediction_source='saved' ainda não implementado (a versão "
            "'reload' evita a descontinuidade val/test e a ambiguidade de "
            "linhas duplicadas do lazy — ver plano)."
        )

    manager = ArtifactManager(out_dir, exp_name)
    print(f"[corrected_grid] Experimento: {manager.root}")

    # ── 1. Vencedores ────────────────────────────────────────────────────
    # precomputed_winner_table/precomputed_combos (Kedro): se já vieram
    # prontos de um passo anterior (ex.: apply_fallback_to_saved_models),
    # não recalcula — evita rodar a seleção duas vezes com a métrica errada
    # por engano (bug real encontrado: o wrapper Kedro descartava a tabela
    # já calculada e a métrica passada, e `run()` recalculava do zero com
    # seu próprio default). Nenhum chamador existente (CLI/Modal) passa
    # esses parâmetros — comportamento 100% inalterado pra eles.
    if precomputed_winner_table is not None and precomputed_combos is not None:
        print("[corrected_grid] Usando tabela de vencedores já calculada (Kedro)...")
        winner_table, combos = precomputed_winner_table, precomputed_combos
    else:
        print("[corrected_grid] Selecionando melhor modelo por cluster × trimestre...")
        winner_table, combos = build_winner_table(artifacts_root, metric)
    if clusters:
        wanted = {int(c) for c in clusters.split(",")}
        winner_table = winner_table[winner_table["cluster_id"].isin(wanted)]

    winners_path = manager.get_root_path("winners.csv")
    winner_table.to_csv(winners_path, index=False)
    print(f"[corrected_grid] Tabela de vencedores salva: {winners_path}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(winner_table.to_string(index=False))

    if winners_only:
        return

    quarterly = select_quarterly_winners(winner_table)

    # ── 2. Predições por estação — reload dos modelos já treinados ───────
    time_slice = (start_date, end_date)
    all_preds = predict_all_combos(quarterly, combos, raw_dir, shp_dir, time_slice)

    # ── 3. "Costura": cada estação usa o combo que venceu SEU cluster,
    # NAQUELE trimestre específico (não o vencedor do ano inteiro). ──────
    stitched = stitch_predictions(quarterly, all_preds)

    # ── 4. Grade ERA5-Basin (fonte de verdade da grade + do campo bruto) ─
    print("[corrected_grid] Carregando grade ERA5-Basin...")
    ds_basin = _load_era5_basin(raw_dir)
    grid_lats = np.sort(ds_basin.latitude.values)[::-1]
    grid_lons = np.sort(ds_basin.longitude.values)

    # ── 5. IDW pré-computado — magnitude (resíduo do vencedor) e direção
    # (observado INMET, todas as estações, independente do vencedor). ────
    mag_stations = (
        stitched[["estacao", "latitude", "longitude"]]
        .drop_duplicates("estacao")
        .reset_index(drop=True)
    )
    mag_idw = VectorIDW(
        grid_lats, grid_lons,
        mag_stations["latitude"].to_numpy(float), mag_stations["longitude"].to_numpy(float),
        k=idw_neighbors, power=idw_power,
    )
    resid_pivot = (
        stitched.pivot_table(index="time", columns="estacao", values="residual", aggfunc="mean")
        .reindex(columns=mag_stations["estacao"])
    )

    print("[corrected_grid] Carregando direção observada (INMET)...")
    dir_stations, da_dir = _load_direction_stations(raw_dir)
    dir_idw = VectorIDW(
        grid_lats, grid_lons,
        dir_stations["latitude"].to_numpy(float), dir_stations["longitude"].to_numpy(float),
        k=idw_neighbors, power=idw_power,
    )

    # ── 6. Um NetCDF por ano ──────────────────────────────────────────────
    from scipy.ndimage import gaussian_filter

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    train_start, train_end = pd.Timestamp(TRAIN_SLICE[0]), pd.Timestamp(TRAIN_SLICE[1])
    years = range(start_ts.year, end_ts.year + 1)
    created: list[str] = []

    for year in years:
        y_start = max(pd.Timestamp(f"{year}-01-01"), start_ts)
        y_end = min(pd.Timestamp(f"{year}-12-31"), end_ts)
        dates = pd.date_range(y_start, y_end, freq="D")
        if len(dates) == 0:
            continue
        print(f"[corrected_grid] Ano {year}: {len(dates)} dias...")

        era5_year = (
            ds_basin["ws_max"].sel(time=dates.intersection(pd.DatetimeIndex(ds_basin.time.values)))
            .reindex(latitude=grid_lats, longitude=grid_lons)
        )
        dates = pd.DatetimeIndex(era5_year.time.values)  # só dias que existem nos 2 lados
        if len(dates) == 0:
            print(f"  ⚠ Nenhuma data em comum com o ERA5-Basin — pulando {year}.")
            continue

        mag_shape = (len(dates), len(grid_lats), len(grid_lons))
        bias_grid = np.full(mag_shape, np.nan, dtype=np.float32)
        dir_grid = np.full(mag_shape, np.nan, dtype=np.float32)

        for i, d in enumerate(dates):
            resid_vec = (
                resid_pivot.loc[d].to_numpy(dtype=float)
                if d in resid_pivot.index
                else np.full(len(mag_stations), np.nan)
            )
            bias_grid[i] = mag_idw.interpolate_scalar(resid_vec)

            if d in da_dir["time"].values:
                dir_vec = da_dir.sel(time=d).values.astype(float)
            else:
                dir_vec = np.full(len(dir_stations), np.nan)
            dir_grid[i] = dir_idw.interpolate_direction(dir_vec)

        if smoothing == "gaussian":
            for i in range(len(dates)):
                if not np.all(np.isnan(bias_grid[i])):
                    bias_grid[i] = gaussian_filter(bias_grid[i], sigma=1.0)

        era5_vals = era5_year.values.astype(np.float32)
        rajada_corrigida = np.clip(era5_vals + bias_grid, 0, None)
        # Antes deste fix, in_sample = dates <= train_end (só o limite
        # superior) — marcava 2000-2007 como in_sample=True mesmo quando
        # TRAIN_SLICE começava em 2008, dando a entender que o modelo tinha
        # visto dado que na verdade nunca esteve em nenhum split de treino.
        in_sample = np.asarray((dates >= train_start) & (dates <= train_end))

        ds_out = xr.Dataset(
            {
                "rajada_max_corrigida": (["time", "latitude", "longitude"], rajada_corrigida),
                "direcao_vento": (["time", "latitude", "longitude"], dir_grid),
                "bias": (["time", "latitude", "longitude"], bias_grid),
            },
            coords={
                "time": dates,
                "latitude": grid_lats,
                "longitude": grid_lons,
                "in_sample": ("time", in_sample),
            },
        )
        ds_out["rajada_max_corrigida"].attrs = {
            "units": "m/s", "long_name": "Rajada máxima corrigida (IDW do resíduo do melhor modelo)",
        }
        ds_out["direcao_vento"].attrs = {
            "units": "degrees", "long_name": "Direção do vento (IDW vetorial, INMET observado)",
        }
        ds_out["bias"].attrs = {
            "units": "m/s", "long_name": "Resíduo interpolado (predição do vencedor − ERA5)",
        }
        ds_out.attrs = {
            "source": "IRC Vendaval — corrected_grid (best-model-per-cluster-per-season)",
            "metric_used": metric,
            "in_sample_period": f"{TRAIN_SLICE[0]}/{TRAIN_SLICE[1]}",
            "n_stations_magnitude": int(len(mag_stations)),
            "n_stations_direction": int(len(dir_stations)),
            "idw_k": idw_neighbors,
            "idw_power": idw_power,
            "winners_csv": "winners.csv",
        }
        for var in ("rajada_max_corrigida", "direcao_vento", "bias"):
            ds_out[var].encoding = {"zlib": True, "complevel": 4}

        out_path = manager.get_root_path(f"grid_corrected_{year}.nc")
        ds_out.to_netcdf(out_path)
        created.append(str(out_path))
        print(f"  ✓ Salvo: {out_path}")

    ds_basin.close()
    print(f"\n[corrected_grid] {len(created)} arquivo(s) gerado(s) em {manager.root}")
