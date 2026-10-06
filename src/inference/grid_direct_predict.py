"""Aplica o modelo vencedor de cada cluster × trimestre DIRETAMENTE em cada
célula da grade ERA5-Basin, produzindo a rajada máxima diária sem passar por
estações nem por IDW de resíduo.

Ver `grid_features.py` para a construção do vetor de features e para a
discussão das 6 features que dependem de estação (imputadas aqui).

Organização: processa um CLUSTER por vez, com a série temporal COMPLETA das
células daquele cluster. Isso é exigência do LSTM (venceu 29 dos 56 combos),
que precisa de janelas contíguas de `lookback` dias — fatiar por ano quebraria
a janela na virada. Dentro do cluster, cada trimestre é predito pelo seu
próprio modelo vencedor.

Convenções replicadas do caminho canônico (spatial_correction.py /
spatial_correction_dl.py), porque divergir aqui produz predição fora da
distribuição de treino:
  - clássicos (lazy/mlp): imputer -> scaler -> DataFrame nomeado -> predict;
    `target_kind="ratio"` multiplica pelo proxy ERA5 do dia-alvo.
  - LSTM (schema v2, `dl_metadata`): features base imputadas+escaladas,
    one-hots de cluster concatenados SEM escala, janela [D-L+1 .. D] da
    convenção única de `src/pipeline/data/windowing.py` (padding de borda no
    início da série), saída única já em m/s (sem razão × proxy). Artefatos
    dual-head antigos levantam `LegacyLSTMArtifactError` — retreinar.
  - ambos: clip físico [0, 80] m/s.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.climatology import get_harmonic_climatology
from src.inference.dl_metadata import METADATA_FILENAME, load_dl_metadata
from src.pipeline.data.splits import MonthBlockSplit
from src.pipeline.data.target import inverse_target
from src.pipeline.data.windowing import daily_window_positions
from src.pipelines.common import ERA5_GUST_PROXY, SEASONS, reject_inmet_derived_features

PIPELINE_ROOTS = {
    "lazy": Path("lazy_modal/lazy_clusters"),
    "mlp": Path("mlp_modal/mlp_clusters"),
    "lstm": Path("modal/experiments"),
}
CLIP_RANGE = (0.0, 80.0)


def _load_classic(models_dir: Path, cid: int, season: str | None):
    """Artefato joblib do cluster (por trimestre se existir, senão o pooled)."""
    for name in ([f"best_model_c{cid}_{season}.joblib"] if season else []) + [
        f"best_model_c{cid}.joblib"
    ]:
        p = models_dir / name
        if p.exists():
            art = joblib.load(p)
            reject_inmet_derived_features(art["features"], source=str(p))
            return art
    return None


def _predict_classic(art: dict, df: pd.DataFrame) -> np.ndarray:
    feats = art["features"]
    x = df.reindex(columns=feats)
    # Sem imputação: linha (célula × dia) com feature ausente — ex.: célula
    # fora da cobertura de uma feature nova — fica NaN.
    valid = x.notna().all(axis=1).to_numpy()
    out = np.full(len(x), np.nan)
    if not valid.any():
        return out
    x_df = pd.DataFrame(art["scaler"].transform(x[valid]), columns=feats)

    kind = art.get(
        "target_kind",
        "ratio" if art.get("model_name") == "MLPRegressor" else "absolute",
    )
    preds = np.asarray(art["model"].predict(x_df), dtype="float64").reshape(-1)
    if kind == "ratio":
        preds = preds * np.clip(df[ERA5_GUST_PROXY].to_numpy(float)[valid], 0.1, None)
    out[valid] = np.clip(preds, *CLIP_RANGE)
    return out


def _load_lstm(models_dir: Path, cid: int):
    """Modelo keras + metadados v2 do cluster. Metadados legados (dual-head)
    ou de modelo horário levantam erro — não há como aplicá-los na grade."""
    if not (models_dir / METADATA_FILENAME).exists():
        return None, None
    meta = load_dl_metadata(models_dir, require_resolution="daily")
    for name in (f"best_model_c{cid}.keras", f"best_model_c{cid}.h5"):
        p = models_dir / name
        if p.exists():
            from tensorflow import keras
            return keras.models.load_model(p, compile=False), meta
    return None, None


def _harmonic_clim_for_cells(df: pd.DataFrame, meta: dict,
                             n_cells: int, n_times: int) -> np.ndarray:
    """`era5_clim_wind` como o modelo a viu no treino: climatologia harmônica
    do proxy ERA5 ajustada nos dias "train" do split salvo — por célula, sobre
    a própria série da grade. Vale para as três pipelines (todas usam blocos de
    mês + harmônica); `grid_features` monta uma média por dia-do-ano sobre todo
    o período, que não é a mesma grandeza e é sempre sobrescrita aqui."""
    import xarray as xr

    times = pd.DatetimeIndex(df["time"].to_numpy()[:n_times])
    wind = df[ERA5_GUST_PROXY].to_numpy(float).reshape(n_cells, n_times)
    ds = xr.Dataset({ERA5_GUST_PROXY: (("cell", "time"), wind)}, coords={"time": times})
    split = MonthBlockSplit.from_dict(meta["split"])
    clim = get_harmonic_climatology(
        ds, ERA5_GUST_PROXY, times[split.label(times) == "train"],
        int(meta.get("climatology", {}).get("n_harmonics", 3)),
    )
    vals = clim.sel(dayofyear=np.asarray(times.dayofyear)).transpose("cell", "dayofyear")
    return np.asarray(vals.values).ravel()


def _with_training_climatology(df: pd.DataFrame, art: dict,
                               n_cells: int, n_times: int) -> pd.DataFrame:
    """`era5_clim_wind` reproduzida como o modelo clássico a viu no treino.

    lazy/mlp passaram a ajustar a climatologia por série harmônica sobre os
    dias de TREINO do split por blocos de mês. A versão que `grid_features`
    monta (média por dia-do-ano sobre todo o período) é outra grandeza --- usá-la
    poria o modelo fora da distribuição de treino, silenciosamente. Artefato
    que use a feature sem registrar a partição é recusado: não há como
    reproduzir o ajuste, e adivinhar seria o mesmo erro silencioso.
    """
    if "era5_clim_wind" not in art["features"]:
        return df
    if art.get("climatology", {}).get("method") != "harmonic" or not art.get("split"):
        raise ValueError(
            f"artefato do cluster {art.get('cluster_id')}/{art.get('season')} usa "
            "'era5_clim_wind' mas não registra a partição de treino "
            "('split'/'climatology') — retreine com a pipeline atual"
        )
    return df.assign(era5_clim_wind=_harmonic_clim_for_cells(df, art, n_cells, n_times))


def _predict_lstm(model, meta: dict, df: pd.DataFrame,
                  cid: int, n_cells: int, n_times: int) -> np.ndarray:
    lookback = int(meta["lookback"])
    base = list(meta["base_feature_names"])
    cluster_cols = list(meta["cluster_feature_names"])

    x = df.reindex(columns=base)
    if "era5_clim_wind" in base and meta.get("climatology", {}).get("method") == "harmonic":
        x["era5_clim_wind"] = _harmonic_clim_for_cells(df, meta, n_cells, n_times)
    x = meta["scaler_x"].transform(x)

    onehot = np.zeros((len(df), len(cluster_cols)), dtype="float32")
    for j, c in enumerate(cluster_cols):
        try:
            onehot[:, j] = float(int(c.split("_", 1)[1]) == cid)
        except (IndexError, ValueError):
            pass
    mat = np.hstack([x, onehot]).astype("float32")

    # (cell, time, feat) -> janelas por célula. Materializar as janelas de
    # TODAS as células de uma vez chega a ~4 GB no maior cluster (277 células
    # × 9132 dias × 7 × n_feats), então processa em blocos de células.
    mat = mat.reshape(n_cells, n_times, -1)
    # Janela [D-L+1 .. D] com padding de borda nos primeiros L-1 dias
    # (decisão do usuário: não reintroduzir um gap em jan/2000).
    pos, _ = daily_window_positions(n_times, lookback, pad="edge")
    CELL_CHUNK = 24

    partes = []
    for ini in range(0, n_cells, CELL_CHUNK):
        bloco = range(ini, min(ini + CELL_CHUNK, n_cells))
        windows = np.concatenate([mat[c][pos] for c in bloco], axis=0)
        # Sem imputação: janela com feature ausente fica NaN.
        valid = np.isfinite(windows).all(axis=(1, 2))
        pred = np.full(len(windows), np.nan)
        if valid.any():
            pred[valid] = np.ravel(model.predict(windows[valid], batch_size=4096, verbose=0))
        partes.append(pred)
        del windows
    return inverse_target(meta["scaler_y"], np.concatenate(partes), clip=True)


def predict_cluster(
    df_cluster: pd.DataFrame, cid: int, winners: pd.DataFrame,
    artifacts_root: str | Path, n_cells: int, n_times: int,
) -> np.ndarray:
    """Prediz todas as linhas (célula × dia) de um cluster, cada trimestre com
    seu vencedor. Retorna vetor alinhado a `df_cluster`."""
    root = Path(artifacts_root)
    preds = np.full(len(df_cluster), np.nan)

    for season in SEASONS:
        w = winners[(winners["cluster_id"].astype(int) == cid)
                    & (winners["season"] == season)]
        if w.empty:
            print(f"  [AVISO] cluster {cid}/{season}: sem vencedor — fica NaN")
            continue
        row = w.iloc[0]
        models_dir = root / PIPELINE_ROOTS[row["pipeline"]] / row["arm"] / "fitted_models"
        mask = (df_cluster["season"] == season).to_numpy()
        if not mask.any():
            continue

        if row["pipeline"] == "lstm":
            model, meta = _load_lstm(models_dir, cid)
            if model is None:
                print(f"  [AVISO] cluster {cid}/{season}: modelo LSTM ausente "
                      f"em {models_dir} — fica NaN")
                continue
            # O LSTM precisa da série contígua inteira (janelas), então prediz
            # todos os dias do cluster e só depois recorta o trimestre.
            todos = _predict_lstm(model, meta, df_cluster, cid, n_cells, n_times)
            preds[mask] = todos[mask]
        else:
            art = _load_classic(models_dir, cid, season)
            if art is None:
                print(f"  [AVISO] cluster {cid}/{season}: modelo ausente em "
                      f"{models_dir} — fica NaN")
                continue
            # A climatologia precisa ser a MESMA do treino. Ajustá-la exige a
            # série completa do cluster, então é calculada antes do recorte
            # por trimestre.
            df_pred = _with_training_climatology(df_cluster, art, n_cells, n_times)
            preds[mask] = _predict_classic(art, df_pred.loc[mask])

        print(f"  cluster {cid}/{season}: {row['pipeline']}/{row['arm']} "
              f"-> {int(mask.sum())} predições", flush=True)

    return preds
