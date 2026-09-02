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
  - LSTM: features base imputadas+escaladas, one-hots de cluster concatenados
    SEM escala, janela = os `lookback` dias ANTERIORES ao dia-alvo, cabeça
    dupla (normal/extrema por threshold), e reconstrução por RAZÃO
    (scaler_y.inverse * proxy ERA5 do dia-alvo).
  - ambos: clip físico [0, 80] m/s.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipelines.common import ERA5_GUST_PROXY, SEASONS

PIPELINE_ROOTS = {
    "lazy": Path("lazy_modal/lazy_clusters"),
    "mlp": Path("mlp_modal/mlp_clusters"),
    "lstm": Path("modal/experiments"),
}
CLIP_RANGE = (0.0, 80.0)

# `era5_clim_wind` tem PROVENIÊNCIA DIFERENTE por pipeline:
#   cluster_lazy.py:944       -> get_climatology(ds_inmet, TARGET_VAR)  (observação)
#   cluster_mlp.py:140 / LSTM -> get_climatology(ds_era5, ERA5_GUST_PROXY)
# grid_features.py produz a versão ERA5 (a única derivável numa célula), que
# bate exatamente com o que mlp/lstm viram no treino — validado, erro 0.0000.
# Para o lazy essa coluna é outra grandeza (climatologia da rajada observada,
# sistematicamente maior que o vento ERA5: erro relativo 62%, corr 0.59 na
# validação), então alimentá-la seria pôr o modelo fora da distribuição de
# treino — pior que deixar o imputer preencher. Só para lazy, vai NaN.
LAZY_ONLY_BLANK = ["era5_clim_wind"]


def _load_classic(models_dir: Path, cid: int, season: str | None):
    """Artefato joblib do cluster (por trimestre se existir, senão o pooled)."""
    for name in ([f"best_model_c{cid}_{season}.joblib"] if season else []) + [
        f"best_model_c{cid}.joblib"
    ]:
        p = models_dir / name
        if p.exists():
            return joblib.load(p)
    return None


def _predict_classic(art: dict, df: pd.DataFrame,
                     blank: list[str] | None = None) -> np.ndarray:
    feats = art["features"]
    x = df.reindex(columns=feats)          # features ausentes viram NaN
    for col in (blank or []):              # ver LAZY_ONLY_BLANK
        if col in x.columns:
            x[col] = np.nan
    x_imp = art["imputer"].transform(x)
    x_scaled = art["scaler"].transform(x_imp)
    x_df = pd.DataFrame(x_scaled, columns=feats)

    kind = art.get(
        "target_kind",
        "ratio" if art.get("model_name") == "MLPRegressor" else "absolute",
    )
    preds = np.asarray(art["model"].predict(x_df), dtype="float64").reshape(-1)
    if kind == "ratio":
        preds = preds * np.clip(df[ERA5_GUST_PROXY].to_numpy(float), 0.1, None)
    return np.clip(preds, *CLIP_RANGE)


def _load_lstm(models_dir: Path, cid: int):
    """Modelo keras + metadados (scalers/lookback/feature_names) do cluster."""
    meta_path = models_dir / "dl_metadata.joblib"
    if not meta_path.exists():
        return None, None
    for name in (f"best_model_c{cid}.keras", f"best_model_c{cid}.h5"):
        p = models_dir / name
        if p.exists():
            from tensorflow import keras
            return keras.models.load_model(p, compile=False), joblib.load(meta_path)
    return None, None


def _build_windows(mat: np.ndarray, lookback: int) -> np.ndarray:
    """Janelas deslizantes de uma célula: saída (n_dias, lookback, n_feats).

    A janela do dia i são os `lookback` dias ANTERIORES (mesma convenção de
    `_make_windows`/`predict_stations`). Nos primeiros `lookback` dias da
    série não há janela completa — replica-se a borda (padding), decisão
    explícita do usuário para não reintroduzir um gap de 7 dias em jan/2000.
    """
    padded = np.concatenate([np.repeat(mat[:1], lookback, axis=0), mat], axis=0)
    idx = np.arange(len(mat))[:, None] + np.arange(lookback)[None, :]
    return padded[idx]


def _predict_lstm(model, meta: dict, df: pd.DataFrame,
                  cid: int, n_cells: int, n_times: int) -> np.ndarray:
    lookback = meta["lookback"]
    features = meta["feature_names"]
    base = [f for f in features if not str(f).startswith("cluster_")]
    cluster_cols = [f for f in features if str(f).startswith("cluster_")]

    x = df.reindex(columns=base)
    if meta.get("imputer_x") is not None:
        x = meta["imputer_x"].transform(x)
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
    thr = meta.get("extreme_threshold", np.inf)
    CELL_CHUNK = 24

    partes = []
    for ini in range(0, n_cells, CELL_CHUNK):
        bloco = range(ini, min(ini + CELL_CHUNK, n_cells))
        windows = np.concatenate(
            [_build_windows(mat[c], lookback) for c in bloco], axis=0
        )
        out = model.predict(windows, batch_size=4096, verbose=0)
        if isinstance(out, list):
            normal = out[0].reshape(-1)
            extreme = out[1].reshape(-1)
            partes.append(np.where(normal > thr, extreme, normal))
        else:
            partes.append(np.asarray(out).reshape(-1))
        del windows
    pred = np.concatenate(partes)

    pred = meta["scaler_y"].inverse_transform(pred.reshape(-1, 1)).ravel()
    proxy = np.clip(df[ERA5_GUST_PROXY].to_numpy(float), 0.1, None)
    return np.clip(pred * proxy, *CLIP_RANGE)


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
            blank = LAZY_ONLY_BLANK if row["pipeline"] == "lazy" else None
            preds[mask] = _predict_classic(art, df_cluster.loc[mask], blank)

        print(f"  cluster {cid}/{season}: {row['pipeline']}/{row['arm']} "
              f"-> {int(mask.sum())} predições", flush=True)

    return preds
