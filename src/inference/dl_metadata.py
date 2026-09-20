"""Metadados da LSTM (`fitted_models/dl_metadata.joblib`) — schema v2.

A LSTM dual-head (schema implícito v1) salvava só scalers, features, lookback e
`extreme_threshold`; os consumidores tinham que ADIVINHAR o resto (alvo em
razão, janela excluindo o dia-alvo, período da climatologia) e cada um
hardcodava uma suposição diferente. O v2 registra tudo que a inferência precisa
reproduzir, e `load_dl_metadata` recusa artefatos antigos com um erro claro em
vez de deixá-los produzir predição errada em silêncio.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.pipeline.data.splits import MonthBlockSplit
from src.pipeline.data.target import TARGET_CLIP
from src.pipeline.data.windowing import WindowSpec
from src.pipelines.common import TARGET_VAR, reject_inmet_derived_features

SCHEMA_VERSION = 2
MODEL_NAME = "cluster_lstm"
LEGACY_MODEL_NAMES = {"cluster_dual_head_lstm", "cluster_tr_lstm"}
METADATA_FILENAME = "dl_metadata.joblib"


class LegacyLSTMArtifactError(RuntimeError):
    """Artefato da LSTM dual-head (alvo em razão × ERA5) — não compatível com os
    consumidores atuais. Precisa retreinar."""


class UnsupportedResolutionError(RuntimeError):
    """Resolução do modelo diferente da exigida pelo consumidor (ex.: modelo
    horário na grade diária — não existe ERA5 horário em grade)."""


def build_dl_metadata(
    *,
    scaler_x,
    scaler_y,
    feature_names: list[str],
    window: WindowSpec,
    split: MonthBlockSplit,
    interp_method: str,
    climatology: dict,
    hourly: dict | None,
    model_params: dict,
    seed: int,
    target_var: str = TARGET_VAR,
) -> dict:
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = None
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "target_kind": "absolute",
        "target_var": target_var,
        "target_units": "m/s",
        "output_clip": list(TARGET_CLIP),
        "resolution": window.resolution,
        "window": window.to_dict(),
        "lookback": window.length,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_names": list(feature_names),
        "base_feature_names": [f for f in feature_names if not str(f).startswith("cluster_")],
        "cluster_feature_names": [f for f in feature_names if str(f).startswith("cluster_")],
        "split": split.to_dict(),
        "climatology": dict(climatology),
        "interp_method": interp_method,
        "hourly": dict(hourly) if hourly else None,
        "model_params": dict(model_params),
        "seed": int(seed),
        "tf_version": tf_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def validate_dl_metadata(
    meta: dict, require_resolution: str | None = None, source: str = "",
) -> dict:
    where = f" ({source})" if source else ""
    if (
        meta.get("schema_version", 1) < SCHEMA_VERSION
        or "target_kind" not in meta
        or meta.get("model_name") in LEGACY_MODEL_NAMES
    ):
        raise LegacyLSTMArtifactError(
            f"Artefato LSTM legado{where}: model_name={meta.get('model_name')!r}, "
            f"schema_version={meta.get('schema_version', 1)}. A LSTM dual-head "
            "(alvo em razão × ERA5) foi substituída pela LSTM de saída única em "
            "m/s — retreine o braço com a config atual (schema v2)."
        )
    reject_inmet_derived_features(meta.get("base_feature_names", []), source=source)
    if require_resolution is not None and meta["resolution"] != require_resolution:
        raise UnsupportedResolutionError(
            f"Modelo com resolução {meta['resolution']!r}{where}, mas este "
            f"consumidor exige {require_resolution!r}."
        )
    return meta


def load_dl_metadata(models_dir: str | Path, require_resolution: str | None = None) -> dict:
    path = Path(models_dir) / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{METADATA_FILENAME} não encontrado em {models_dir}")
    return validate_dl_metadata(joblib.load(path), require_resolution, source=str(path))
