"""Interface comum das fontes de dados da LSTM: valida o YAML (schema v2) e
despacha para a fonte diária (`ClusterPreprocessor`) ou horária
(`build_hourly_batch`) conforme `data.resolution`."""
from __future__ import annotations

from src.data.interp import METHODS
from src.inference.dl_metadata import MODEL_NAME
from src.pipeline.data.cluster_preprocessor import ClusterDataBatch

RESOLUTIONS = ("daily", "hourly")
REMOVED_DATA_KEYS = ("train_slice", "val_slice", "test_slice", "test_station_fraction")
# Features de observação INMET foram removidas (não existem na grade ERA5).
REMOVED_OBSERVATION_KEYS = ("exclude_observation_features", "blank_cross_split_observations")
# O GAN/difusão (data augmentation) saiu do repositório: a config não pode mais pedi-lo.
REMOVED_TOP_LEVEL_KEYS = ("augmentation",)
REMOVED_MODEL_PARAMS = (
    "l2_reg", "weight_normal", "weight_extreme", "extreme_weight", "extreme_threshold",
    "static_hidden", "dropout_static", "recurrent_dropout",
)


def validate_lstm_config(cfg: dict) -> dict:
    """Falha cedo (antes de carregar dado) em config da LSTM dual-head/TR ou
    com valor inválido — chaves antigas não podem ser ignoradas em silêncio,
    senão um YAML velho treinaria com um split diferente do que diz."""
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    params = model.get("params") or {}
    errors: list[str] = []

    removed = [f"data.{k}" for k in REMOVED_DATA_KEYS if k in data]
    removed += [f"model.params.{k}" for k in REMOVED_MODEL_PARAMS if k in params]
    if removed:
        errors.append(
            f"chaves removidas na LSTM v2: {removed} — o split agora é data.split "
            "(blocos de mês) e o modelo é LSTM → Dropout → Dense(1) com Huber"
        )
    removed_top = [k for k in REMOVED_TOP_LEVEL_KEYS if k in cfg]
    if removed_top:
        errors.append(
            f"chaves removidas: {removed_top} — a geração de dados sintéticos (GAN/difusão) "
            "saiu deste repositório; remova o bloco do YAML"
        )
    removed_obs = [f"data.{k}" for k in REMOVED_OBSERVATION_KEYS if k in data]
    if removed_obs:
        errors.append(
            f"chaves removidas: {removed_obs} — features de observação INMET "
            "não são mais usadas (INMET entra só como alvo)"
        )
    name = model.get("name", MODEL_NAME)
    if name != MODEL_NAME:
        errors.append(f"model.name={name!r}; o único modelo suportado é {MODEL_NAME!r}")
    resolution = data.get("resolution", "daily")
    if resolution not in RESOLUTIONS:
        errors.append(f"data.resolution={resolution!r} (válidos: {RESOLUTIONS})")
    interp = data.get("interp_method", "nearest")
    if interp not in METHODS:
        errors.append(f"data.interp_method={interp!r} (válidos: {METHODS})")
    clim = (data.get("climatology") or {}).get("method", "harmonic")
    if clim != "harmonic":
        errors.append(f"data.climatology.method={clim!r}; só 'harmonic' é suportado")
    split = data.get("split") or {}
    if split.get("scheme", "month_block") != "month_block":
        errors.append(f"data.split.scheme={split.get('scheme')!r}; só 'month_block'")
    if resolution == "hourly" and data.get("feature_groups"):
        errors.append("data.feature_groups não se aplica à fonte horária (features vêm de data.hourly)")
    if split.get("purge", "strict") != "strict":
        errors.append(f"data.split.purge={split.get('purge')!r}; só 'strict'")
    if errors:
        raise ValueError("Config LSTM inválida:\n - " + "\n - ".join(errors))
    return cfg


def build_lstm_batch(cfg: dict) -> ClusterDataBatch:
    exp_cfg = cfg.get("experiment") or {}
    data_cfg = cfg["data"]
    prep_cfg = cfg.get("preprocessing") or {}
    common = dict(
        split_cfg=data_cfg.get("split"),
        seed=int(exp_cfg.get("seed", 42)),
        target_var=data_cfg.get("target_var", "daily_wind_gust_max"),
    )
    resolution = data_cfg.get("resolution", "daily")

    if resolution == "daily":
        from src.pipeline.data.cluster_preprocessor import ClusterPreprocessor

        return ClusterPreprocessor(
            data_cfg["raw_dir"],
            data_cfg["shp_dir"],
            lookback=int(prep_cfg.get("lookback", 7)),
            feature_groups=data_cfg.get("feature_groups"),
            interp_method=data_cfg.get("interp_method", "nearest"),
            n_harmonics=int((data_cfg.get("climatology") or {}).get("n_harmonics", 3)),
            **common,
        ).run()

    if resolution == "hourly":
        from src.pipeline.data.hourly_builder import build_hourly_batch

        return build_hourly_batch(
            data_cfg["raw_dir"], data_cfg["shp_dir"],
            hourly_cfg=data_cfg.get("hourly"), **common,
        )

    raise ValueError(f"data.resolution={resolution!r} (válidos: {RESOLUTIONS})")
