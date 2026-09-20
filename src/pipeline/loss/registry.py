from __future__ import annotations

from typing import Any, Callable

from src.config.schema import LossConfig

_LOSS_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _LOSS_REGISTRY[name] = fn
        return fn

    return decorator


@_register("mse")
def _mse() -> str:
    return "mse"


@_register("mae")
def _mae() -> str:
    return "mae"


@_register("huber")
def _huber(delta: float = 1.0):
    import tensorflow as tf

    return tf.keras.losses.Huber(delta=delta)


@_register("pinball")
def _pinball(tau: float):
    import tensorflow as tf

    def loss(y_true, y_pred):
        e = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1.0) * e))

    loss.__name__ = f"pinball_tau{tau}"
    return loss


@_register("combined")
def _combined(components: list[dict]) -> Callable:
    fns_weights = []
    for c in components:
        c = dict(c)
        weight = float(c.pop("weight"))
        name = c.pop("name")
        sub_cfg = LossConfig(name=name, params=c)
        fn = LossRegistry.build(sub_cfg)
        fns_weights.append((fn, weight))

    import tensorflow as tf

    def loss(y_true, y_pred):
        total = tf.constant(0.0)
        for fn, w in fns_weights:
            val = fn(y_true, y_pred) if callable(fn) else tf.keras.losses.get(fn)(y_true, y_pred)
            total = total + w * tf.cast(val, tf.float32)
        return total

    loss.__name__ = "combined_loss"
    return loss


class LossRegistry:
    @staticmethod
    def build(cfg: LossConfig) -> Any:
        fn = _LOSS_REGISTRY.get(cfg.name)
        if fn is None:
            available = list(_LOSS_REGISTRY)
            raise ValueError(
                f"Loss '{cfg.name}' not registered. Available: {available}"
            )
        return fn(**cfg.params)

    @staticmethod
    def available() -> list[str]:
        return list(_LOSS_REGISTRY)
