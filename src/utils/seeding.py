"""Semente global única para random, numpy e TensorFlow.

Antes desta função não havia semente em lugar nenhum do pipeline LSTM
(`experiment.seed` do YAML só chegava ao ClusterPreprocessor, que não a usava),
então dois treinos com a mesma config davam modelos diferentes.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Fixa as sementes de random/numpy/TensorFlow.

    `deterministic=True` também liga `enable_op_determinism()` do TF —
    reprodutibilidade bit a bit, ao custo de kernels mais lentos em GPU.
    `PYTHONHASHSEED` só vale para subprocessos (o hash do processo atual já foi
    semeado na inicialização), mas fica exportado para os que forem criados.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
