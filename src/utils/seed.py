from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force CPU only when explicitly requested (e.g. local dev).
    # On Modal/GPU environments, leave CUDA_VISIBLE_DEVICES untouched.
    if os.environ.get("FORCE_CPU", "0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Op-level determinism trades speed for reproducibility; opt-in only.
        if os.environ.get("TF_DETERMINISTIC", "0") == "1":
            tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
