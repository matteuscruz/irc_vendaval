from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainResult:
    history: dict
    y_pred_scaled: np.ndarray


class BaseTrainer(ABC):
    @abstractmethod
    def fit(self, data) -> TrainResult:
        """Train model and return predictions on test set (scaled)."""
        raise NotImplementedError
