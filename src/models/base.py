from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelBuilder(ABC):
    @abstractmethod
    def build(self, n_features: int, lookback: int, loss_fn: Any):
        """Instantiate and compile a model ready for training."""
        raise NotImplementedError
