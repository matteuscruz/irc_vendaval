from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer


class DataImputer:
    def __init__(self, strategy: str = "median"):
        self._imp = SimpleImputer(strategy=strategy)
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._imp.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "DataImputer must be fit on training data before calling transform()."
            )
        return self._imp.transform(X)
