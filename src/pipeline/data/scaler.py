from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    def __init__(self):
        self._sc = StandardScaler()
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        result = self._sc.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler must be fit on training data before calling transform()."
            )
        return self._sc.transform(X)


class TargetScaler:
    def __init__(self):
        self._sc = StandardScaler()
        self._fitted = False

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        result = self._sc.fit_transform(y.reshape(-1, 1)).ravel()
        self._fitted = True
        return result

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TargetScaler must be fit on training data before calling transform()."
            )
        return self._sc.transform(y.reshape(-1, 1)).ravel()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._sc.inverse_transform(y.reshape(-1, 1)).ravel()
