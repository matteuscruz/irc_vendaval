from __future__ import annotations

import numpy as np


class Sequencer:
    """Converts flat arrays into overlapping lookback windows."""

    def __init__(self, lookback: int):
        self.lookback = lookback

    def make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(self.lookback, len(X)):
            xs.append(X[i - self.lookback : i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    def transform(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tr_s, y_tr_s = self.make_sequences(X_train, y_train)
        X_vl_s, y_vl_s = self.make_sequences(X_val, y_val)
        X_te_s, y_te_s = self.make_sequences(X_test, y_test)
        return X_tr_s, X_vl_s, X_te_s, y_tr_s, y_vl_s, y_te_s
