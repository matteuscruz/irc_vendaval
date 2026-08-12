from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.schema import DataConfig, PreprocessingConfig
from src.pipeline.data.features import FeatureEngineer
from src.pipeline.data.imputer import DataImputer
from src.pipeline.data.scaler import FeatureScaler, TargetScaler
from src.pipeline.data.sequence import Sequencer
from src.pipeline.data.splitter import SpatialSplitter, TemporalSplitterByStation


@dataclass
class DataBatch:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_test_original: np.ndarray  # desnormalizado, para métricas
    target_scaler: TargetScaler
    n_features: int


class Preprocessor:
    def __init__(self, prep_cfg: PreprocessingConfig, data_cfg: DataConfig):
        self.prep_cfg = prep_cfg
        self.data_cfg = data_cfg

        self.fe = FeatureEngineer(prep_cfg.feature_engineering)
        self.spatial_splitter = SpatialSplitter()
        self.temporal_splitter = TemporalSplitterByStation()
        self.imputer = DataImputer(prep_cfg.imputer_strategy)
        self.feat_sc = FeatureScaler()
        self.tgt_sc = TargetScaler()
        self.seq = Sequencer(prep_cfg.lookback)

    def fit_transform(self, df: pd.DataFrame) -> DataBatch:
        df = self.fe.transform(df)

        # spatial split: holdout station(s) → test
        train_df, test_df = self.spatial_splitter.split(df, self.data_cfg)

        # temporal split per station: 80/20
        tr_df, val_df = self.temporal_splitter.split(
            train_df, self.prep_cfg, self.data_cfg.station_col
        )

        X_tr_raw, y_tr = self._split_xy(tr_df)
        X_vl_raw, y_vl = self._split_xy(val_df)
        X_te_raw, y_te = self._split_xy(test_df)

        # FIT on train only, then transform all splits
        X_tr = self.feat_sc.fit_transform(self.imputer.fit_transform(X_tr_raw))
        X_vl = self.feat_sc.transform(self.imputer.transform(X_vl_raw))
        X_te = self.feat_sc.transform(self.imputer.transform(X_te_raw))

        y_tr_sc = self.tgt_sc.fit_transform(y_tr)
        y_vl_sc = self.tgt_sc.transform(y_vl)
        y_te_sc = self.tgt_sc.transform(y_te)

        X_tr_s, X_vl_s, X_te_s, y_tr_s, y_vl_s, y_te_s = self.seq.transform(
            X_tr, X_vl, X_te, y_tr_sc, y_vl_sc, y_te_sc
        )

        # y_test_original: align with sequenced test (drop first `lookback` rows)
        y_te_orig = y_te[self.prep_cfg.lookback :]

        return DataBatch(
            X_train=X_tr_s,
            X_val=X_vl_s,
            X_test=X_te_s,
            y_train=y_tr_s,
            y_val=y_vl_s,
            y_test=y_te_s,
            y_test_original=y_te_orig,
            target_scaler=self.tgt_sc,
            n_features=X_tr_s.shape[2],
        )

    _DATE_KEYWORDS = ("data", "date", "time", "dt_", "fecha")

    def _split_xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols_to_drop = set(
            [self.data_cfg.target]
            + self.data_cfg.drop_cols
            + [
                c for c in df.columns
                if any(kw in c.lower() for kw in self._DATE_KEYWORDS)
                or not pd.api.types.is_numeric_dtype(df[c])
            ]
        )
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        X = df[feature_cols].values.astype(np.float32)
        y = df[self.data_cfg.target].values.astype(np.float32)
        return X, y
