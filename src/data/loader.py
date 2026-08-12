from __future__ import annotations

import pandas as pd

from src.config.schema import DataConfig


class DataLoader:
    def load(self, cfg: DataConfig) -> pd.DataFrame:
        df = pd.read_csv(cfg.path, parse_dates=True)
        return df
