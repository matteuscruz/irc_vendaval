from __future__ import annotations

import pandas as pd

from src.config.schema import FeatureEngineeringConfig


_SEASON_MAP = {
    12: 1, 1: 1, 2: 1,   # Summer
    3: 2,  4: 2, 5: 2,   # Fall
    6: 3,  7: 3, 8: 3,   # Winter
    9: 4, 10: 4, 11: 4,  # Spring
}


class FeatureEngineer:
    def __init__(self, cfg: FeatureEngineeringConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.cfg.add_month:
            date_col = self._find_date_col(df)
            if date_col:
                df["mes"] = pd.to_datetime(df[date_col]).dt.month
            elif df.index.dtype == "datetime64[ns]":
                df["mes"] = df.index.month

        if self.cfg.add_season_dummies:
            if "mes" not in df.columns:
                raise ValueError(
                    "add_season_dummies=True requires add_month=True or 'mes' column."
                )
            df["estacao_do_ano"] = df["mes"].map(_SEASON_MAP)
            dummies = pd.get_dummies(df["estacao_do_ano"], prefix="estacao")
            # ensure all 4 seasons are present even if some are absent in the split
            for i in range(1, 5):
                col = f"estacao_{i}"
                if col not in dummies.columns:
                    dummies[col] = False
            dummies = dummies[[f"estacao_{i}" for i in range(1, 5)]].astype(int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=["estacao_do_ano"])

        return df

    @staticmethod
    def _find_date_col(df: pd.DataFrame) -> str | None:
        _DATE_KEYWORDS = ("data", "date", "time", "dt_", "fecha")
        # 1. match by column name
        for col in df.columns:
            if any(kw in col.lower() for kw in _DATE_KEYWORDS):
                return col
        # 2. match by dtype (already parsed as datetime)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None
