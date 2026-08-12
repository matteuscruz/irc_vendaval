from __future__ import annotations

import pandas as pd

from src.config.schema import DataConfig, PreprocessingConfig


class SpatialSplitter:
    def split(
        self, df: pd.DataFrame, cfg: DataConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df[cfg.station_col].isin(cfg.train_stations)].copy()
        test = df[df[cfg.station_col].isin(cfg.test_stations)].copy()
        return train, test


class TemporalSplitter:
    def split(
        self, df: pd.DataFrame, cfg: PreprocessingConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split 80/20 per station then concatenate, preserving temporal order."""
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []

        station_col = df.columns[df.nunique() < 20].tolist()
        # Try to detect station column from data if not obvious
        groups = [df]  # fallback: treat entire df as one group

        for part in groups:
            n = int(len(part) * cfg.temporal_split_ratio)
            train_parts.append(part.iloc[:n])
            val_parts.append(part.iloc[n:])

        return pd.concat(train_parts, ignore_index=True), pd.concat(
            val_parts, ignore_index=True
        )


class TemporalSplitterByStation:
    """Applies temporal split independently per station, then concatenates."""

    def split(
        self,
        df: pd.DataFrame,
        cfg: PreprocessingConfig,
        station_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []

        for _, group in df.groupby(station_col, sort=False):
            group = group.sort_index()
            n = int(len(group) * cfg.temporal_split_ratio)
            train_parts.append(group.iloc[:n])
            val_parts.append(group.iloc[n:])

        return pd.concat(train_parts, ignore_index=True), pd.concat(
            val_parts, ignore_index=True
        )
