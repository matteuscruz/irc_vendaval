from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExperimentMeta(BaseModel):
    name: str
    seed: int = 42
    output_dir: str = "artifacts/experiments"


class DataConfig(BaseModel):
    path: str
    target: str
    station_col: str
    test_stations: list[str]
    train_stations: list[str]
    drop_cols: list[str] = Field(default_factory=list)


class FeatureEngineeringConfig(BaseModel):
    add_month: bool = True
    add_season_dummies: bool = True


class PreprocessingConfig(BaseModel):
    temporal_split_ratio: float = 0.8
    lookback: int = 7
    imputer_strategy: str = "median"
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )


class ModelConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class EarlyStoppingConfig(BaseModel):
    monitor: str = "val_loss"
    patience: int = 12
    restore_best_weights: bool = True


class ReduceLRConfig(BaseModel):
    monitor: str = "val_loss"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6


class CheckpointConfig(BaseModel):
    enabled: bool = True
    monitor: str = "val_loss"


class FinetuneConfig(BaseModel):
    loss_name: str = "pinball"
    loss_params: dict = Field(default_factory=lambda: {"tau": 0.90})
    learning_rate: float = 1e-5
    max_epochs: int = 300
    patience: int = 20


class SampleWeightConfig(BaseModel):
    enabled: bool = False
    percentile: float = 90.0
    extreme_weight: float = 5.0


class OversamplingConfig(BaseModel):
    enabled: bool = False
    percentile: float = 90.0
    repeat: int = 3


class TrainingConfig(BaseModel):
    strategy: Literal["standard", "bagging", "finetune"] = "standard"
    batch_size: int = 56
    max_epochs: int = 1000
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    reduce_lr: ReduceLRConfig = Field(default_factory=ReduceLRConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    sample_weight: SampleWeightConfig = Field(default_factory=SampleWeightConfig)
    oversampling: OversamplingConfig = Field(default_factory=OversamplingConfig)


class LossConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    percentiles: list[float] = Field(default_factory=lambda: [0.90, 0.95])
    metrics: list[str] = Field(default_factory=lambda: ["r2", "rmse", "mae", "bias"])


class VisualizationConfig(BaseModel):
    enabled: bool = True
    save_dir: str = "artifacts/plots"
    plots: list[str] = Field(default_factory=list)


class SweepsConfig(BaseModel):
    enabled: bool = False
    model: list[str] = Field(default_factory=list)
    seed: list[int] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    version: int = 1
    experiment: ExperimentMeta
    data: DataConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    losses: list[LossConfig]
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    sweeps: SweepsConfig = Field(default_factory=SweepsConfig)
