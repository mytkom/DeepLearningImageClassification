from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional, ClassVar

from dataclass_wizard import JSONPyWizard


@dataclasses.dataclass
class TrainingConfig:
    engine: str = "engine"
    label_smoothing: float = 0.0
    batch_size: int = 32
    val_freq: int = 1
    epochs: int = 50
    num_workers: int = 4
    accum_iter: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    lr: float = 0.0003
    weight_decay: float = 0.0001


@dataclasses.dataclass
class EvalConfig:
    num_workers: int = 4
    batch_size: int = 32


@dataclasses.dataclass
class CNNArchitectureConfig:
    architecture: Literal["Classic", "VGGlike", "ResNet"] = "ResNet"
    base_dim: int = 32
    batch_normalization: bool = False
    dropout: float = 0.0


@dataclasses.dataclass
class ModelConfig:
    architecture: Literal["CNN", "Pretrained"] = "Pretrained"
    resume_path: Optional[str] = None


@dataclasses.dataclass
class PretrainedModelConfig:
    model_name: str = 'efficientnet_b0.ra_in1k'
    freeze_pretrained: bool = False


@dataclasses.dataclass
class DataConfig:
    num_classes: int = 10
    in_channels: int = 3
    image_size: int = 32
    root: str = "data"
    subset_size: Optional[int] = None
    augmentation: Literal["BasicTransform", "BasicColors", "AutoAugment"] | None = None


@dataclasses.dataclass
class Config(JSONPyWizard):
    # Config for training option
    training: TrainingConfig

    # Config for model option
    model: ModelConfig
    pretrained_model: PretrainedModelConfig
    cnn: CNNArchitectureConfig

    # Config for data option
    data: DataConfig

    # Config for evaluation option
    evaluation: EvalConfig

    project_dir: str = "project"
    log_dir: str = "logs"
    project_tracker: List[str] = dataclasses.field(default_factory=lambda: ["wandb"])
    mixed_precision: str = "no"
    seed: int = 0
    config: Optional[str] = None
