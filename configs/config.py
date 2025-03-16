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
class ModelArchitectureConfig:
    TYPE: ClassVar[str] = "None"

@dataclasses.dataclass
class ClassicCNNArchitectureConfig(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "ClassicCNN"
    base_dim: int = 32

@dataclasses.dataclass
class ResNetCNNArchitectureConfig(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "ResNetCNN"
    base_dim: int = 32

@dataclasses.dataclass
class VGGlikeArchitectureConfig(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "VGGlike"
    base_dim: int = 32

@dataclasses.dataclass
class PretrainedCNNArchitectureConfig(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "PretrainedCNN"
    base_pretrained_model: Literal["EfficientNetB0"] = "EfficientNetB0"
    fine_tuning: bool = False
    mlp_hidden_dim: int = 32

@dataclasses.dataclass
class ViT(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "ViT"

@dataclasses.dataclass
class PretrainedViTArchitectureConfig(ModelArchitectureConfig):
    TYPE: ClassVar[str] = "PretrainedViT"
    base_pretrained_model: Literal["b_16"] = "b_16"
    fine_tuning: bool = False
    mlp_hidden_dim: int = 32

@dataclasses.dataclass
class ModelConfig:
    model_architecture: ModelArchitectureConfig
    resume_path: Optional[str] = None


@dataclasses.dataclass
class DataConfig:
    num_classes: int = 10
    in_channels: int = 3
    root: str = "data"
    subset_size: Optional[int] = None


@dataclasses.dataclass
class Config(JSONPyWizard):
    # Config for training option
    training: TrainingConfig

    # Config for model option
    model: ModelConfig

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
