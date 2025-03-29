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
class WandbConfig:
    name: str = None
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CNNArchitectureConfig:
    architecture: Literal["Classic", "VGGlike", "ResNet", "ResNetDeep"] = "ResNet"
    base_dim: int = 32
    batch_normalization: bool = False
    dropout: float = 0.0

@dataclasses.dataclass
class EnsembleModelConfig:
    voting: Literal['soft', 'hard', 'stacking'] = 'soft'


@dataclasses.dataclass
class ModelConfig:
    architecture: Literal["CNN", "ViT", "Pretrained", "Ensemble"] = "CNN"
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
    augmentation: Literal["BasicTransform", "BasicColors", "AutoAugment", "All"] | None = None
    mix_augmentations: bool = False


@dataclasses.dataclass
class SweepConfig:
    name: Optional[str] = None
    config: str = ""
    project_name: str = ""


@dataclasses.dataclass
class Config(JSONPyWizard):
    # Config for training option
    training: TrainingConfig

    # Config for model option
    model: ModelConfig
    pretrained_model: PretrainedModelConfig
    cnn: CNNArchitectureConfig
    ensemble: EnsembleModelConfig

    # Config for data option
    data: DataConfig

    wandb: WandbConfig

    # Config for evaluation option
    evaluation: EvalConfig

    sweep: SweepConfig
    project_dir: str = "project"
    log_dir: str = "logs"
    project_tracker: List[str] = dataclasses.field(default_factory=lambda: ["wandb"])
    mixed_precision: str = "no"
    seed: int = 0
    config: Optional[str] = None
