from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class LossConfig:
    class_weights: Optional[torch.Tensor] = None
    label_smoothing: float = 0.0

@dataclass(frozen=True)
class PhaseConfig:
    name: str
    epochs: int
    lr_head: float
    lr_backbone: float
    weight_decay: float
    freeze_backbone: bool
    eta_min: float

@dataclass(frozen=True)
class TrainingPlan:
    probe: PhaseConfig
    finetune: PhaseConfig
