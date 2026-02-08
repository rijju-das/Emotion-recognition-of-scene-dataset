from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class Paths:
    img_root: Path = Path("/home/rdas/color_transfer/Emotion6_new")  # e.g., Emotion6/disgust/1.jpg
    train_csv: Path = Path("emotion6_train_80.csv")
    test_csv: Path = Path("emotion6_test_20.csv")

@dataclass(frozen=True)
class TrainConfig:

    backbone_name: str = "dinov2_vitl14"  # dinov2_vits14, dinov2_vitb14
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs_probe: int = 200  # Increased: still improving at epoch 60
    epochs_finetune: int = 40
    lr_head: float = 5e-5
    lr_backbone: float = 1e-6
    weight_decay: float = 0.5
    lam_va: float = 2.0
    dropout: float = 0.5
    early_stopping_patience: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
