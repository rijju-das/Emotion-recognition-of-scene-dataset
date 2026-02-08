from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    img_root: Path = Path("/home/rdas/color_transfer/Emotion6_new")  # e.g., Emotion6/disgust/1.jpg
    train_csv: Path = Path("emotion6_train_80.csv")
    test_csv: Path = Path("emotion6_test_20.csv")

@dataclass(frozen=True)
class TrainConfig:
    backbone_name: str = "dinov2_vitb14"  # try "dinov2_vits14" if needed
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs_probe: int = 30
    epochs_finetune: int = 100
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    lam_va: float = 1.0
    seed: int = 42
    device: str = "cuda" #"cuda"  # or "cpu"
