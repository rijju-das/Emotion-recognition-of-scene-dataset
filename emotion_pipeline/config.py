from dataclasses import dataclass
from pathlib import Path
import torch

EMOTION_LABELS_EMOTION6 = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Joy",
    4: "Neutral",
    5: "Sadness",
}

EMOTION_LABELS_DVISA = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Amusement",
    4: "Awe",
    5: "Sadness",
    6: "Excitement",
    7: "Contentment",
}

# Switch between datasets: "emotion6" or "dvisa"
DATASET_NAME = "emotion6"
EMOTION_LABEL_SET = DATASET_NAME
EMOTION_LABELS = EMOTION_LABELS_EMOTION6 if EMOTION_LABEL_SET == "emotion6" else EMOTION_LABELS_DVISA

@dataclass(frozen=True)
class PathsEmotion6:
    img_root: Path = Path("/home/rdas/color_transfer/Emotion6_new")  # e.g., Emotion6/disgust/1.jpg
    train_csv: Path = Path("emotion6_train_80.csv")
    test_csv: Path = Path("emotion6_test_20.csv")


class PathsDVisa:
    img_root: Path = Path("/home/rdas/color_transfer/D-Visa/Abstract_Expressionism")
    train_csv: Path = Path("/home/rdas/color_transfer/D-Visa/D-ViSA_train_80.csv")
    test_csv: Path = Path("/home/rdas/color_transfer/D-Visa/D-ViSA_test_20.csv")


def get_paths():
    if DATASET_NAME == "dvisa":
        return PathsDVisa()
    return PathsEmotion6()

@dataclass(frozen=True)
class TrainConfig:

    backbone_name: str = "dinov2_vitb14" #"dinov2_vitl14", dinov2_vits14, dinov2_vitb14
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs_probe: int = 200  # Increased: still improving at epoch 60
    epochs_finetune: int = 40
    lr_head: float = 5e-5
    lr_backbone: float = 1e-6
    weight_decay: float = 0.3 #0.5
    lam_va: float = 2.0
    dropout: float = 0.6  #0.5
    # Classifier head options
    num_emotions: int = len(EMOTION_LABELS)
    head_type: str = "linear"  # linear | mlp
    head_hidden_dim: int = 512
    head_dropout: float = 0.3
    early_stopping_patience: int = 8
    early_stop_lambda: float = 0.2 #0.3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
