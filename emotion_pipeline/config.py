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

EMOTION_LABELS_EMOTION_ORI = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Joy",
    4: "Sadness",
    5: "Surprise",
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

EMOTION_LABELS_EMOSET_NEW = {
    0: "Amusement",
    1: "Anger",
    2: "Contentment",
    3: "Disgust",
    4: "Excitement",
    5: "Fear",
    6: "Sadness",
    7: "Awe",
}

# Switch between datasets: "emotion6", "emotion_ori", "dvisa", or "emoset_new"
DATASET_NAME = "emoset_new"

# Training variant: "base", "extended", or "attention"
TRAIN_VARIANT = "extended"

# Task mode: "auto", "multitask", or "emotion_only"
TASK_MODE = "auto"
EMOTION_LABEL_SET = DATASET_NAME
if EMOTION_LABEL_SET == "emotion6":
    EMOTION_LABELS = EMOTION_LABELS_EMOTION6
elif EMOTION_LABEL_SET == "emotion_ori":
    EMOTION_LABELS = EMOTION_LABELS_EMOTION_ORI
elif EMOTION_LABEL_SET == "dvisa":
    EMOTION_LABELS = EMOTION_LABELS_DVISA
else:
    EMOTION_LABELS = EMOTION_LABELS_EMOSET_NEW

@dataclass(frozen=True)
class PathsEmotion6:
    img_root: Path = Path("/home/rdas/color_transfer/Emotion6_new")  # e.g., Emotion6/disgust/1.jpg
    train_csv: Path = Path("/home/rdas/color_transfer/Emotion6_new/emotion6_train_80.csv")
    test_csv: Path = Path("/home/rdas/color_transfer/Emotion6_new/emotion6_test_20.csv")


class PathsEmotionOri:
    img_root: Path = Path("/home/rdas/color_transfer/Emotion6")  # e.g., Emotion6/disgust/1.jpg
    train_csv: Path = Path("/home/rdas/color_transfer/Emotion6/emotion6_ori_train_80.csv")
    test_csv: Path = Path("/home/rdas/color_transfer/Emotion6/emotion6_ori_test_20.csv")


class PathsDVisa:
    img_root: Path = Path("/home/rdas/color_transfer/D-Visa/Abstract_Expressionism")
    train_csv: Path = Path("/home/rdas/color_transfer/D-Visa/D-ViSA_train_80.csv")
    test_csv: Path = Path("/home/rdas/color_transfer/D-Visa/D-ViSA_test_20.csv")


class PathsEmoSetNew:
    img_root: Path = Path("/home/rdas/color_transfer/emoset_new")
    train_csv: Path = Path("/home/rdas/color_transfer/emoset_new/emoset_train_60.csv")
    test_csv: Path = Path("/home/rdas/color_transfer/emoset_new/emoset_val_20.csv")  # Using val set for testing in this example


def get_paths():
    if DATASET_NAME == "dvisa":
        return PathsDVisa()
    if DATASET_NAME == "emoset_new":
        return PathsEmoSetNew()
    if DATASET_NAME == "emotion_ori":
        return PathsEmotionOri()
    return PathsEmotion6()


def get_task_mode() -> str:
    if TASK_MODE == "auto":
        return "emotion_only" if DATASET_NAME == "emoset_new" else "multitask"
    if TASK_MODE not in {"multitask", "emotion_only"}:
        raise ValueError("TASK_MODE must be 'auto', 'multitask', or 'emotion_only'")
    return TASK_MODE


def get_train_variant() -> str:
    if TRAIN_VARIANT not in {"base", "extended", "attention"}:
        raise ValueError("TRAIN_VARIANT must be 'base', 'extended', or 'attention'")
    return TRAIN_VARIANT


CHECKPOINT_ROOT = Path("checkpoints") / DATASET_NAME
OUTPUT_ROOT = Path("outputs") / DATASET_NAME


def get_checkpoint_dir(run_name: str) -> Path:
    """Get checkpoint directory for a specific run (no auto-creation)."""
    return CHECKPOINT_ROOT / run_name


def get_output_dir(category: str) -> Path:
    """Get output directory for a specific category (returns path, caller must create)."""
    return OUTPUT_ROOT / category


# Predefined output categories
OUTPUT_CATEGORIES = {
    "xai": "Explainability outputs (Grad-CAM, Integrated Gradients, SHAP, etc.)",
    "attention": "Attention pooling visualizations",
    "visualizations": "Model visualizations and plots",
    "attention_maps": "Raw attention module outputs",
    "metrics": "Evaluation metrics and results",
    "logs": "Training logs and outputs",
}

@dataclass(frozen=True)
class TrainConfig:

    backbone_name: str = "dinov2_vitb14" #"dinov2_vitl14", dinov2_vits14, dinov2_vitb14
    image_size: int = 224
    batch_size: int = 64  #32
    num_workers: int = 2  #4
    epochs_probe: int = 100  # 200 Reduced for faster training
    epochs_finetune: int = 20 #40 Reduced for faster training
    lr_head: float = 1e-4 if DATASET_NAME == "dvisa" else 5e-5
    lr_backbone: float = 5e-6 if DATASET_NAME == "dvisa" else 1e-6
    weight_decay: float = 0.1 if DATASET_NAME == "dvisa" else 0.3
    lam_va: float = 0.8 if DATASET_NAME == "dvisa" else 2.0
    dropout: float = 0.4 if DATASET_NAME == "dvisa" else 0.6
    va_dims: int = 3 if DATASET_NAME == "dvisa" else 2
    # Classifier head options
    num_emotions: int = len(EMOTION_LABELS)
    head_type: str = "linear"  # linear | mlp
    head_hidden_dim: int = 512
    head_dropout: float = 0.3
    early_stopping_patience: int = 8
    early_stop_lambda: float = 0.2 #0.3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
