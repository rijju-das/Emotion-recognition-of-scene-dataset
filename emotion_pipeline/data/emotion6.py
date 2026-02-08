import os
import pandas as pd
from PIL import Image
import torch
from .base import BaseDataset

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}

class Emotion6Dataset(BaseDataset):
    """
    Reads:
      - filename (e.g., disgust/1.jpg)
      - valence_scaled, arousal_scaled (preferred) OR valence, arousal
      - optional soft labels: prob. anger ... prob. surprise (+ maybe prob. neutral)
    """
    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform=None,
        use_scaled_va: bool = True,
        return_soft: bool = False,
    ):
        self.df = pd.read_csv(csv_path).copy()
        self.img_root = img_root
        self.transform = transform
        self.use_scaled_va = use_scaled_va
        self.return_soft = return_soft

        self.df["label_str"] = self.df["filename"].astype(str).str.split("/").str[0]
        self.df = self.df[self.df["label_str"].isin(EMOTIONS)].reset_index(drop=True)

        self.soft_cols = [f"prob. {e}" for e in EMOTIONS]
        self.has_soft = all(c in self.df.columns for c in self.soft_cols)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["filename"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(EMO2ID[row["label_str"]], dtype=torch.long)

        if self.use_scaled_va and ("valence_scaled" in self.df.columns) and ("arousal_scaled" in self.df.columns):
            va = torch.tensor([row["valence_scaled"], row["arousal_scaled"]], dtype=torch.float32)
        else:
            va = torch.tensor([row["valence"], row["arousal"]], dtype=torch.float32)

        if self.return_soft and self.has_soft:
            soft = torch.tensor(row[self.soft_cols].values, dtype=torch.float32)
            soft = soft / (soft.sum() + 1e-12)
            return img, y, va, soft

        return img, y, va

    def num_classes(self) -> int:
        return len(EMOTIONS)

    def class_names(self) -> list[str]:
        return EMOTIONS
