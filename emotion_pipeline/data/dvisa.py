import os
import pandas as pd
from PIL import Image
import torch
from .base import BaseDataset

DVISA_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "amusement",
    "awe",
    "sadness",
    "excitement",
    "contentment",
]
EMO2ID = {e: i for i, e in enumerate(DVISA_EMOTIONS)}

class DVisaDataset(BaseDataset):
    """
    D-ViSA dataset loader.

    Expects a CSV with at least:
      - filename (or image_filename / image_path)
      - emotion (or final_emo)
      - valence_scaled/arousal_scaled or valence/arousal
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

        self.filename_col = self._infer_filename_col(self.df)
        self.emotion_col = self._infer_emotion_col(self.df)
        self.df["label_str"] = self.df[self.emotion_col].astype(str).str.lower()

        if self.emotion_col is None:
            self.df["label_str"] = self._label_from_filename(self.df[self.filename_col])

        self.df = self.df[self.df["label_str"].isin(DVISA_EMOTIONS)].reset_index(drop=True)

        self.soft_cols = [f"prob. {e}" for e in DVISA_EMOTIONS]
        self.has_soft = all(c in self.df.columns for c in self.soft_cols)

    def _infer_filename_col(self, df: pd.DataFrame) -> str:
        for col in ["filename", "image_filename", "image_path", "file"]:
            if col in df.columns:
                return col
        raise ValueError("CSV must include a filename column (filename/image_filename/image_path/file).")

    def _infer_emotion_col(self, df: pd.DataFrame) -> str | None:
        for col in ["emotion", "final_emo", "label_str", "label"]:
            if col in df.columns:
                return col
        return None

    def _label_from_filename(self, series: pd.Series) -> pd.Series:
        labels = series.astype(str).str.split("/").str[0].str.lower()
        return labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row[self.filename_col])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(EMO2ID[row["label_str"]], dtype=torch.long)

        if self.use_scaled_va and ("valence_scaled" in self.df.columns) and ("arousal_scaled" in self.df.columns):
            va_vals = [row["valence_scaled"], row["arousal_scaled"]]
            if "dominance_scaled" in self.df.columns:
                va_vals.append(row["dominance_scaled"])
            va = torch.tensor(va_vals, dtype=torch.float32)
        elif ("valence" in self.df.columns) and ("arousal" in self.df.columns):
            va_vals = [row["valence"], row["arousal"]]
            if "dominance" in self.df.columns:
                va_vals.append(row["dominance"])
            va = torch.tensor(va_vals, dtype=torch.float32)
        else:
            raise ValueError("CSV must include valence/arousal or valence_scaled/arousal_scaled columns.")

        if self.return_soft and self.has_soft:
            soft = torch.tensor(row[self.soft_cols].values, dtype=torch.float32)
            soft = soft / (soft.sum() + 1e-12)
            return img, y, va, soft

        return img, y, va

    def num_classes(self) -> int:
        return len(DVISA_EMOTIONS)

    def class_names(self) -> list[str]:
        return DVISA_EMOTIONS
