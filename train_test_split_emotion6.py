# make_emotion6_split.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def derive_label_from_filename(filename: str) -> str:
    # Expected: "disgust/1.jpg" -> "disgust"
    # Works also if full paths exist: ".../disgust/1.jpg" -> "disgust" (last folder)
    parts = str(filename).replace("\\", "/").split("/")
    return parts[0] if len(parts) >= 2 else parts[0]

def add_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds soft_anger ... soft_surprise based on 'prob. <emotion>' columns,
    renormalized to sum to 1 across the 6 emotions (ignoring neutral).
    """
    prob_cols = [f"prob. {e}" for e in EMOTIONS]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Soft-label columns missing, skipping soft labels: {missing}")
        return df

    probs = df[prob_cols].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1, keepdims=True)

    # Avoid divide-by-zero: if sum==0, make uniform
    probs = np.where(row_sums > 0, probs / row_sums, np.full_like(probs, 1.0 / len(EMOTIONS)))

    for i, emo in enumerate(EMOTIONS):
        df[f"soft_{emo}"] = probs[:, i]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="Emotion6_New_groundtruth.csv")
    ap.add_argument("--out_dir", type=str, default=".", help="Where to save train/test CSVs")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--add_soft", action="store_true", help="Add renormalized soft labels soft_* (ignoring neutral)")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if "filename" not in df.columns:
        raise ValueError("CSV must contain a 'filename' column.")

    # Hard label from the folder name in filename
    df["label"] = df["filename"].apply(derive_label_from_filename)

    # Keep only the 6 Emotion6 classes
    df = df[df["label"].isin(EMOTIONS)].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("After filtering to 6 emotions, dataframe is empty. Check filename format or labels.")

    # Stratified split by label (important because sadness is often small)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    # Optional: add soft labels (from prob. columns)
    if args.add_soft:
        train_df = add_soft_labels(train_df)
        test_df = add_soft_labels(test_df)

    # Save
    train_path = out_dir / "emotion6_train_80.csv"
    test_path = out_dir / "emotion6_test_20.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print split stats
    print("Saved:")
    print(f"  Train: {train_path}  (n={len(train_df)})")
    print(f"  Test : {test_path}   (n={len(test_df)})")
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts())
    print("\nLabel distribution (test):")
    print(test_df["label"].value_counts())

if __name__ == "__main__":
    main()
