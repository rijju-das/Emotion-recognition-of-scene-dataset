import os
import argparse
import pandas as pd
import numpy as np

LABEL_COL_CANDIDATES = ["emotion", "final_emo", "label_str", "label"]
FILENAME_COL_CANDIDATES = ["filename", "image_filename", "image_path", "file"]


def infer_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def stratified_split(df, label_col, train_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for label, sub in df.groupby(label_col):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(len(idx) * train_ratio)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])

    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default ="/home/rdas/color_transfer/D-Visa/D-ViSA_processed.csv")
    ap.add_argument("--train_out", default="/home/rdas/color_transfer/D-Visa/D-ViSA_train_80.csv", help="Output train CSV")
    ap.add_argument("--test_out", default="/home/rdas/color_transfer/D-Visa/D-ViSA_test_20.csv", help="Output test CSV")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    label_col = infer_col(df, LABEL_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"CSV missing label column. Expected one of {LABEL_COL_CANDIDATES}")

    filename_col = infer_col(df, FILENAME_COL_CANDIDATES)
    if filename_col is None:
        raise ValueError(f"CSV missing filename column. Expected one of {FILENAME_COL_CANDIDATES}")

    df[label_col] = df[label_col].astype(str).str.lower()

    train_df, test_df = stratified_split(df, label_col, args.train_ratio, args.seed)

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print(f"Saved train: {args.train_out} ({len(train_df)})")
    print(f"Saved test:  {args.test_out} ({len(test_df)})")


if __name__ == "__main__":
    main()
