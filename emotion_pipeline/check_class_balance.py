# emotion_pipeline/check_class_balance.py
import csv
from collections import Counter
from pathlib import Path
import numpy as np

CSV = Path(__file__).resolve().parent / ".." / "emotion6_train_80.csv"
CSV = CSV.resolve()

def compute_weights(counts, n_classes):
    total = sum(counts.values())
    weights = []
    for c in range(n_classes):
        cnt = counts.get(c, 0)
        # balanced weight: total / (n_classes * count)
        w = (total / (n_classes * cnt)) if cnt > 0 else 0.0
        weights.append(w)
    # normalize to mean 1
    weights = np.array(weights, dtype=float)
    if weights.mean() > 0:
        weights = weights / weights.mean()
    return weights

def main():
    # expects last column 'label' with textual class names -> map to indices
    labels = []
    with open(CSV, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            labels.append(r["label"].strip())
    class_names = sorted(list(set(labels)))
    name2idx = {n: i for i, n in enumerate(class_names)}
    idxs = [name2idx[l] for l in labels]

    counts = Counter(idxs)
    print("Class mapping (name -> idx):")
    for n, i in name2idx.items():
        print(f"  {n} -> {i}")
    print("\nCounts:")
    for i in range(len(class_names)):
        print(f"  {i} ({class_names[i]}): {counts.get(i,0)}")

    weights = compute_weights(counts, n_classes=len(class_names))
    print("\nComputed class weights (balanced, normalized mean=1):")
    for i,w in enumerate(weights):
        print(f"  {i} ({class_names[i]}): {w:.4f}")

if __name__ == "__main__":
    main()