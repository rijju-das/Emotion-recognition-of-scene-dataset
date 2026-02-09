from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv
import torch

class ClassWeightProvider(ABC):
    @abstractmethod
    def get(self) -> torch.Tensor:
        ...

    @abstractmethod
    def class_names(self) -> List[str]:
        ...

@dataclass
class CsvClassWeightProvider(ClassWeightProvider):
    csv_path: Path
    label_field: str = "label"
    class_names_override: Optional[List[str]] = None

    def _labels(self) -> List[str]:
        labels: List[str] = []
        with open(self.csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                labels.append(row[self.label_field].strip())
        return labels

    def class_names(self) -> List[str]:
        if self.class_names_override is not None:
            return self.class_names_override
        labels = self._labels()
        return sorted(list(set(labels)))

    def get(self) -> torch.Tensor:
        labels = self._labels()
        class_names = self.class_names()
        name2idx = {n: i for i, n in enumerate(class_names)}
        counts = {i: 0 for i in range(len(class_names))}
        for label in labels:
            counts[name2idx[label]] += 1

        total = sum(counts.values())
        n_classes = len(class_names)
        weights: List[float] = []
        for c in range(n_classes):
            cnt = counts.get(c, 0)
            w = (total / (n_classes * cnt)) if cnt > 0 else 0.0
            weights.append(w)
        if weights:
            mean_w = sum(weights) / len(weights)
            if mean_w > 0:
                weights = [w / mean_w for w in weights]
        return torch.tensor(weights, dtype=torch.float)
