import torch

class Metrics:
    @staticmethod
    def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

    @staticmethod
    def macro_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
        pred = logits.argmax(dim=1)
        f1s = []
        for c in range(num_classes):
            tp = ((pred == c) & (y == c)).sum().item()
            fp = ((pred == c) & (y != c)).sum().item()
            fn = ((pred != c) & (y == c)).sum().item()
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            f1s.append(f1)
        return float(sum(f1s) / len(f1s))

    @staticmethod
    def rmse_va(pred_va: torch.Tensor, true_va: torch.Tensor) -> float:
        return torch.sqrt(((pred_va - true_va) ** 2).mean()).item()
