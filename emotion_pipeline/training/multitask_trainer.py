from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import BaseTrainer
from ..eval.metrics import Metrics

@dataclass
class TrainerState:
    best_val_loss: float = 1e18

class MultiTaskTrainer(BaseTrainer):
    def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader, device: str, lam_va: float):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lam_va = lam_va
        self.state = TrainerState()

    def _step(self, batch, train: bool, optimizer=None):
        x, y, va = batch[:3]
        x, y, va = x.to(self.device), y.to(self.device), va.to(self.device)

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            logits, va_pred = self.model(x)
            loss_cls = F.cross_entropy(logits, y)
            loss_va = F.smooth_l1_loss(va_pred, va)  # Huber
            loss = loss_cls + self.lam_va * loss_va

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.item(), logits.detach(), va_pred.detach(), y.detach(), va.detach()

    def fit(self, optimizer, epochs: int, num_classes: int = 6):
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for batch in self.train_loader:
                loss, *_ = self._step(batch, train=True, optimizer=optimizer)
                total_loss += loss

            val = self.evaluate(num_classes=num_classes)
            print(f"Epoch {ep:03d} | train_loss={total_loss/len(self.train_loader):.4f} | "
                  f"test_loss={val['loss']:.4f} acc={val['acc']:.4f} f1={val['f1']:.4f} rmse_va={val['rmse_va']:.4f}")

    def evaluate(self, num_classes: int = 6):
        total_loss = 0.0
        all_logits, all_y = [], []
        all_va_pred, all_va_true = [], []

        for batch in self.test_loader:
            loss, logits, va_pred, y, va = self._step(batch, train=False)
            total_loss += loss
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
            all_va_pred.append(va_pred.cpu())
            all_va_true.append(va.cpu())

        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        va_pred = torch.cat(all_va_pred, dim=0)
        va_true = torch.cat(all_va_true, dim=0)

        return {
            "loss": total_loss / len(self.test_loader),
            "acc": Metrics.accuracy(logits, y),
            "f1": Metrics.macro_f1(logits, y, num_classes=num_classes),
            "rmse_va": Metrics.rmse_va(va_pred, va_true),
        }
