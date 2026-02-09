from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .multitask_trainer import MultiTaskTrainer
from .extended_configs import LossConfig

class ExtendedMultiTaskTrainer(MultiTaskTrainer):
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        lam_va: float,
        early_stop_lambda: float = 0.3,
        loss_config: Optional[LossConfig] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        super().__init__(
            model,
            train_loader,
            test_loader,
            device,
            lam_va,
            early_stop_lambda=early_stop_lambda,
            checkpoint_dir=checkpoint_dir,
        )
        self.loss_config = loss_config or LossConfig()

    def _step(self, batch, train: bool, optimizer=None):
        x, y, va = batch[:3]
        x, y, va = x.to(self.device), y.to(self.device), va.to(self.device)

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            logits, va_pred = self.model(x)
            weight = self.loss_config.class_weights
            if weight is not None:
                weight = weight.to(self.device)
            loss_cls = F.cross_entropy(
                logits,
                y,
                weight=weight,
                label_smoothing=self.loss_config.label_smoothing,
            )
            loss_va = F.smooth_l1_loss(va_pred, va)
            loss = loss_cls + self.lam_va * loss_va

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.item(), logits.detach(), va_pred.detach(), y.detach(), va.detach()
