from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from .base import BaseTrainer
from ..eval.metrics import Metrics

@dataclass
class TrainerState:
    best_val_loss: float = 1e18
    best_val_acc: float = 0.0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_accs: list = field(default_factory=list)
    val_accs: list = field(default_factory=list)
    best_epoch: int = 0

class MultiTaskTrainer(BaseTrainer):
    def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader, device: str, lam_va: float, checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lam_va = lam_va
        self.state = TrainerState()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(exist_ok=True)

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

    def fit(self, optimizer, epochs: int, num_classes: int = 6, patience: int = 20, scheduler=None):
        print(f"\n{'='*80}")
        print(f"Training for {epochs} epochs | device={self.device} | early_stopping_patience={patience} | scheduler={'CosineAnnealingLR' if scheduler else 'None'}")
        print(f"{'='*80}\n")
        
        no_improve_count = 0
        
        for ep in range(1, epochs + 1):
            # Training phase: compute loss and train accuracy
            total_train_loss = 0.0
            all_train_logits, all_train_y = [], []
            
            self.model.train()
            for batch in self.train_loader:
                loss, logits, va_pred, y, va = self._step(batch, train=True, optimizer=optimizer)
                total_train_loss += loss
                all_train_logits.append(logits.cpu())
                all_train_y.append(y.cpu())
            
            train_logits = torch.cat(all_train_logits, dim=0)
            train_y = torch.cat(all_train_y, dim=0)
            train_acc = Metrics.accuracy(train_logits, train_y)
            train_f1 = Metrics.macro_f1(train_logits, train_y, num_classes=num_classes)
            train_loss_avg = total_train_loss / len(self.train_loader)
            
            # Validation phase
            val = self.evaluate(num_classes=num_classes)
            
            # Track best model and early stopping
            if val['loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val['loss']
                self.state.best_val_acc = val['acc']
                self.state.best_epoch = ep
                no_improve_count = 0
                best_marker = " ⭐ (best)"
                
                # Save best model (overwrite)
                if self.checkpoint_dir is not None:
                    checkpoint_path = self.checkpoint_dir / "best_model.pt"
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                no_improve_count += 1
                best_marker = ""
            
            # Store metrics for later plotting
            self.state.train_losses.append(train_loss_avg)
            self.state.val_losses.append(val['loss'])
            self.state.train_accs.append(train_acc)
            self.state.val_accs.append(val['acc'])
            
            # Detailed logging
            print(f"Epoch {ep:3d}/{epochs} | "
                  f"Train: loss={train_loss_avg:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
                  f"Val: loss={val['loss']:.4f} acc={val['acc']:.4f} f1={val['f1']:.4f} rmse_va={val['rmse_va']:.4f}{best_marker}")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⭐ Early stopping at epoch {ep} (val loss didn't improve for {patience} epochs)")
                break
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"Training complete. Best val_loss: {self.state.best_val_loss:.4f}")
        print(f"{'='*80}\n")

        if self.checkpoint_dir is not None:
            final_checkpoint = self.checkpoint_dir / "final_model.pt"
            torch.save(self.model.state_dict(), final_checkpoint)

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
            "logits": logits,
            "y": y,
        }
    
    def get_per_class_accuracy(self, num_classes: int = 6):
        """Compute per-class accuracy on test set."""
        self.model.eval()
        all_logits, all_y = [], []
        
        with torch.no_grad():
            for batch in self.test_loader:
                x, y, _ = batch[:3]
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                all_logits.append(logits.cpu())
                all_y.append(y.cpu())
        
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        pred = logits.argmax(dim=1)
        
        per_class_acc = {}
        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() > 0:
                acc = (pred[mask] == c).float().mean().item()
                per_class_acc[c] = acc
        return per_class_acc
    
    def get_confusion_matrix(self, num_classes: int = 6):
        """Compute confusion matrix on test set."""
        self.model.eval()
        cm = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for batch in self.test_loader:
                x, y, _ = batch[:3]
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                pred = logits.argmax(dim=1)
                
                for i in range(len(y)):
                    cm[y[i].item(), pred[i].item()] += 1
        
        return cm.numpy()
