"""
Analyze training results: loss curves, per-class accuracy, confusion matrix.
Run this after training completes.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import config, EMOTION_LABELS, get_checkpoint_dir
from dataset_registry import get_dataset_info
from emotion_pipeline.models.dinov2_multitask import DinoV2EmotionVA
from emotion_pipeline.run_train import make_transforms
from emotion_pipeline.training.multitask_trainer import MultiTaskTrainer
from torch.utils.data import DataLoader

def load_and_analyze(
    checkpoint_path: str | None = None,
    trainer_state_path: str | None = None,
):
    """Load trained model and analyze on test set."""
    if checkpoint_path is None:
        checkpoint_path = str(get_checkpoint_dir("attention") / "best_model.pt")
    if trainer_state_path is None:
        trainer_state_path = str(get_checkpoint_dir("attention") / "trainer_state_attention.pkl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load trainer state (loss curves, etc.)
    trainer_state = None
    if Path(trainer_state_path).exists():
        import pickle
        with open(trainer_state_path, "rb") as f:
            trainer_state = pickle.load(f)
        print(f"✓ Loaded trainer state from {trainer_state_path}")
    
    # Load model
    model = DinoV2EmotionVA(
        backbone_name=config.backbone_name,
        use_cls_plus_patchmean=True,
        num_emotions=config.num_emotions,
        va_dims=config.va_dims,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"✓ Loaded model from {checkpoint_path}")
    
    # Load test dataset
    ds_info = get_dataset_info(getattr(config, "DATASET_NAME", "emotion6"))
    test_ds = ds_info.dataset_cls(
        str(config.test_csv), 
        str(config.img_root), 
        transform=make_transforms(config.image_size, train=False)
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"✓ Loaded {len(test_ds)} test samples")
    
    # Create trainer
    train_ds = ds_info.dataset_cls(
        str(config.train_csv),
        str(config.img_root),
        transform=make_transforms(config.image_size, train=True)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    trainer = MultiTaskTrainer(model, train_loader, test_loader, device, lam_va=1.0)
    
    # Get per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    per_class_acc = trainer.get_per_class_accuracy(num_classes=6)
    for class_id, acc in per_class_acc.items():
        emotion = EMOTION_LABELS.get(class_id, f"Class {class_id}")
        print(f"  {emotion:12s}: {acc*100:6.2f}%")
    
    avg_acc = np.mean(list(per_class_acc.values()))
    print(f"  {'Average':12s}: {avg_acc*100:6.2f}%")
    
    # Get confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = trainer.get_confusion_matrix(num_classes=6)
    print("\nPredicted →")
    print("True ↓")
    header = "        " + "  ".join([f"{EMOTION_LABELS[i][:3]}" for i in range(6)])
    print(header)
    for i in range(6):
        row = f"{EMOTION_LABELS[i][:3]:6s} | " + " ".join([f"{cm[i, j]:4.0f}" for j in range(6)])
        print(row)
    
    # Plot loss curves
    print("\n[1/3] Plotting loss curves...")
    plot_loss_curves(trainer_state)
    
    # Plot per-class accuracy
    print("[2/3] Plotting per-class accuracy...")
    plot_per_class_accuracy(per_class_acc)
    
    # Plot confusion matrix
    print("[3/3] Plotting confusion matrix...")
    plot_confusion_matrix(cm)
    
    print("\n✓ All plots saved to current directory")

def plot_loss_curves(trainer_state: dict = None):
    """Plot train/test loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if trainer_state and len(trainer_state.get("train_losses", [])) > 0:
        epochs = range(1, len(trainer_state["train_losses"]) + 1)
        
        # Loss curve
        ax1.plot(epochs, trainer_state["train_losses"], 'o-', label='Train Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, trainer_state["val_losses"], 's-', label='Val Loss', linewidth=2, markersize=4)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title("Train/Test Loss Curves", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, [a*100 for a in trainer_state["train_accs"]], 'o-', label='Train Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs, [a*100 for a in trainer_state["val_accs"]], 's-', label='Val Accuracy', linewidth=2, markersize=4)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_title("Train/Test Accuracy Curves", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No trainer state found.\n\n"
                 "Run training first to generate trainer_state.pkl",
                 ha="center", va="center", fontsize=12, transform=ax1.transAxes)
        ax1.axis("off")
        ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: loss_curves.png")

def plot_per_class_accuracy(per_class_acc: dict):
    """Plot per-class accuracy as bar chart."""
    emotions = [EMOTION_LABELS[i] for i in range(6)]
    accs = [per_class_acc.get(i, 0) * 100 for i in range(6)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(emotions, accs, color="steelblue", alpha=0.7, edgecolor="navy")
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy on Test Set", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: per_class_accuracy.png")

def plot_confusion_matrix(cm: np.ndarray):
    """Plot confusion matrix as heatmap."""
    emotions = [EMOTION_LABELS[i] for i in range(6)]
    
    # Normalize by row (true class) for better interpretation
    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", 
                xticklabels=emotions, yticklabels=emotions, ax=ax1, cbar_kws={"label": "Count"})
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("True Label", fontsize=11)
    ax1.set_xlabel("Predicted Label", fontsize=11)
    
    # Normalized (per-class)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=emotions, yticklabels=emotions, ax=ax2, cbar_kws={"label": "Proportion"})
    ax2.set_title("Confusion Matrix (Normalized by True Class)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("True Label", fontsize=11)
    ax2.set_xlabel("Predicted Label", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: confusion_matrix.png")
    
    # Print top misclassifications
    print("\nTop Misclassifications:")
    misclass = []
    for i in range(6):
        for j in range(6):
            if i != j:
                misclass.append((EMOTION_LABELS[i], EMOTION_LABELS[j], cm[i, j]))
    
    misclass.sort(key=lambda x: x[2], reverse=True)
    for true_label, pred_label, count in misclass[:5]:
        if count > 0:
            print(f"  {true_label} misclassified as {pred_label}: {int(count)} samples")

if __name__ == "__main__":
    load_and_analyze()
