"""
Training Script for Attention-Based Emotion Recognition Models

This is a standalone training script specifically for models with novel attention pooling.
Separate from the baseline run_train.py and run_train_extended.py.

Usage:
    1. Edit emotion_pipeline/attention_config.py to select model_type
    2. Run: python -m emotion_pipeline.run_train_attention
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from pathlib import Path

from .config import get_paths, get_checkpoint_dir, DATASET_NAME
from .attention_config import AttentionModelConfig
from .dataset_registry import get_dataset_info, infer_label_field
from .models.dinov2_multitask_extended import create_model
from .training.extended_trainer import ExtendedMultiTaskTrainer
from .training.extended_configs import LossConfig, PhaseConfig
from .training.class_weight_provider import CsvClassWeightProvider
from .training.phase_runner import PhaseRunner

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_transforms(image_size: int, train: bool):
    """Create data augmentation transforms"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # Validation: no augmentation
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def main():
    """Main training function for attention-based models"""
    
    # === Initialize Configuration ===
    paths = get_paths()
    cfg = AttentionModelConfig()
    torch.manual_seed(cfg.seed)
    
    print("=" * 60)
    print("ATTENTION-BASED EMOTION RECOGNITION TRAINING")
    print("=" * 60)
    print(f"Model architecture: {cfg.model_type}")
    print(f"Backbone: {cfg.backbone_name}")
    if cfg.model_type != "baseline":
        print(f"Attention config: {cfg.num_queries} queries, {cfg.num_attention_heads} heads")
    print("=" * 60)
    
    # === Load Class Weights ===
    csv_path = Path(paths.train_csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    import csv
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    label_field = infer_label_field(header, DATASET_NAME)
    ds_info = get_dataset_info(DATASET_NAME)
    counts = {name: 0 for name in ds_info.class_names}
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            label = str(row.get(label_field, "")).strip().lower()
            if label in counts:
                counts[label] += 1
    weight_provider = CsvClassWeightProvider(
        csv_path,
        label_field=label_field,
        class_names_override=ds_info.class_names,
    )
    class_weights = weight_provider.get()
    print("\n📊 Class Distribution:")
    print(f"  Label field: {label_field}")
    print("  Class names:", weight_provider.class_names())
    print("  Class counts:", counts)
    print("  Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])
    
    # === Create Datasets ===
    print("\n📁 Loading datasets...")
    train_ds = ds_info.dataset_cls(
        str(paths.train_csv),
        str(paths.img_root),
        transform=make_transforms(cfg.image_size, train=True)
    )
    test_ds = ds_info.dataset_cls(
        str(paths.test_csv),
        str(paths.img_root),
        transform=make_transforms(cfg.image_size, train=False)
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    
    # === Create DataLoaders ===
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # === Create Model with Attention ===
    print(f"\n🧠 Creating {cfg.model_type} model...")
    model = create_model(
        model_type=cfg.model_type,
        backbone_name=cfg.backbone_name,
        dropout=cfg.dropout,
        use_cls_plus_patchmean=True,
        num_queries=cfg.num_queries,
        num_heads=cfg.num_attention_heads,
        num_emotions=cfg.num_emotions,
        va_dims=cfg.va_dims,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check if attention pooling exists
    if hasattr(model, 'attention_pooling') and model.attention_pooling is not None:
        attn_params = sum(p.numel() for p in model.attention_pooling.parameters())
        print(f"  Attention pooling parameters: {attn_params:,}")
    
    # === Configure Loss ===
    loss_cfg = LossConfig(
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    
    # === Create Trainer ===
    print("\n🎯 Initializing trainer...")
    checkpoint_dir = get_checkpoint_dir("attention")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = ExtendedMultiTaskTrainer(
        model,
        train_loader,
        test_loader,
        cfg.device,
        cfg.lam_va,
        early_stop_lambda=cfg.early_stop_lambda,
        loss_config=loss_cfg,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # === Define Training Plan ===
    print("\n📋 Training plan:")
    print("  Phase A: Linear probe (frozen backbone)")
    
    probe_phase = PhaseConfig(
        name="PHASE A: Linear probe (frozen backbone)",
        epochs=cfg.epochs_probe,
        lr_head=cfg.lr_head_probe,
        lr_backbone=0.0,
        weight_decay=cfg.weight_decay,
        freeze_backbone=True,
        eta_min=0.0,
    )
    
    # === Run Training ===
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    runner = PhaseRunner(model, trainer)
    early_stop_patience = max(1, cfg.early_stopping_patience * 2)
    runner.run(probe_phase, patience=early_stop_patience)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Best model saved with composite score: {trainer.state.best_score:.4f}")
    print(f"  Checkpoint: {checkpoint_dir / 'best_model.pt'}")

    # Save trainer state for later analysis
    trainer_state = {
        "train_losses": trainer.state.train_losses,
        "val_losses": trainer.state.val_losses,
        "train_accs": trainer.state.train_accs,
        "val_accs": trainer.state.val_accs,
        "best_epoch": trainer.state.best_epoch,
        "best_val_acc": trainer.state.best_val_acc,
        "best_val_f1": trainer.state.best_val_f1,
        "best_val_rmse_va": trainer.state.best_val_rmse_va,
        "best_score": trainer.state.best_score,
    }
    trainer_state_path = checkpoint_dir / "trainer_state_attention.pkl"
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)
    print(f"Saved: {trainer_state_path} (for loss curve plotting)")
    
    # === Final Statistics ===
    if hasattr(trainer.state, 'best_metrics') and trainer.state.best_metrics:
        metrics = trainer.state.best_metrics
        print("\n📊 Best validation metrics:")
        if 'accuracy' in metrics:
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if 'f1' in metrics:
            print(f"  Macro F1:  {metrics['f1']:.4f}")
        if 'rmse_va' in metrics:
            print(f"  RMSE V-A:  {metrics['rmse_va']:.4f}")
    
    print("\n✨ Training complete! Check logs and checkpoints for details.\n")


if __name__ == "__main__":
    main()
