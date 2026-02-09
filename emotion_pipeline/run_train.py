import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from pathlib import Path

from .config import get_paths, get_checkpoint_dir, TrainConfig, DATASET_NAME
from .dataset_registry import get_dataset_info
from .models.dinov2_multitask import DinoV2EmotionVA
from .training.multitask_trainer import MultiTaskTrainer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),  # More aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),  # Increased from 10
            # transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),  # Stronger
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Translation
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),    # NEW: blur
            # transforms.RandomPerspective(distortion_scale=0.2),          # NEW: perspective
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # Test/val: NO augmentation, just resize and crop
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def freeze_backbone(model: DinoV2EmotionVA, freeze: bool):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze

def main():
    paths = get_paths()
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    ds_info = get_dataset_info(DATASET_NAME)
    train_ds = ds_info.dataset_cls(str(paths.train_csv), str(paths.img_root), transform=make_transforms(cfg.image_size, True))
    test_ds  = ds_info.dataset_cls(str(paths.test_csv),  str(paths.img_root), transform=make_transforms(cfg.image_size, False))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = DinoV2EmotionVA(
        backbone_name=cfg.backbone_name,
        use_cls_plus_patchmean=True,
        dropout=cfg.dropout,
        num_emotions=cfg.num_emotions,
    )

    # Phase A: linear probe
    freeze_backbone(model, True)
    optimizer = torch.optim.AdamW(
        list(model.emotion_head.parameters()) + list(model.va_head.parameters()),
        lr=cfg.lr_head, weight_decay=cfg.weight_decay
    )
    checkpoint_dir = get_checkpoint_dir("base")
    trainer = MultiTaskTrainer(
        model,
        train_loader,
        test_loader,
        cfg.device,
        cfg.lam_va,
        early_stop_lambda=cfg.early_stop_lambda,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Add LR scheduler for phase A
    scheduler_probe = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_probe, eta_min=1e-5)
    trainer.fit(optimizer=optimizer, epochs=cfg.epochs_probe, patience=cfg.early_stopping_patience, scheduler=scheduler_probe)

    # Phase B: conservative fine-tune (very small LR on backbone)
    print("\n" + "="*80)
    print("PHASE B: Fine-tuning with VERY low backbone LR to prevent overfitting")
    print("="*80)
    freeze_backbone(model, False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.emotion_head.parameters(), "lr": cfg.lr_head},
            {"params": model.va_head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay
    )
    
    # Add LR scheduler for phase B with even lower eta_min
    scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs_finetune, eta_min=1e-7)
    trainer.fit(optimizer=optimizer, epochs=cfg.epochs_finetune, patience=cfg.early_stopping_patience, scheduler=scheduler_finetune)

    # Load best checkpoint and save as final model
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        print(f"\n✓ Loading best checkpoint: {best_checkpoint}")
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))
        print(
            "  Best score: "
            f"{trainer.state.best_score:.4f} (f1={trainer.state.best_val_f1:.4f}, rmse_va={trainer.state.best_val_rmse_va:.4f}, epoch {trainer.state.best_epoch})"
        )
    
    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Saved final model: {final_model_path}")
    
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
    trainer_state_path = checkpoint_dir / "trainer_state_base.pkl"
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)
    print(f"Saved: {trainer_state_path} (for loss curve plotting)")

if __name__ == "__main__":
    main()
