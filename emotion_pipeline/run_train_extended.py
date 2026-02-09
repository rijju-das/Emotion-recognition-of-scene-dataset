import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from pathlib import Path

from .config import get_paths, get_checkpoint_dir, DATASET_NAME
from .extended_config import ExtendedTrainConfig
from .dataset_registry import get_dataset_info, infer_label_field
from .models.dinov2_multitask_extended import create_model
from .training.extended_trainer import ExtendedMultiTaskTrainer
from .training.extended_configs import LossConfig, PhaseConfig, TrainingPlan
from .training.class_weight_provider import CsvClassWeightProvider
from .training.phase_runner import PhaseRunner

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

def main():
    paths = get_paths()
    cfg = ExtendedTrainConfig()
    torch.manual_seed(cfg.seed)

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
    print("Class names:", weight_provider.class_names())
    print(f"Label field: {label_field}")
    print("Class counts:", counts)
    print("Class weights (normalized):", class_weights.tolist())

    train_ds = ds_info.dataset_cls(str(paths.train_csv), str(paths.img_root), transform=make_transforms(cfg.image_size, True))
    test_ds  = ds_info.dataset_cls(str(paths.test_csv),  str(paths.img_root), transform=make_transforms(cfg.image_size, False))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Create model using factory function
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
    print(f"✓ Using {cfg.model_type} model with {cfg.backbone_name}")
    if cfg.model_type != "baseline":
        print(f"  Attention: {cfg.num_queries} queries/groups, {cfg.num_attention_heads} heads")

    loss_cfg = LossConfig(
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    checkpoint_dir = get_checkpoint_dir("extended")
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
    runner = PhaseRunner(model, trainer)

    plan = TrainingPlan(
        probe=PhaseConfig(
            name="PHASE A: Linear probe (frozen backbone)",
            epochs=cfg.epochs_probe,
            lr_head=cfg.lr_head_probe,
            lr_backbone=0.0,
            weight_decay=cfg.weight_decay_probe,
            freeze_backbone=True,
            eta_min=cfg.eta_min_probe,
        ),
        finetune=PhaseConfig(
            name="PHASE B: Fine-tuning (very low backbone LR)",
            epochs=cfg.epochs_finetune,
            lr_head=cfg.lr_head,
            lr_backbone=cfg.lr_backbone,
            weight_decay=cfg.weight_decay,
            freeze_backbone=False,
            eta_min=cfg.eta_min_finetune,
        ),
    )

    runner.run(plan.probe, patience=cfg.early_stopping_patience)
    runner.run(plan.finetune, patience=cfg.early_stopping_patience)

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
    trainer_state_path = checkpoint_dir / "trainer_state_extended.pkl"
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)
    print(f"Saved: {trainer_state_path} (for loss curve plotting)")

if __name__ == "__main__":
    main()
