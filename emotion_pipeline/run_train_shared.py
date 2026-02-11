"""
Shared training entrypoint for base, extended, and attention variants.

Control behavior in emotion_pipeline/config.py:
- DATASET_NAME: "emotion6", "dvisa", or "emoset_new"
- TRAIN_VARIANT: "base", "extended", or "attention"
- TASK_MODE: "auto", "multitask", or "emotion_only"
"""

import csv
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import (
    DATASET_NAME,
    get_paths,
    get_checkpoint_dir,
    get_task_mode,
    get_train_variant,
    TrainConfig,
)
from .extended_config import ExtendedTrainConfig
from .attention_config import AttentionModelConfig
from .dataset_registry import get_dataset_info, infer_label_field
from .models.dinov2_multitask import DinoV2EmotionVA
from .models.dinov2_multitask_extended import create_model
from .training.multitask_trainer import MultiTaskTrainer
from .training.emotion_trainer import EmotionTrainer
from .training.extended_trainer import ExtendedMultiTaskTrainer
from .training.extended_configs import LossConfig, PhaseConfig, TrainingPlan
from .training.class_weight_provider import CsvClassWeightProvider
from .training.phase_runner import PhaseRunner

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def resolve_label_field(csv_path: Path) -> str:
    if DATASET_NAME == "emoset_new":
        return "emotion"
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    return infer_label_field(header, DATASET_NAME)


def load_class_weights(csv_path: Path, label_field: str, class_names: list[str]):
    counts = {name: 0 for name in class_names}
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            label = str(row.get(label_field, "")).strip().lower()
            if label in counts:
                counts[label] += 1
    weight_provider = CsvClassWeightProvider(
        csv_path,
        label_field=label_field,
        class_names_override=class_names,
    )
    return weight_provider.get(), counts, weight_provider.class_names()


def freeze_backbone(model, freeze: bool):
    if hasattr(model, "backbone"):
        model.backbone.requires_grad_(not freeze)


def set_va_trainable(model, trainable: bool):
    if hasattr(model, "va_head") and model.va_head is not None:
        model.va_head.requires_grad_(trainable)


def train_base(model, cfg: TrainConfig, train_loader, test_loader, task_mode: str):
    if task_mode == "multitask":
        checkpoint_name = "base"
        trainer = MultiTaskTrainer(
            model,
            train_loader,
            test_loader,
            cfg.device,
            cfg.lam_va,
            early_stop_lambda=cfg.early_stop_lambda,
            checkpoint_dir=str(get_checkpoint_dir(checkpoint_name)),
        )
    else:
        checkpoint_name = "emoset_emotion"
        trainer = EmotionTrainer(
            model,
            train_loader,
            test_loader,
            cfg.device,
            checkpoint_dir=str(get_checkpoint_dir(checkpoint_name)),
        )
        set_va_trainable(model, False)

    freeze_backbone(model, True)
    if task_mode == "multitask":
        params = list(model.emotion_head.parameters()) + list(model.va_head.parameters())
    else:
        params = list(model.emotion_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler_probe = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs_probe,
        eta_min=1e-5,
    )
    trainer.fit(
        optimizer=optimizer,
        epochs=cfg.epochs_probe,
        patience=cfg.early_stopping_patience,
        scheduler=scheduler_probe,
    )

    freeze_backbone(model, False)
    if task_mode == "multitask":
        params = [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.emotion_head.parameters(), "lr": cfg.lr_head},
            {"params": model.va_head.parameters(), "lr": cfg.lr_head},
        ]
    else:
        params = [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.emotion_head.parameters(), "lr": cfg.lr_head},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs_finetune,
        eta_min=1e-7,
    )
    trainer.fit(
        optimizer=optimizer,
        epochs=cfg.epochs_finetune,
        patience=cfg.early_stopping_patience,
        scheduler=scheduler_finetune,
    )

    checkpoint_dir = get_checkpoint_dir(checkpoint_name)
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    trainer_state = {
        "train_losses": trainer.state.train_losses,
        "val_losses": trainer.state.val_losses,
        "train_accs": trainer.state.train_accs,
        "val_accs": trainer.state.val_accs,
        "best_epoch": trainer.state.best_epoch,
        "best_val_acc": trainer.state.best_val_acc,
        "best_val_f1": trainer.state.best_val_f1,
        "best_score": trainer.state.best_score,
    }
    if task_mode == "multitask":
        trainer_state["best_val_rmse_va"] = trainer.state.best_val_rmse_va
    trainer_state_path = checkpoint_dir / (
        "trainer_state_base.pkl" if task_mode == "multitask" else "trainer_state_emoset.pkl"
    )
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)


def train_extended_multitask(
    model,
    cfg: ExtendedTrainConfig,
    train_loader,
    test_loader,
    class_weights,
):
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

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

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


def train_extended_emotion_only(
    model,
    cfg: ExtendedTrainConfig,
    train_loader,
    test_loader,
    class_weights,
):
    checkpoint_dir = get_checkpoint_dir("extended_emoset")
    trainer = EmotionTrainer(
        model,
        train_loader,
        test_loader,
        cfg.device,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
        checkpoint_dir=str(checkpoint_dir),
    )

    freeze_backbone(model, True)
    set_va_trainable(model, False)
    if hasattr(model, "attention_pooling") and model.attention_pooling is not None:
        model.attention_pooling.requires_grad_(True)
    model.emotion_head.requires_grad_(True)

    optimizer_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head_probe,
        weight_decay=cfg.weight_decay_probe,
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a,
        T_max=cfg.epochs_probe,
        eta_min=cfg.eta_min_probe,
    )
    trainer.fit(
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        epochs=cfg.epochs_probe,
        patience=cfg.early_stopping_patience,
    )

    freeze_backbone(model, False)
    set_va_trainable(model, False)
    if hasattr(model, "attention_pooling") and model.attention_pooling is not None:
        model.attention_pooling.requires_grad_(True)
    model.emotion_head.requires_grad_(True)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
        {"params": model.emotion_head.parameters(), "lr": cfg.lr_head},
    ]
    if hasattr(model, "attention_pooling") and model.attention_pooling is not None:
        param_groups.append({"params": model.attention_pooling.parameters(), "lr": cfg.lr_head})

    optimizer_b = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b,
        T_max=cfg.epochs_finetune,
        eta_min=cfg.eta_min_finetune,
    )
    trainer.fit(
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        epochs=cfg.epochs_finetune,
        patience=cfg.early_stopping_patience,
    )

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    trainer_state = {
        "train_losses": trainer.state.train_losses,
        "val_losses": trainer.state.val_losses,
        "train_accs": trainer.state.train_accs,
        "val_accs": trainer.state.val_accs,
        "best_epoch": trainer.state.best_epoch,
        "best_val_acc": trainer.state.best_val_acc,
        "best_val_f1": trainer.state.best_val_f1,
    }
    trainer_state_path = checkpoint_dir / "trainer_state_extended_emoset.pkl"
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)


def train_attention_multitask(
    model,
    cfg: AttentionModelConfig,
    train_loader,
    test_loader,
    class_weights,
):
    loss_cfg = LossConfig(
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    checkpoint_dir = get_checkpoint_dir("attention")
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
    probe_phase = PhaseConfig(
        name="PHASE A: Linear probe (frozen backbone)",
        epochs=cfg.epochs_probe,
        lr_head=cfg.lr_head_probe,
        lr_backbone=0.0,
        weight_decay=cfg.weight_decay,
        freeze_backbone=True,
        eta_min=0.0,
    )

    runner = PhaseRunner(model, trainer)
    early_stop_patience = max(1, cfg.early_stopping_patience * 2)
    runner.run(probe_phase, patience=early_stop_patience)

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

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


def train_attention_emotion_only(
    model,
    cfg: AttentionModelConfig,
    train_loader,
    test_loader,
    class_weights,
):
    checkpoint_dir = get_checkpoint_dir("attention_emoset")
    trainer = EmotionTrainer(
        model,
        train_loader,
        test_loader,
        cfg.device,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
        checkpoint_dir=str(checkpoint_dir),
    )

    freeze_backbone(model, True)
    set_va_trainable(model, False)
    if hasattr(model, "attention_pooling") and model.attention_pooling is not None:
        model.attention_pooling.requires_grad_(True)
    model.emotion_head.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head_probe,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs_probe,
        eta_min=0.0,
    )
    trainer.fit(
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.epochs_probe,
        patience=max(1, cfg.early_stopping_patience * 2),
    )

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=cfg.device))

    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    trainer_state = {
        "train_losses": trainer.state.train_losses,
        "val_losses": trainer.state.val_losses,
        "train_accs": trainer.state.train_accs,
        "val_accs": trainer.state.val_accs,
        "best_epoch": trainer.state.best_epoch,
        "best_val_acc": trainer.state.best_val_acc,
        "best_val_f1": trainer.state.best_val_f1,
    }
    trainer_state_path = checkpoint_dir / "trainer_state_attention_emoset.pkl"
    with open(trainer_state_path, "wb") as f:
        pickle.dump(trainer_state, f)


def main():
    paths = get_paths()
    variant = get_train_variant()
    task_mode = get_task_mode()

    if variant == "base":
        cfg = TrainConfig()
    elif variant == "extended":
        cfg = ExtendedTrainConfig()
    else:
        cfg = AttentionModelConfig()

    torch.manual_seed(cfg.seed)

    ds_info = get_dataset_info(DATASET_NAME)
    csv_path = Path(paths.train_csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path

    label_field = resolve_label_field(csv_path)
    class_weights, counts, class_names = load_class_weights(
        csv_path,
        label_field=label_field,
        class_names=ds_info.class_names,
    )

    print("Dataset:", DATASET_NAME)
    print("Train variant:", variant)
    print("Task mode:", task_mode)
    print("Label field:", label_field)
    print("Class names:", class_names)
    print("Class counts:", counts)
    print("Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])

    train_ds = ds_info.dataset_cls(
        str(paths.train_csv),
        str(paths.img_root),
        transform=make_transforms(cfg.image_size, True),
    )
    test_ds = ds_info.dataset_cls(
        str(paths.test_csv),
        str(paths.img_root),
        transform=make_transforms(cfg.image_size, False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    if variant == "base":
        model = DinoV2EmotionVA(
            backbone_name=cfg.backbone_name,
            use_cls_plus_patchmean=True,
            dropout=cfg.dropout,
            num_emotions=cfg.num_emotions,
            va_dims=cfg.va_dims,
        )
        train_base(model, cfg, train_loader, test_loader, task_mode)
        return

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

    if variant == "extended":
        if task_mode == "multitask":
            train_extended_multitask(model, cfg, train_loader, test_loader, class_weights)
        else:
            train_extended_emotion_only(model, cfg, train_loader, test_loader, class_weights)
    else:
        if task_mode == "multitask":
            train_attention_multitask(model, cfg, train_loader, test_loader, class_weights)
        else:
            train_attention_emotion_only(model, cfg, train_loader, test_loader, class_weights)


if __name__ == "__main__":
    main()
