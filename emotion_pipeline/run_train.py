import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import Paths, TrainConfig
from .data.emotion6 import Emotion6Dataset
from .models.dinov2_multitask import DinoV2EmotionVA
from .training.multitask_trainer import MultiTaskTrainer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
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
    paths = Paths()
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    train_ds = Emotion6Dataset(str(paths.train_csv), str(paths.img_root), transform=make_transforms(cfg.image_size, True))
    test_ds  = Emotion6Dataset(str(paths.test_csv),  str(paths.img_root), transform=make_transforms(cfg.image_size, False))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = DinoV2EmotionVA(backbone_name=cfg.backbone_name, use_cls_plus_patchmean=True)

    # Phase A: linear probe
    freeze_backbone(model, True)
    optimizer = torch.optim.AdamW(
        list(model.emotion_head.parameters()) + list(model.va_head.parameters()),
        lr=cfg.lr_head, weight_decay=cfg.weight_decay
    )
    trainer = MultiTaskTrainer(model, train_loader, test_loader, cfg.device, cfg.lam_va)
    trainer.fit(optimizer=optimizer, epochs=cfg.epochs_probe)

    # Phase B: partial fine-tune (simple version: unfreeze all backbone)
    freeze_backbone(model, False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.emotion_head.parameters(), "lr": cfg.lr_head},
            {"params": model.va_head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay
    )
    trainer.fit(optimizer=optimizer, epochs=cfg.epochs_finetune)

    torch.save(model.state_dict(), "dinov2_emotion_va.pt")
    print("Saved: dinov2_emotion_va.pt")

if __name__ == "__main__":
    main()
