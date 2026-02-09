from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .extended_configs import PhaseConfig
from ..models.dinov2_multitask import DinoV2EmotionVA
from .multitask_trainer import MultiTaskTrainer

class PhaseRunner:
    def __init__(self, model: DinoV2EmotionVA, trainer: MultiTaskTrainer):
        self.model = model
        self.trainer = trainer

    def _freeze_backbone(self, freeze: bool) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = not freeze

    def run(self, phase: PhaseConfig, patience: int) -> None:
        print("\n" + "=" * 80)
        print(phase.name)
        print("=" * 80)

        self._freeze_backbone(phase.freeze_backbone)

        params = []
        if not phase.freeze_backbone and phase.lr_backbone > 0:
            params.append({"params": self.model.backbone.parameters(), "lr": phase.lr_backbone})
        params.append({"params": self.model.emotion_head.parameters(), "lr": phase.lr_head})
        params.append({"params": self.model.va_head.parameters(), "lr": phase.lr_head})
        
        # Include attention pooling parameters if present
        if hasattr(self.model, 'attention_pooling') and self.model.attention_pooling is not None:
            params.append({"params": self.model.attention_pooling.parameters(), "lr": phase.lr_head})

        optimizer = AdamW(params, weight_decay=phase.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase.epochs, eta_min=phase.eta_min)

        self.trainer.fit(
            optimizer=optimizer,
            epochs=phase.epochs,
            patience=patience,
            scheduler=scheduler,
        )
