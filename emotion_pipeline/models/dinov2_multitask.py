import torch
import torch.nn as nn
from .base import BaseModel

class DinoV2EmotionVA(BaseModel):
    def __init__(self, backbone_name: str = "dinov2_vitb14", use_cls_plus_patchmean: bool = True, dropout: float = 0.4):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.use_cls_plus_patchmean = use_cls_plus_patchmean

        embed_dim = getattr(self.backbone, "embed_dim", 768)
        feat_dim = embed_dim * 2 if use_cls_plus_patchmean else embed_dim

        # Add dropout to heads for regularization
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 6)
        )
        self.va_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]          # (B, D)

        if self.use_cls_plus_patchmean:
            patch = feats["x_norm_patchtokens"] # (B, N, D)
            patch_mean = patch.mean(dim=1)      # (B, D)
            f = torch.cat([cls, patch_mean], dim=-1)
        else:
            f = cls

        logits = self.emotion_head(f)
        va = self.va_head(f)
        return logits, va
