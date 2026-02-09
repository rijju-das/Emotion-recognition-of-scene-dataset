import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:  # Keep optional dependency
    GradCAM = None
    ClassifierOutputTarget = None

class GradCAMExplainer:
    """Grad-CAM-style attribution over ViT patch tokens."""

    def __init__(self, model, head: str = "emotion"):
        self.model = model
        self.head = head

    def _forward_with_tokens(self, x: torch.Tensor):
        feats = self.model.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        patch = feats["x_norm_patchtokens"]

        if hasattr(self.model, "attention_pooling"):
            f, *_ = self.model.attention_pooling(patch, cls)
        else:
            if getattr(self.model, "use_cls_plus_patchmean", False):
                patch_mean = patch.mean(dim=1)
                f = torch.cat([cls, patch_mean], dim=-1)
            else:
                f = cls

        logits = self.model.emotion_head(f)
        va = self.model.va_head(f)
        return logits, va, patch

    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns a (H, W) heatmap for each input in the batch.
        """
        x = x.requires_grad_(True)
        logits, va, patch = self._forward_with_tokens(x)

        if self.head == "emotion":
            if target is None:
                target = logits.argmax(dim=1)
            score = logits[torch.arange(logits.size(0)), target]
        else:
            idx = 0 if target is None else int(target)
            score = va[:, idx]

        grads = torch.autograd.grad(score.sum(), patch, retain_graph=False)[0]
        cam = (grads * patch).sum(dim=2)
        cam = F.relu(cam)

        bsz, n_tokens = cam.shape
        grid = int(math.sqrt(n_tokens))
        if grid * grid != n_tokens:
            cam = cam.view(bsz, 1, 1, n_tokens)
            cam = F.interpolate(cam, size=(1, n_tokens), mode="bilinear", align_corners=False)
            cam = cam.view(bsz, 1, 1, n_tokens)
        else:
            cam = cam.view(bsz, 1, grid, grid)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        cam_min = cam.flatten(1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()


class ViTGradCAMExplainer:
    """PyTorch-Grad-CAM wrapper for ViT using reshape_transform."""

    def __init__(self, model, head: str = "emotion", target_layer=None):
        if GradCAM is None:
            raise ImportError(
                "pytorch-grad-cam is not installed. Install with: pip install grad-cam"
            )
        self.model = model
        self.head = head
        self.target_layer = target_layer or self._default_target_layer()

    def _default_target_layer(self):
        # DINOv2 / ViT typically exposes transformer blocks in backbone.blocks
        if hasattr(self.model.backbone, "blocks") and len(self.model.backbone.blocks) > 0:
            return self.model.backbone.blocks[-1]
        raise ValueError("Could not infer target_layer; please provide one explicitly.")

    @staticmethod
    def reshape_transform(x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), where N = 1 + num_patches (CLS + patches)
        if x.dim() != 3:
            raise ValueError(f"Expected tokens shape (B, N, C), got {tuple(x.shape)}")
        x = x[:, 1:, :]
        grid = int(math.sqrt(x.shape[1]))
        if grid * grid != x.shape[1]:
            raise ValueError("Token count is not a perfect square; cannot reshape to grid.")
        x = x.reshape(x.size(0), grid, grid, x.size(2))
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

    def explain(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Grad-CAM expects class targets; we use emotion logits by default.
        if self.head != "emotion":
            raise ValueError("ViTGradCAMExplainer currently supports head='emotion' only.")

        if target is None:
            with torch.no_grad():
                feats = self.model.backbone.forward_features(x)
                cls = feats["x_norm_clstoken"]
                patch = feats["x_norm_patchtokens"]
                if hasattr(self.model, "attention_pooling"):
                    f, *_ = self.model.attention_pooling(patch, cls)
                else:
                    if getattr(self.model, "use_cls_plus_patchmean", False):
                        patch_mean = patch.mean(dim=1)
                        f = torch.cat([cls, patch_mean], dim=-1)
                    else:
                        f = cls
                logits = self.model.emotion_head(f)
                target = logits.argmax(dim=1)

        targets = [ClassifierOutputTarget(int(t)) for t in target]

        cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        )

        grayscale_cam = cam(input_tensor=x, targets=targets)
        return torch.from_numpy(grayscale_cam)
