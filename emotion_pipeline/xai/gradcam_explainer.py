import math
import torch
import torch.nn.functional as F

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
