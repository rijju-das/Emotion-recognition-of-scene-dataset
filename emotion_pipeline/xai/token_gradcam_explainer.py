import torch
import torch.nn.functional as F


class TokenGradCAMExplainer:
    """
    ViT Grad-CAM-like attribution using patch tokens, then converts to a TOP-K patch mask.

    Returns:
      - mask_up: (B, 1, H, W) in {0,1} if mode="binary"
      - or a weighted mask in [0,1] if mode="weighted"
    """
    def __init__(self, model, keep_ratio: float = 0.30, mode: str = "binary", smooth: bool = False):
        """
        keep_ratio: fraction of patches to keep (0.30 -> top 30%)
        mode:
          - "binary": kept patches=1 else 0
          - "weighted": kept patches retain normalized scores (more informative)
        """
        assert 0 < keep_ratio <= 1.0
        assert mode in ("binary", "weighted")
        self.model = model
        self.keep_ratio = keep_ratio
        self.mode = mode
        self.smooth = smooth

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits.argmax(dim=1)

    def explain(self, x: torch.Tensor, target: torch.Tensor | int | None = None) -> torch.Tensor:
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)

        feats = self.model.backbone.forward_features(x)
        patch = feats["x_norm_patchtokens"]     # (B, N, D)
        cls = feats["x_norm_clstoken"]          # (B, D)
        patch.retain_grad()

        if hasattr(self.model, "attention_pooling"):
            f, *_ = self.model.attention_pooling(patch, cls)
        else:
            if getattr(self.model, "use_cls_plus_patchmean", False):
                patch_mean = patch.mean(dim=1)     # (B, D)
                f = torch.cat([cls, patch_mean], dim=-1)
            else:
                f = cls

        logits = self.model.emotion_head(f)

        if target is None:
            target = logits.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)

        score = logits.gather(1, target[:, None]).squeeze(1)  # (B,)
        grads = torch.autograd.grad(score.sum(), patch, retain_graph=False, allow_unused=True)[0]
        if grads is None:
            raise RuntimeError("TokenGradCAMExplainer: no gradients for patch tokens. Check model forward path.")

        # token importance (B, N)
        token_imp = (grads * patch).sum(dim=-1)
        token_imp = F.relu(token_imp)

        # top-k per sample
        bsz, n_tokens = token_imp.shape
        k = max(int(n_tokens * self.keep_ratio), 1)

        topk_vals, _ = token_imp.topk(k, dim=1)          # (B, k)
        thr = topk_vals[:, -1].unsqueeze(1)              # (B, 1)

        keep = token_imp >= thr                          # (B, N)

        if self.mode == "binary":
            mask = keep.float()
        else:
            # weighted: normalize token_imp to [0,1], then keep only top-k
            imp = token_imp.clone()
            imp_min = imp.min(dim=1, keepdim=True).values
            imp_max = imp.max(dim=1, keepdim=True).values
            imp = (imp - imp_min) / (imp_max - imp_min + 1e-12)
            mask = imp * keep.float()

        # reshape to patch grid
        grid = int(n_tokens ** 0.5)
        if grid * grid != n_tokens:
            raise ValueError(f"N={n_tokens} patch tokens is not a square; cannot reshape to grid.")
        mask = mask.view(bsz, 1, grid, grid)

        # upsample to input size
        _, _, height, width = x.shape
        interp_mode = "bilinear" if self.smooth else "nearest"
        mask_up = F.interpolate(mask, size=(height, width), mode=interp_mode, align_corners=False)

        return mask_up.detach()
