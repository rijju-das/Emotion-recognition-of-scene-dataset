import torch
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float

class SHAPSuperpixelExplainer:
    """SHAP with superpixels: faithful perturbation-based attribution."""
    def __init__(self, model, head: str = "emotion", n_superpixels: int = 50):
        self.model = model
        self.head = head
        self.n_superpixels = n_superpixels
    
    def _get_superpixels(self, img_np: np.ndarray) -> np.ndarray:
        """Segment image into superpixels using felzenszwalb."""
        # img_np: (H, W, 3) in [0, 1]
        segments = felzenszwalb(img_np, scale=100, sigma=0.5, min_size=20)
        return segments
    
    def _mask_superpixels(self, x: torch.Tensor, segments: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """
        x: (3, H, W) tensor
        segments: (H, W) superpixel IDs
        mask: (n_superpixels,) binary mask indicating which superpixels to keep
        """
        x_masked = x.clone()
        for sp_id in range(segments.max() + 1):
            if not mask[sp_id]:
                x_masked[:, segments == sp_id] = 0  # zero out masked superpixels
        return x_masked
    
    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None, n_samples: int = 100) -> dict:
        """
        x: (1, C, H, W) single image
        Returns: dict with 'attributions' (H, W) and 'segments'
        """
        x = x.squeeze(0)  # (C, H, W)
        x_np = x.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)  # normalize to [0, 1]
        
        segments = self._get_superpixels(x_np)
        n_sp = segments.max() + 1
        
        attributions = np.zeros(n_sp)
        baseline = torch.zeros_like(x)
        
        # SHAP: average effect of including each superpixel
        for _ in range(n_samples):
            # Random coalition
            mask = np.random.rand(n_sp) > 0.5
            x_masked = self._mask_superpixels(x, segments, mask)
            
            with torch.no_grad():
                logits, va = self.model(x_masked.unsqueeze(0).to(x.device))
                if self.head == "emotion":
                    if target is None:
                        target = logits.argmax(dim=1)
                    score_on = logits[0, target[0]].item()
                else:
                    idx = 0 if target is None else int(target)
                    score_on = va[0, idx].item()
            
            # Also evaluate with baseline
            with torch.no_grad():
                logits_base, va_base = self.model(baseline.unsqueeze(0).to(x.device))
                if self.head == "emotion":
                    score_base = logits_base[0, target[0]].item() if target is not None else 0
                else:
                    idx = 0 if target is None else int(target)
                    score_base = va_base[0, idx].item()
            
            attributions[mask] += (score_on - score_base) / n_samples
        
        # Map superpixel attributions back to pixel space
        pixel_attr = attributions[segments]
        
        return {
            "attributions": pixel_attr,
            "segments": segments,
            "n_superpixels": n_sp
        }