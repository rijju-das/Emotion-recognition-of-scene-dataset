import torch
import torch.nn.functional as F

class GradientInputExplainer:
    """Gradient × Input: simple baseline pixel-level attribution."""
    def __init__(self, model, head: str = "emotion"):
        self.model = model
        self.head = head
    
    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C, H, W) image tensor
        target: class indices (B,) for emotion head
        Returns: (B, C, H, W) attribution map
        """
        x = x.clone().detach().requires_grad_(True)
        logits, va = self.model(x)
        
        if self.head == "emotion":
            if target is None:
                target = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, target)
        else:
            idx = 0 if target is None else int(target)
            loss = va[:, idx].sum()
        
        loss.backward()
        grad = x.grad
        attribution = grad * x  # element-wise product
        return attribution.detach()