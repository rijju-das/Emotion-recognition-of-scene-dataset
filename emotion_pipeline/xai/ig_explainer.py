# pip install captum
import torch
from captum.attr import IntegratedGradients
from .base import BaseExplainer

class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, model, head: str = "emotion"):
        """
        head: "emotion" or "va"
        - emotion: explain logits for a target class
        - va: explain valence or arousal dimension (target should specify index)
        """
        self.model = model
        self.head = head
        self.ig = IntegratedGradients(self._forward)

    def _forward(self, x, target_index=None):
        logits, va = self.model(x)
        if self.head == "emotion":
            return logits
        # VA: choose one dimension, default 0 (valence)
        idx = 0 if target_index is None else int(target_index)
        return va[:, idx]

    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """
        emotion head: target = class indices shape (B,)
        va head: target = scalar index (0 or 1) OR tensor broadcastable
        """
        x = x.requires_grad_(True)
        if self.head == "emotion":
            attributions = self.ig.attribute(x, target=target)
        else:
            # target is dimension index (0/1); Captum passes it to forward via additional args
            attributions = self.ig.attribute(x, additional_forward_args=(target,))
        return attributions
