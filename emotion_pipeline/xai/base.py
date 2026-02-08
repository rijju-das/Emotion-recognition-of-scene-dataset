from abc import ABC, abstractmethod
import torch

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns attribution map aligned with input x (B, C, H, W).
        """
        ...
