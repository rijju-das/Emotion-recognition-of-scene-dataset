import torch
import torch.nn.functional as F

class GradCAMExplainer:
    """Grad-CAM: region-level importance from conv layer activations."""
    def __init__(self, model, head: str = "emotion", layer_name: str = "backbone"):
        self.model = model
        self.head = head
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward & backward hooks on backbone."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Assumes model.backbone exists
        if hasattr(self.model, self.layer_name):
            layer = getattr(self.model, self.layer_name)
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, 1, H, W) heatmap
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
        
        # Grad-CAM: weight activations by gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H_feat, W_feat)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        
        return cam.detach()