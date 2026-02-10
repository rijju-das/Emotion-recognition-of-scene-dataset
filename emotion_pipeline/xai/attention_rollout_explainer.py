import torch
import torch.nn.functional as F

class AttentionRolloutExplainer:
    """Attention Rollout for ViT - native attention-based attribution."""
    
    def __init__(self, model, discard_ratio: float = 0.0):
        """
        Args:
            model: ViT model
            discard_ratio: Threshold for small attention values (0.9 = keep top 10%)
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Hook into attention layers to capture weights."""
        def get_attention(module, input, output):
            # Prefer computing attention from qkv when available (e.g., MemEffAttention).
            attn = None
            if len(input) > 0 and hasattr(module, "qkv") and hasattr(module, "num_heads"):
                x = input[0]
                if x is not None and x.dim() == 3:
                    try:
                        qkv = module.qkv(x)
                        bsz, seq_len, three_dim = qkv.shape
                        num_heads = int(module.num_heads)
                        head_dim = three_dim // (3 * num_heads)
                        qkv = qkv.reshape(bsz, seq_len, 3, num_heads, head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]
                        scale = getattr(module, "scale", head_dim ** -0.5)
                        attn = (q @ k.transpose(-2, -1)) * scale
                        attn = attn.softmax(dim=-1)
                    except Exception:
                        attn = None

            if attn is None:
                # Fall back to any attention-like tensor in output/input
                candidate = output
                if isinstance(candidate, (tuple, list)) and len(candidate) > 0:
                    candidate = candidate[0]
                if candidate is None and len(input) > 0:
                    candidate = input[0]
                if candidate is not None and candidate.dim() == 4:
                    attn = candidate

            if attn is None or attn.dim() != 4:
                return
            self.attention_maps.append(attn.detach())
        
        if hasattr(self.model.backbone, 'blocks'):
            for block in self.model.backbone.blocks:
                if hasattr(block, 'attn'):
                    # DINOv2/timm Attention usually exposes attn_drop on attention weights
                    attn_drop = getattr(block.attn, 'attn_drop', None)
                    if hasattr(attn_drop, 'register_forward_hook'):
                        attn_drop.register_forward_hook(get_attention)
                    elif hasattr(block.attn, 'register_forward_hook'):
                        block.attn.register_forward_hook(get_attention)
    
    def _rollout(self, attention_masks: torch.Tensor, last_k_layers: int = 4) -> torch.Tensor:
        """
        attention_masks: (L, H, S, S)
        returns joint attention: (S, S)
        """
        L, H, S, _ = attention_masks.shape
        device = attention_masks.device

        # Use only last K layers (prevents collapse)
        start = max(0, L - last_k_layers)
        attn_stack = attention_masks[start:]  # (K, H, S, S)

        result = torch.eye(S, device=device)

        for attn in attn_stack:
            # fuse heads
            attn = attn.mean(dim=0)  # (S, S)

            # IMPORTANT: add residual connection and renormalize
            attn = attn + torch.eye(S, device=device)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # OPTIONAL: discard, but do it gently and per-row (NOT global)
            if self.discard_ratio > 0:
                k = max(int(S * (1 - self.discard_ratio)), 1)  # tokens to keep per row
                topk_vals, _ = attn.topk(k, dim=-1)
                thr = topk_vals[:, -1].unsqueeze(-1)  # per-row threshold
                attn = torch.where(attn >= thr, attn, torch.zeros_like(attn))
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # IMPORTANT: correct order for CLS flow
            result = result @ attn

        return result

    
    def explain(self, x: torch.Tensor, target=None):
        """
        Returns (B, seq_len) attention rollout for CLS token to each patch.
        
        Args:
            x: Input image (B, 3, H, W)
            target: Emotion class (unused for attention rollout, just for consistency)
        
        Returns:
            Heatmap (B, H, W) resized to image space
        """
        self.attention_maps.clear()
        
        # Forward pass (captures attention via hooks)
        with torch.no_grad():
            feats = self.model.backbone.forward_features(x)
        
        if not self.attention_maps:
            raise RuntimeError("No attention maps captured!")
        
        # Keep only valid attention tensors
        valid = [a for a in self.attention_maps if a is not None and a.dim() == 4]
        if not valid:
            shapes = [tuple(a.shape) for a in self.attention_maps if a is not None]
            raise ValueError(
                "No attention maps with shape (B, H, S, S) captured. "
                f"Captured shapes: {shapes}. "
                "Your ViT attention module may not expose attention weights."
            )

        # Stack attention maps: list of (B, H, S, S)
        attention_stacked = torch.stack(valid)  # (num_layers, B, H, S, S)
        
        # Rollout for each sample in batch
        batch_size = x.size(0)
        rollout = []
        
        for b in range(batch_size):
            attention_b = attention_stacked[:, b]  # (num_layers, num_heads, seq_len, seq_len)
            result = self._rollout(attention_b)
            
            # CLS token (index 0) attention to patches
            num_reg = getattr(self.model.backbone, "num_register_tokens", 0)
            cls_attention = result[0, 1 + num_reg:]  # drop CLS + reg tokens

            # cls_attention = result[0, 1:]  # Skip CLS, get patches only
            rollout.append(cls_attention)
        
        rollout = torch.stack(rollout)  # (B, num_patches)
        
        # Reshape to grid and upsampling to image size
        seq_len = rollout.size(1)
        grid_size = int(seq_len ** 0.5)
        rollout = rollout.view(batch_size, grid_size, grid_size)
        rollout = rollout.unsqueeze(1)  # (B, 1, grid, grid)
        
        # Upsample to image size
        rollout = F.interpolate(
            rollout, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        rollout = rollout.squeeze(1)  # (B, H, W)
        
        # Normalize
        rollout_min = rollout.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        rollout_max = rollout.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        rollout = (rollout - rollout_min) / (rollout_max - rollout_min + 1e-8)
        
        return rollout.detach()