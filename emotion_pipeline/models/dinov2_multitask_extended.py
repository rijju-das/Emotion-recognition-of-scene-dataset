"""
Extended DinoV2 models with novel attention pooling mechanisms.
These classes inherit from the baseline DinoV2EmotionVA and add advanced pooling strategies.
"""

import torch
import torch.nn as nn
from .dinov2_multitask import DinoV2EmotionVA
from .attention_pooling import (
    MultiQueryCrossAttentionPooling,
    HierarchicalAttentionPooling,
    EmotionAwareAttentionPooling
)


class DinoV2EmotionVA_MultiQuery(DinoV2EmotionVA):
    """
    Extended model with Multi-Query Cross-Attention Pooling (MQCAP-EG).
    
    Novel contributions:
    - Multiple learnable emotion-specific query vectors
    - Cross-attention between queries and spatial patch tokens
    - Emotion-guided gating for dynamic query weighting
    
    Expected improvement: +3-5% accuracy over baseline
    """
    
    def __init__(
        self,
        backbone_name: str | None = None,
        dropout: float = 0.4,
        num_queries: int = 4,
        num_heads: int = 8,
        num_emotions: int | None = None,
        va_dims: int | None = None,
    ):
        """
        Args:
            backbone_name: DINOv2 model variant
            dropout: Dropout rate
            num_queries: Number of learnable query vectors
            num_heads: Number of attention heads
        """
        # Don't call super().__init__() - we'll rebuild the model
        nn.Module.__init__(self)
        
        if backbone_name is None:
            backbone_name = "dinov2_vitb14"
        if num_emotions is None:
            num_emotions = 6
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        
        # Novel attention pooling mechanism
        self.attention_pooling = MultiQueryCrossAttentionPooling(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Task heads (single embed_dim, not doubled)
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_emotions)
        )
        if va_dims is None:
            va_dims = 2
        self.va_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, va_dims)
        )
        
        self.num_queries = num_queries
        self.num_heads = num_heads
    
    def forward(self, x):
        """Forward pass with multi-query attention pooling."""
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]          # (B, D)
        patch = feats["x_norm_patchtokens"]     # (B, N, D)
        
        # Multi-query cross-attention pooling
        f, attention_weights = self.attention_pooling(patch, cls)
        
        # Store for visualization
        self.last_attention = attention_weights
        
        logits = self.emotion_head(f)
        va = self.va_head(f)
        return logits, va
    
    def get_attention_maps(self):
        """Retrieve attention maps from last forward pass."""
        return getattr(self, 'last_attention', None)


class DinoV2EmotionVA_Hierarchical(DinoV2EmotionVA):
    """
    Extended model with Hierarchical Attention Pooling (HAP).
    
    Novel contributions:
    - Two-stage attention: local groups → global aggregation
    - Captures multi-scale emotional patterns
    - Hierarchical structure mirrors human visual perception
    
    Expected improvement: +2-4% accuracy over baseline
    """
    
    def __init__(
        self,
        backbone_name: str | None = None,
        dropout: float = 0.4,
        num_groups: int = 4,
        num_heads: int = 8,
        num_emotions: int | None = None,
        va_dims: int | None = None,
    ):
        """
        Args:
            backbone_name: DINOv2 model variant
            dropout: Dropout rate
            num_groups: Number of hierarchical groups
            num_heads: Number of attention heads
        """
        nn.Module.__init__(self)
        
        if backbone_name is None:
            backbone_name = "dinov2_vitb14"
        if num_emotions is None:
            num_emotions = 6
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        
        # Hierarchical attention pooling
        self.attention_pooling = HierarchicalAttentionPooling(
            embed_dim=embed_dim,
            num_groups=num_groups,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Task heads
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_emotions)
        )
        if va_dims is None:
            va_dims = 2
        self.va_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, va_dims)
        )
        
        self.num_groups = num_groups
        self.num_heads = num_heads
    
    def forward(self, x):
        """Forward pass with hierarchical attention pooling."""
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        patch = feats["x_norm_patchtokens"]
        
        # Hierarchical attention pooling
        f, local_attention = self.attention_pooling(patch, cls)
        
        # Store for visualization
        self.last_attention = local_attention
        
        logits = self.emotion_head(f)
        va = self.va_head(f)
        return logits, va
    
    def get_attention_maps(self):
        """Retrieve local attention maps from last forward pass."""
        return getattr(self, 'last_attention', None)


class DinoV2EmotionVA_EmotionAware(DinoV2EmotionVA):
    """
    Extended model with Emotion-Aware Spatial Attention Pooling (EASAP).
    
    Novel contributions:
    - Spatial attention conditioned on preliminary emotion prediction
    - Different emotions guide attention to different regions
    - Explicit emotion-to-spatial-focus modeling
    
    Expected improvement: +3-6% accuracy over baseline
    """
    
    def __init__(
        self,
        backbone_name: str | None = None,
        dropout: float = 0.4,
        num_emotions: int | None = None,
        num_heads: int = 8,
        va_dims: int | None = None,
    ):
        """
        Args:
            backbone_name: DINOv2 model variant
            dropout: Dropout rate
            num_emotions: Number of emotion categories
            num_heads: Number of attention heads
        """
        nn.Module.__init__(self)
        
        if backbone_name is None:
            backbone_name = "dinov2_vitb14"
        if num_emotions is None:
            num_emotions = 6
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        
        # Emotion-aware attention pooling
        self.attention_pooling = EmotionAwareAttentionPooling(
            embed_dim=embed_dim,
            num_emotions=num_emotions,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Task heads
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_emotions)
        )
        if va_dims is None:
            va_dims = 2
        self.va_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, va_dims)
        )
        
        self.num_emotions = num_emotions
        self.num_heads = num_heads
    
    def forward(self, x):
        """Forward pass with emotion-aware spatial attention pooling."""
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        patch = feats["x_norm_patchtokens"]
        
        # Emotion-aware spatial attention (requires CLS token)
        f, emotion_probs, spatial_attention = self.attention_pooling(patch, cls)
        
        # Store for visualization
        self.last_attention = spatial_attention
        self.last_emotion_probs = emotion_probs
        
        logits = self.emotion_head(f)
        va = self.va_head(f)
        return logits, va
    
    def get_attention_maps(self):
        """Retrieve spatial attention maps from last forward pass."""
        return getattr(self, 'last_attention', None)
    
    def get_emotion_probs(self):
        """Retrieve preliminary emotion probabilities."""
        return getattr(self, 'last_emotion_probs', None)


def create_model(
    model_type: str = "baseline",
    backbone_name: str | None = None,
    dropout: float = 0.4,
    use_cls_plus_patchmean: bool = True,
    num_queries: int = 4,
    num_heads: int = 8,
    num_emotions: int | None = None,
    va_dims: int | None = None,
):
    """
    Factory function to create emotion recognition models.
    
    Args:
        model_type: "baseline", "multi_query", "hierarchical", or "emotion_aware"
        backbone_name: DINOv2 model variant
        dropout: Dropout rate
        use_cls_plus_patchmean: For baseline only
        num_queries: Number of queries/groups for extended models
        num_heads: Number of attention heads for extended models
    
    Returns:
        Model instance
    """
    if model_type == "baseline":
        return DinoV2EmotionVA(
            backbone_name=backbone_name,
            use_cls_plus_patchmean=use_cls_plus_patchmean,
            dropout=dropout,
            num_emotions=num_emotions,
            va_dims=va_dims,
        )
    elif model_type == "multi_query":
        return DinoV2EmotionVA_MultiQuery(
            backbone_name=backbone_name,
            dropout=dropout,
            num_queries=num_queries,
            num_heads=num_heads,
            num_emotions=num_emotions,
            va_dims=va_dims,
        )
    elif model_type == "hierarchical":
        return DinoV2EmotionVA_Hierarchical(
            backbone_name=backbone_name,
            dropout=dropout,
            num_groups=num_queries,
            num_heads=num_heads,
            num_emotions=num_emotions,
            va_dims=va_dims,
        )
    elif model_type == "emotion_aware":
        return DinoV2EmotionVA_EmotionAware(
            backbone_name=backbone_name,
            dropout=dropout,
            num_emotions=num_emotions,
            num_heads=num_heads,
            va_dims=va_dims,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
