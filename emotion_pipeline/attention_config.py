"""
Configuration for Attention-Based Emotion Recognition Models

This configuration is specifically for training models with novel attention pooling mechanisms.
Separate from the baseline config.py and extended_config.py.
"""

from dataclasses import dataclass
from .config import TrainConfig

@dataclass(frozen=True)
class AttentionModelConfig(TrainConfig):
    """
    Configuration for attention-based models.
    Inherits all baseline settings from TrainConfig.
    
    Novel Parameters:
    - model_type: Choose attention mechanism
    - num_queries: Number of learnable queries/groups
    - num_attention_heads: Number of attention heads
    """
    
    # Overrides from baseline
    lr_head_probe: float = 1e-5  # Linear probe learning rate
    label_smoothing: float = 0.1  # Label smoothing for better generalization
    
    # === Attention Model Selection ===
    model_type: str = "multi_query"
    # Options:
    #   "baseline"        - Simple CLS + patch mean (no attention)
    #   "multi_query"     - Multi-Query Cross-Attention Pooling (MQCAP-EG)
    #   "hierarchical"    - Hierarchical Attention Pooling (HAP)
    #   "emotion_aware"   - Emotion-Aware Spatial Attention Pooling (EASAP)
    
    # === Attention Hyperparameters ===
    num_queries: int = 2              # Number of learnable queries (for multi_query, hierarchical)
    num_attention_heads: int = 4      # Number of attention heads
    
    # === Additional Training Parameters ===
    use_mixed_precision: bool = True   # Use automatic mixed precision training
    gradient_clip_norm: float = 1.0    # Gradient clipping to prevent explosions
    
    def __post_init__(self):
        """Validate configuration"""
        valid_model_types = ["baseline", "multi_query", "hierarchical", "emotion_aware"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}, got {self.model_type}")
        
        if self.num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {self.num_queries}")
        
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {self.num_attention_heads}")
        
        print(f"✓ AttentionModelConfig initialized:")
        print(f"  Model type: {self.model_type}")
        if self.model_type != "baseline":
            print(f"  Num queries: {self.num_queries}")
            print(f"  Attention heads: {self.num_attention_heads}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Image size: {self.image_size}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Early stop lambda: {self.early_stop_lambda}")
