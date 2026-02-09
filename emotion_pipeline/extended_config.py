from dataclasses import dataclass
from .config import TrainConfig

@dataclass(frozen=True)
class ExtendedTrainConfig(TrainConfig):
    lr_head_probe: float = 1e-5
    weight_decay_probe: float = 1.0
    label_smoothing: float = 0.1
    eta_min_probe: float = 1e-6
    eta_min_finetune: float = 1e-7
    
    # Novel attention-based model selection
    model_type: str = "multi_query"  # "baseline", "multi_query", "hierarchical", "emotion_aware"
    
    # Attention pooling configurations (for extended models)
    num_queries: int = 4  # Number of query vectors for multi_query or groups for hierarchical
    num_attention_heads: int = 8  # Number of attention heads
