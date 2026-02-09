"""
Novel Attention Pooling Mechanisms for Feature Aggregation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryCrossAttentionPooling(nn.Module):
    """
    Multi-Query Cross-Attention Pooling with Emotion-Guided Gating (MQCAP-EG)
    
    Novel contributions:
    1. Multiple learnable emotion-specific query vectors (not just CLS token)
    2. Cross-attention mechanism between queries and spatial patch tokens
    3. Emotion-guided gating to dynamically weight query importance
    4. Hierarchical aggregation combining local and global context
    
    This allows the model to learn multiple "viewpoints" for emotion recognition,
    where each query can focus on different emotional cues (e.g., color distribution,
    composition, salient objects).
    """
    
    def __init__(self, embed_dim: int, num_queries: int = 4, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of input features (from DINOv2)
            num_queries: Number of learnable query vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Learnable emotion-specific query vectors
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.queries)
        
        # Multi-head cross-attention (queries attend to patch tokens)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Emotion-guided gating network
        # Takes aggregated features and produces importance weights for each query
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * num_queries, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_queries),
            nn.Softmax(dim=-1)
        )
        
        # Layer norms for stability
        self.ln_queries = nn.LayerNorm(embed_dim)
        self.ln_output = nn.LayerNorm(embed_dim)
        
        # Projection for final output
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patch_tokens, cls_token=None):
        """
        Args:
            patch_tokens: (B, N, D) - patch tokens from DINOv2
            cls_token: (B, D) - optional CLS token to incorporate
            
        Returns:
            pooled_features: (B, D) - aggregated features for downstream tasks
            attention_weights: (B, num_queries, N) - attention weights for interpretability
        """
        B, N, D = patch_tokens.shape
        
        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # (B, num_queries, D)
        queries = self.ln_queries(queries)
        
        # Cross-attention: queries attend to patch tokens
        # query, key, value format for MultiheadAttention
        attended_queries, attn_weights = self.cross_attention(
            query=queries,              # (B, num_queries, D)
            key=patch_tokens,           # (B, N, D)
            value=patch_tokens,         # (B, N, D)
            need_weights=True,
            average_attn_weights=False
        )  # attended_queries: (B, num_queries, D), attn_weights: (B, num_heads, num_queries, N)
        
        # Average attention weights across heads for interpretability
        attn_weights_avg = attn_weights.mean(dim=1)  # (B, num_queries, N)
        
        # Residual connection
        queries = queries + self.dropout(attended_queries)
        
        # Flatten all queries for gating network
        queries_flat = queries.reshape(B, -1)  # (B, num_queries * D)
        
        # Compute query importance weights using gating network
        query_weights = self.gate_net(queries_flat)  # (B, num_queries)
        
        # Weighted aggregation of queries
        query_weights = query_weights.unsqueeze(-1)  # (B, num_queries, 1)
        pooled = (queries * query_weights).sum(dim=1)  # (B, D)
        
        # Optional: incorporate CLS token if provided
        if cls_token is not None:
            pooled = pooled + cls_token  # Residual connection with CLS
        
        # Final projection and normalization
        pooled = self.ln_output(pooled)
        pooled = self.output_proj(pooled)
        
        return pooled, attn_weights_avg


class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical Attention Pooling (HAP)
    
    Novel approach: Two-stage attention mechanism
    1. Local attention: Group patches and attend within groups
    2. Global attention: Attend across group representatives
    
    This captures both fine-grained local patterns and global composition.
    """
    
    def __init__(self, embed_dim: int, num_groups: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        
        # Group query tokens
        self.group_queries = nn.Parameter(torch.randn(1, num_groups, embed_dim))
        nn.init.xavier_uniform_(self.group_queries)
        
        # Local attention (within groups)
        self.local_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Global attention (across groups)
        self.global_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Final aggregation token
        self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.agg_token)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patch_tokens, cls_token=None):
        """
        Args:
            patch_tokens: (B, N, D)
            cls_token: (B, D) optional
            
        Returns:
            pooled: (B, D)
            local_attn: (B, num_groups, N) - local attention weights
        """
        B, N, D = patch_tokens.shape
        
        # Stage 1: Local attention - group patches
        group_queries = self.group_queries.expand(B, -1, -1)  # (B, num_groups, D)
        group_queries = self.ln1(group_queries)
        
        # Each group query attends to all patches
        group_feats, local_attn = self.local_attention(
            query=group_queries,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=True,
            average_attn_weights=True
        )  # (B, num_groups, D)
        
        group_feats = self.ln2(group_feats + group_queries)
        
        # Stage 2: Global attention - aggregate groups
        agg_token = self.agg_token.expand(B, -1, -1)  # (B, 1, D)
        
        pooled, global_attn = self.global_attention(
            query=agg_token,
            key=group_feats,
            value=group_feats,
            need_weights=True,
            average_attn_weights=True
        )  # (B, 1, D)
        
        pooled = pooled.squeeze(1)  # (B, D)
        
        # Optional CLS residual
        if cls_token is not None:
            pooled = pooled + cls_token
        
        pooled = self.ln3(pooled)
        
        return pooled, local_attn


class EmotionAwareAttentionPooling(nn.Module):
    """
    Emotion-Aware Spatial Attention Pooling (EASAP)
    
    Novel idea: Learn emotion-specific spatial attention patterns
    Each emotion category may focus on different image regions.
    This module learns to generate spatial attention maps conditioned on
    preliminary emotion predictions.
    """
    
    def __init__(self, embed_dim: int, num_emotions: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_emotions = num_emotions
        
        # Preliminary emotion classifier from CLS token
        self.prelim_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_emotions)
        )
        
        # Emotion-conditioned attention query generator
        self.query_generator = nn.Sequential(
            nn.Linear(num_emotions + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, patch_tokens, cls_token):
        """
        Args:
            patch_tokens: (B, N, D)
            cls_token: (B, D) - required for preliminary emotion prediction
            
        Returns:
            pooled: (B, D)
            emotion_probs: (B, num_emotions) - preliminary emotion probabilities
            spatial_attn: (B, 1, N) - spatial attention weights
        """
        B, N, D = patch_tokens.shape
        
        # Get preliminary emotion predictions from CLS token
        emotion_logits = self.prelim_classifier(cls_token)  # (B, num_emotions)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Generate emotion-conditioned query
        query_input = torch.cat([cls_token, emotion_probs], dim=-1)  # (B, D + num_emotions)
        emotion_query = self.query_generator(query_input)  # (B, D)
        emotion_query = self.ln1(emotion_query).unsqueeze(1)  # (B, 1, D)
        
        # Emotion-conditioned spatial attention
        pooled, spatial_attn = self.spatial_attention(
            query=emotion_query,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=True,
            average_attn_weights=True
        )  # (B, 1, D), (B, 1, N)
        
        pooled = pooled.squeeze(1)  # (B, D)
        pooled = self.ln2(pooled + cls_token)  # Residual with CLS
        
        return pooled, emotion_probs, spatial_attn
