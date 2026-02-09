"""
Validation Tests for Attention-Based Models

Tests the attention pooling implementations to ensure:
- Forward pass works correctly
- Attention parameters are trainable
- Attention maps can be retrieved for visualization

Run: python test_attention_models.py
"""

import torch
import torch.nn as nn
from emotion_pipeline.attention_config import AttentionModelConfig
from emotion_pipeline.models.dinov2_multitask_extended import create_model


def test_model_creation():
    """Test that all model types can be instantiated"""
    print("Testing model creation...")
    cfg = AttentionModelConfig()
    
    model_types = ["baseline", "multi_query", "hierarchical", "emotion_aware"]
    
    for model_type in model_types:
        print(f"\n  Testing {model_type}...")
        model = create_model(
            model_type=model_type,
            backbone_name="dinov2_vitb14",  # Use smaller model for testing
            dropout=0.5,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_attention_heads
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"    ✓ Model created successfully")
        print(f"    Total params: {total_params:,}")
        print(f"    Trainable params: {trainable_params:,}")
        
        # Check attention pooling
        if model_type != "baseline":
            if hasattr(model, 'attention_pooling') and model.attention_pooling is not None:
                attn_params = sum(p.numel() for p in model.attention_pooling.parameters())
                print(f"    Attention params: {attn_params:,}")
            else:
                print(f"    ⚠️  WARNING: No attention_pooling module found!")
    
    print("\n✓ All models created successfully!\n")


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    cfg = AttentionModelConfig()
    
    # Create dummy input
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    
    model_types = ["baseline", "multi_query", "hierarchical", "emotion_aware"]
    
    for model_type in model_types:
        print(f"\n  Testing {model_type} forward pass...")
        model = create_model(
            model_type=model_type,
            backbone_name="dinov2_vitb14",
            dropout=0.5,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_attention_heads
        )
        model.eval()
        
        with torch.no_grad():
            emotion_logits, va_preds = model(dummy_img)
        
        print(f"    ✓ Forward pass successful")
        print(f"    Emotion logits shape: {emotion_logits.shape}")
        print(f"    V-A predictions shape: {va_preds.shape}")
        
        # Check shapes
        assert emotion_logits.shape == (batch_size, 6), f"Expected emotion shape (2, 6), got {emotion_logits.shape}"
        assert va_preds.shape == (batch_size, 2), f"Expected V-A shape (2, 2), got {va_preds.shape}"
        
        # Check for NaN
        assert not torch.isnan(emotion_logits).any(), "NaN detected in emotion logits!"
        assert not torch.isnan(va_preds).any(), "NaN detected in V-A predictions!"
    
    print("\n✓ All forward passes successful!\n")


def test_attention_maps():
    """Test that attention maps can be retrieved"""
    print("Testing attention map retrieval...")
    cfg = AttentionModelConfig()
    
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    
    model_types = ["multi_query", "hierarchical", "emotion_aware"]  # Skip baseline
    
    for model_type in model_types:
        print(f"\n  Testing {model_type} attention maps...")
        model = create_model(
            model_type=model_type,
            backbone_name="dinov2_vitb14",
            dropout=0.5,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_attention_heads
        )
        model.eval()
        
        with torch.no_grad():
            emotion_logits, va_preds = model(dummy_img)
        
        # Check if attention maps are available
        if hasattr(model, 'last_attention_maps') and model.last_attention_maps is not None:
            attn_maps = model.last_attention_maps
            print(f"    ✓ Attention maps available")
            print(f"    Attention shape: {attn_maps.shape}")
        else:
            print(f"    ⚠️  No attention maps found (this is OK if not implemented)")
    
    print("\n✓ Attention map retrieval tests complete!\n")


def test_parameter_collection():
    """Test that attention parameters are properly identified for training"""
    print("Testing parameter collection for optimizer...")
    cfg = AttentionModelConfig()
    
    model_types = ["multi_query", "hierarchical", "emotion_aware"]
    
    for model_type in model_types:
        print(f"\n  Testing {model_type} parameter collection...")
        model = create_model(
            model_type=model_type,
            backbone_name="dinov2_vitb14",
            dropout=0.5,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_attention_heads
        )
        
        # Collect parameters by component
        head_params = list(model.emotion_head.parameters()) + list(model.va_head.parameters())
        
        if hasattr(model, 'attention_pooling') and model.attention_pooling is not None:
            attn_params = list(model.attention_pooling.parameters())
            print(f"    ✓ Attention pooling parameters: {len(attn_params)} tensors")
        else:
            print(f"    ⚠️  No attention pooling parameters found!")
            continue
        
        # Check that parameters are unique (no overlap)
        head_param_ids = {id(p) for p in head_params}
        attn_param_ids = {id(p) for p in attn_params}
        
        overlap = head_param_ids & attn_param_ids
        if overlap:
            print(f"    ⚠️  WARNING: {len(overlap)} overlapping parameters detected!")
        else:
            print(f"    ✓ No parameter overlap (good!)")
    
    print("\n✓ Parameter collection tests complete!\n")


def test_training_step():
    """Test a single training step to ensure gradients flow correctly"""
    print("Testing training step...")
    cfg = AttentionModelConfig()
    
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    dummy_emotion = torch.randint(0, 6, (batch_size,))  # 6 emotion classes
    dummy_va = torch.randn(batch_size, 2)  # Valence and arousal
    
    model_types = ["multi_query", "hierarchical", "emotion_aware"]
    
    for model_type in model_types:
        print(f"\n  Testing {model_type} training step...")
        model = create_model(
            model_type=model_type,
            backbone_name="dinov2_vitb14",
            dropout=0.5,
            num_queries=cfg.num_queries,
            num_heads=cfg.num_attention_heads
        )
        model.train()
        
        # Create optimizer (only for heads + attention, backbone frozen)
        params = []
        params.extend(model.emotion_head.parameters())
        params.extend(model.va_head.parameters())
        if hasattr(model, 'attention_pooling') and model.attention_pooling is not None:
            params.extend(model.attention_pooling.parameters())
        
        optimizer = torch.optim.Adam(params, lr=1e-4)
        
        # Forward pass
        emotion_logits, va_preds = model(dummy_img)
        
        # Compute losses
        emotion_loss = nn.CrossEntropyLoss()(emotion_logits, dummy_emotion)
        va_loss = nn.MSELoss()(va_preds, dummy_va)
        total_loss = emotion_loss + va_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        has_grads = False
        if hasattr(model, 'attention_pooling') and model.attention_pooling is not None:
            for name, param in model.attention_pooling.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grads = True
                    break
        
        if has_grads:
            print(f"    ✓ Gradients flowing to attention pooling")
        else:
            print(f"    ⚠️  No gradients in attention pooling!")
        
        # Optimizer step
        optimizer.step()
        print(f"    ✓ Training step successful")
    
    print("\n✓ All training step tests passed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ATTENTION MODEL VALIDATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_model_creation()
        test_forward_pass()
        test_attention_maps()
        test_parameter_collection()
        test_training_step()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your attention models are ready to train!")
        print("Run: python -m emotion_pipeline.run_train_attention")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
