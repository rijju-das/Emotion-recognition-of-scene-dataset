"""
Visualize Novel Attention Pooling Mechanisms

This script demonstrates the attention patterns learned by different pooling strategies:
- Multi-Query Cross-Attention Pooling (MQCAP-EG)
- Hierarchical Attention Pooling (HAP)  
- Emotion-Aware Spatial Attention Pooling (EASAP)
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

from config import Paths
from extended_config import ExtendedTrainConfig
from models.dinov2_multitask_extended import create_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(checkpoint_path: str, model_type: str, device: str = "cuda"):
    """Load model with specified model type."""
    cfg = ExtendedTrainConfig()
    model = create_model(
        model_type=model_type,
        backbone_name=cfg.backbone_name,
        dropout=cfg.dropout,
        use_cls_plus_patchmean=True,
        num_queries=cfg.num_queries,
        num_heads=cfg.num_attention_heads
    )
    
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ No checkpoint found, using random initialization")
    
    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess_image(img_path: Path, image_size: int = 224):
    """Load and preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor, img_pil


def visualize_multi_query_attention(model, img_tensor, img_pil, save_path: str = "attention_multi_query.png"):
    """Visualize Multi-Query Cross-Attention patterns."""
    with torch.no_grad():
        logits, va = model(img_tensor)
        attention = model.get_attention_maps()  # (1, num_queries, N)
    
    if attention is None:
        print("No attention maps available")
        return
    
    attention = attention.squeeze(0).cpu().numpy()  # (num_queries, N)
    num_queries, num_patches = attention.shape
    
    # Reshape attention to spatial grid (assuming square patches)
    grid_size = int(np.sqrt(num_patches))
    attention_spatial = attention.reshape(num_queries, grid_size, grid_size)
    
    # Get prediction
    pred_class = logits.argmax(dim=1).item()
    emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness"]
    pred_emotion = emotion_labels[pred_class]
    
    # Plot
    fig, axes = plt.subplots(2, num_queries // 2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(img_pil)
    axes[0].set_title(f"Original\nPred: {pred_emotion}", fontsize=10)
    axes[0].axis("off")
    
    # Show each query's attention
    for q in range(num_queries):
        axes[q + 1].imshow(img_pil, alpha=0.5)
        im = axes[q + 1].imshow(attention_spatial[q], cmap="hot", alpha=0.5)
        axes[q + 1].set_title(f"Query {q + 1}", fontsize=10)
        axes[q + 1].axis("off")
        plt.colorbar(im, ax=axes[q + 1], fraction=0.046)
    
    # Hide unused subplots
    for q in range(num_queries + 1, len(axes)):
        axes[q].axis("off")
    
    plt.suptitle("Multi-Query Cross-Attention Pooling (MQCAP-EG)\nEach query focuses on different emotional cues", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_hierarchical_attention(model, img_tensor, img_pil, save_path: str = "attention_hierarchical.png"):
    """Visualize Hierarchical Attention patterns."""
    with torch.no_grad():
        logits, va = model(img_tensor)
        local_attention = model.get_attention_maps()  # (1, num_groups, N)
    
    if local_attention is None:
        print("No attention maps available")
        return
    
    local_attention = local_attention.squeeze(0).cpu().numpy()  # (num_groups, N)
    num_groups, num_patches = local_attention.shape
    
    # Reshape to spatial
    grid_size = int(np.sqrt(num_patches))
    attention_spatial = local_attention.reshape(num_groups, grid_size, grid_size)
    
    pred_class = logits.argmax(dim=1).item()
    emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness"]
    pred_emotion = emotion_labels[pred_class]
    
    # Plot
    fig, axes = plt.subplots(2, num_groups // 2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    
    axes[0].imshow(img_pil)
    axes[0].set_title(f"Original\nPred: {pred_emotion}", fontsize=10)
    axes[0].axis("off")
    
    for g in range(num_groups):
        axes[g + 1].imshow(img_pil, alpha=0.5)
        im = axes[g + 1].imshow(attention_spatial[g], cmap="viridis", alpha=0.5)
        axes[g + 1].set_title(f"Group {g + 1}", fontsize=10)
        axes[g + 1].axis("off")
        plt.colorbar(im, ax=axes[g + 1], fraction=0.046)
    
    for g in range(num_groups + 1, len(axes)):
        axes[g].axis("off")
    
    plt.suptitle("Hierarchical Attention Pooling (HAP)\nLocal groups capture different spatial regions", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_emotion_aware_attention(model, img_tensor, img_pil, save_path: str = "attention_emotion_aware.png"):
    """Visualize Emotion-Aware Spatial Attention."""
    with torch.no_grad():
        logits, va = model(img_tensor)
        spatial_attention = model.get_attention_maps()  # (1, 1, N)
        emotion_probs = model.get_emotion_probs()  # (1, 6)
    
    if spatial_attention is None:
        print("No attention maps available")
        return
    
    spatial_attention = spatial_attention.squeeze().cpu().numpy()  # (N,)
    num_patches = spatial_attention.shape[0]
    
    grid_size = int(np.sqrt(num_patches))
    attention_spatial = spatial_attention.reshape(grid_size, grid_size)
    
    emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness"]
    pred_class = logits.argmax(dim=1).item()
    pred_emotion = emotion_labels[pred_class]
    
    emotion_probs = emotion_probs.squeeze().cpu().numpy()
    
    # Plot
    fig = plt.figure(figsize=(15, 5))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_pil)
    ax1.set_title(f"Original Image\nPredicted: {pred_emotion}", fontsize=12)
    ax1.axis("off")
    
    # Attention heatmap
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(img_pil, alpha=0.5)
    im = ax2.imshow(attention_spatial, cmap="hot", alpha=0.6)
    ax2.set_title("Emotion-Conditioned\nSpatial Attention", fontsize=12)
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # Emotion probability distribution
    ax3 = plt.subplot(1, 3, 3)
    colors = plt.cm.Set3(np.linspace(0, 1, 6))
    bars = ax3.barh(emotion_labels, emotion_probs, color=colors)
    bars[pred_class].set_color("red")
    bars[pred_class].set_alpha(0.8)
    ax3.set_xlabel("Probability", fontsize=10)
    ax3.set_title("Emotion Distribution", fontsize=12)
    ax3.set_xlim(0, 1)
    
    plt.suptitle("Emotion-Aware Spatial Attention Pooling (EASAP)\nAttention conditioned on preliminary emotion predictions", 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Compare all attention pooling mechanisms on sample images."""
    paths = Paths()
    cfg = ExtendedTrainConfig()
    device = cfg.device
    
    # Find sample images
    img_root = Path(paths.img_root)
    sample_images = []
    for emotion_dir in ["joy", "anger", "sadness", "fear", "disgust", "neutral"]:
        emotion_path = img_root / emotion_dir
        if emotion_path.exists():
            imgs = list(emotion_path.glob("*.jpg"))[:1]
            sample_images.extend(imgs)
    
    if not sample_images:
        print("No sample images found!")
        return
    
    print(f"Found {len(sample_images)} sample images")
    
    # Test each model type
    model_types = ["multi_query", "hierarchical", "emotion_aware"]
    checkpoint_path = "dinov2_emotion_va.pt"
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Testing {model_type} model")
        print('='*60)
        
        model = load_model(checkpoint_path, model_type, device)
        
        # Visualize on first sample image
        img_path = sample_images[0]
        img_tensor, img_pil = load_and_preprocess_image(img_path, cfg.image_size)
        img_tensor = img_tensor.to(device)
        
        if model_type == "multi_query":
            visualize_multi_query_attention(model, img_tensor, img_pil, f"attention_{model_type}.png")
        elif model_type == "hierarchical":
            visualize_hierarchical_attention(model, img_tensor, img_pil, f"attention_{model_type}.png")
        elif model_type == "emotion_aware":
            visualize_emotion_aware_attention(model, img_tensor, img_pil, f"attention_{model_type}.png")
    
    print("\n" + "="*60)
    print("✓ All visualizations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
