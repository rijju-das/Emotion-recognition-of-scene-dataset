import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from .config import Paths, TrainConfig
from emotion_pipeline.models.dinov2_multitask import DinoV2EmotionVA
from emotion_pipeline.xai.ig_explainer import IntegratedGradientsExplainer
from emotion_pipeline.xai.gradient_input_explainer import GradientInputExplainer
from emotion_pipeline.xai.gradcam_explainer import GradCAMExplainer
from emotion_pipeline.xai.shap_explainer import SHAPSuperpixelExplainer

def load_model(checkpoint_path: str, device: str):
    model = DinoV2EmotionVA(backbone_name="dinov2_vitb14", use_cls_plus_patchmean=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_and_preprocess_image(img_path: str, image_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    return x

def visualize_attribution(x: torch.Tensor, attribution: torch.Tensor, title: str):
    """Visualize input image and attribution heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Denormalize image
    x_vis = x[0].permute(1, 2, 0).detach().cpu()
    x_vis = (x_vis * torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3) + 
             torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3))
    x_vis = x_vis.clamp(0, 1).numpy()

    axes[0].imshow(x_vis)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Attribution heatmap
    if attribution is None:
        axes[1].text(0.5, 0.5, "No attribution", ha="center", va="center")
    else:
        if attribution.dim() == 4:
            attr_vis = attribution.mean(dim=1)[0].detach().cpu().numpy()
        else:
            attr_vis = attribution.detach().cpu().numpy()
        axes[1].imshow(attr_vis, cmap="RdBu_r")
    axes[1].set_title(title)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"xai_{title}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: xai_{title}.png")

def main():
    paths = Paths()
    cfg = TrainConfig()
    device = cfg.device
    model = load_model("dinov2_emotion_va.pt", device=device)
    x = load_and_preprocess_image(paths.img_root / "joy/1.jpg")
    x = x.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits, va = model(x)
        pred_class = logits.argmax(dim=1)
        print(f"Predicted class: {pred_class.item()}, Logits: {logits}")
    
    print("\n=== Integrated Gradients ===")
    ig_explainer = IntegratedGradientsExplainer(model, head="emotion")
    ig_attr = ig_explainer.explain(x, target=pred_class)
    visualize_attribution(x, ig_attr, "IntegratedGradients")
    
    print("\n=== Gradient × Input ===")
    gi_explainer = GradientInputExplainer(model, head="emotion")
    gi_attr = gi_explainer.explain(x, target=pred_class)
    visualize_attribution(x, gi_attr, "Gradient_Input")
    
    print("\n=== Grad-CAM ===")
    gradcam_explainer = GradCAMExplainer(model, head="emotion")
    cam = gradcam_explainer.explain(x, target=pred_class)
    visualize_attribution(x, cam, "GradCAM")
    
    print("\n=== SHAP Superpixels ===")
    shap_explainer = SHAPSuperpixelExplainer(model, head="emotion", n_superpixels=50)
    shap_result = shap_explainer.explain(x, target=pred_class, n_samples=50)
    print(f"SHAP result keys: {shap_result.keys()}")
    print(f"Superpixel attributions shape: {shap_result['attributions'].shape}")

if __name__ == "__main__":
    main()