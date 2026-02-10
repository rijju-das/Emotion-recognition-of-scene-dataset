import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from .config import get_output_dir, get_checkpoint_dir
from .attention_config import AttentionModelConfig
from .models.dinov2_multitask_extended import create_model
from .superpixel_extractor import SuperpixelPaletteExtractor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
    ])


def build_tensor_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_palette_kmeans(lab: np.ndarray, k: int, seed: int, sample_size: int):
    h, w, _ = lab.shape
    lab_flat = lab.reshape(-1, 3)
    n = lab_flat.shape[0]
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample = lab_flat[sample_idx]

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_

    labels = pairwise_distances_argmin(lab_flat, centers)
    labels = labels.reshape(h, w)
    return labels, centers


def make_baseline_rgb(rgb: np.ndarray, lab: np.ndarray, mode: str):
    if mode == "mean_rgb":
        return rgb.mean(axis=(0, 1))
    if mode == "gray":
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if mode == "lab_mean":
        lab_mean = lab.reshape(-1, 3).mean(axis=0)
        rgb_mean = lab2rgb(lab_mean.reshape(1, 1, 3)).reshape(3)
        return np.clip(rgb_mean, 0, 1)
    if mode == "lab_chroma":
        return None
    raise ValueError("baseline must be 'mean_rgb', 'gray', 'lab_mean', or 'lab_chroma'")


def evaluate_model(model, device, tensor: torch.Tensor, head: str, target_class: int | None, va_index: int | None):
    with torch.no_grad():
        logits, va = model(tensor.to(device))
    if head == "emotion":
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = float(logits[0, target_class].item())
        return score, target_class
    idx = 0 if va_index is None else int(va_index)
    score = float(va[0, idx].item())
    return score, idx


def save_palette_figure(
    output_dir: Path,
    palette_rgb: np.ndarray,
    drops: list[float],
    title: str,
    proportions: list[float],
):
    k = palette_rgb.shape[0]
    fig, ax = plt.subplots(figsize=(8, 2 + k * 0.3))
    ax.set_title(title)
    ax.set_axis_off()

    for i in range(k):
        y = k - 1 - i
        color = palette_rgb[i]
        ax.add_patch(plt.Rectangle((0, y), 1.0, 0.8, color=color))
        ax.text(
            1.05,
            y + 0.4,
            f"drop={drops[i]:+.4f}  area={proportions[i]*100:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, k)
    fig.tight_layout()
    path = output_dir / "palette_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_drop_bar(output_dir: Path, drops: list[float]):
    colors = ["#1f77b4" if d >= 0 else "#d62728" for d in drops]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(i) for i in range(len(drops))], drops, color=colors)
    ax.set_title("Logit/VA Drop by Palette Color")
    ax.set_xlabel("Palette Index")
    ax.set_ylabel("Drop (baseline - masked)")
    fig.tight_layout()
    path = output_dir / "color_drop_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_masks_grid(
    output_dir: Path,
    rgb: np.ndarray,
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    drops: list[float],
    baseline_rgb: np.ndarray,
):
    k = palette_rgb.shape[0]
    cols = min(5, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(k):
        ax = axes[i]
        mask = labels == i
        masked = rgb.copy()
        masked[mask] = baseline_rgb
        ax.imshow(masked)
        ax.set_title(f"i={i} drop={drops[i]:+.3f}")
        ax.axis("off")

    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    path = output_dir / "palette_masks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_color_causality_checks(model, device, img: Image.Image, image_size: int, head: str, target_class: int | None, va_index: int | None, output_dir: Path):
    """Comprehensive color dimension analysis: chroma, luminance, and hue."""
    pre_tf = build_eval_transform(image_size)
    tensor_tf = build_tensor_transform()

    img_proc = pre_tf(img)
    rgb = np.asarray(img_proc).astype(np.float32) / 255.0
    lab = rgb2lab(rgb)
    hsv = rgb2hsv(rgb)

    variants = {}
    variants["original"] = img_proc
    
    # === CHROMA (color saturation) tests ===
    # Remove all chroma (a=b=0), keep lightness - tests if "having color" matters
    lab_no_chroma = lab.copy()
    lab_no_chroma[..., 1] = 0  # a = 0
    lab_no_chroma[..., 2] = 0  # b = 0
    rgb_no_chroma = np.clip(lab2rgb(lab_no_chroma), 0, 1)
    variants["no_chroma_L_only"] = Image.fromarray((rgb_no_chroma * 255).astype(np.uint8))
    
    # Reduce chroma by 50% - tests sensitivity to saturation level
    lab_half_chroma = lab.copy()
    lab_half_chroma[..., 1] *= 0.5
    lab_half_chroma[..., 2] *= 0.5
    rgb_half_chroma = np.clip(lab2rgb(lab_half_chroma), 0, 1)
    variants["half_chroma"] = Image.fromarray((rgb_half_chroma * 255).astype(np.uint8))
    
    # Boost chroma by 150% - tests saturation boost impact
    lab_boost_chroma = lab.copy()
    lab_boost_chroma[..., 1] *= 1.5
    lab_boost_chroma[..., 2] *= 1.5
    rgb_boost_chroma = np.clip(lab2rgb(lab_boost_chroma), 0, 1)
    variants["boost_chroma"] = Image.fromarray((rgb_boost_chroma * 255).astype(np.uint8))
    
    # === LUMINANCE (brightness) tests ===
    # Flatten luminance to mean, keep chroma - tests if "being bright/dark" matters
    mean_L = lab[..., 0].mean()
    lab_flat_L = lab.copy()
    lab_flat_L[..., 0] = mean_L
    rgb_flat_L = np.clip(lab2rgb(lab_flat_L), 0, 1)
    variants["flat_luminance_ab_only"] = Image.fromarray((rgb_flat_L * 255).astype(np.uint8))
    
    # Reduce luminance contrast (compress L range)
    L_min, L_max = lab[..., 0].min(), lab[..., 0].max()
    lab_compress_L = lab.copy()
    if L_max > L_min:
        lab_compress_L[..., 0] = mean_L + (lab[..., 0] - mean_L) * 0.3
    rgb_compress_L = np.clip(lab2rgb(lab_compress_L), 0, 1)
    variants["compress_luminance"] = Image.fromarray((rgb_compress_L * 255).astype(np.uint8))
    
    # Invert luminance (dark->bright, bright->dark), keep chroma
    lab_invert_L = lab.copy()
    lab_invert_L[..., 0] = 100 - lab[..., 0]
    rgb_invert_L = np.clip(lab2rgb(lab_invert_L), 0, 1)
    variants["invert_luminance"] = Image.fromarray((rgb_invert_L * 255).astype(np.uint8))
    
    # === HUE (which color) tests ===
    # Rotate hue by 30 degrees - tests if "which color" matters
    variants["hue_shift_30deg"] = TF.adjust_hue(img_proc, 30/360)
    
    # Rotate hue by 60 degrees
    variants["hue_shift_60deg"] = TF.adjust_hue(img_proc, 60/360)
    
    # Rotate hue by 180 degrees (complementary colors)
    variants["hue_shift_180deg"] = TF.adjust_hue(img_proc, 0.5)
    
    # === Combined tests ===
    variants["grayscale"] = TF.to_grayscale(img_proc, num_output_channels=3)

    results = []
    for name, im in variants.items():
        tensor = tensor_tf(im).unsqueeze(0)
        score, target_or_idx = evaluate_model(model, device, tensor, head, target_class, va_index)
        results.append((name, score, target_or_idx))

    # Save text results
    with open(output_dir / "color_causality.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COLOR DIMENSION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        original_score = results[0][1]
        f.write(f"Original: score={original_score:.4f} target={results[0][2]}\n\n")
        
        f.write("--- CHROMA (having color) ---\n")
        for name, score, target_or_idx in results:
            if "chroma" in name.lower() or name == "no_chroma_L_only":
                drop = original_score - score
                f.write(f"{name:30s}: score={score:.4f} drop={drop:+.4f}\n")
        
        f.write("\n--- LUMINANCE (bright/dark) ---\n")
        for name, score, target_or_idx in results:
            if "luminance" in name.lower():
                drop = original_score - score
                f.write(f"{name:30s}: score={score:.4f} drop={drop:+.4f}\n")
        
        f.write("\n--- HUE (which color) ---\n")
        for name, score, target_or_idx in results:
            if "hue" in name.lower():
                drop = original_score - score
                f.write(f"{name:30s}: score={score:.4f} drop={drop:+.4f}\n")
        
        f.write("\n--- OTHER ---\n")
        for name, score, target_or_idx in results:
            if name not in ["original"] and "chroma" not in name.lower() and "luminance" not in name.lower() and "hue" not in name.lower():
                drop = original_score - score
                f.write(f"{name:30s}: score={score:.4f} drop={drop:+.4f}\n")
    
    # Save visual comparison (skip original image)
    variants_to_show = [(name, im, score) for (name, im), (_, score, _) in zip(variants.items(), results) if name != "original"]
    n_variants = len(variants_to_show)
    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    
    for idx, (name, im, score) in enumerate(variants_to_show):
        ax = axes[idx]
        ax.imshow(np.asarray(im))
        drop = original_score - score
        ax.set_title(f"{name}\nscore={score:.3f} drop={drop:+.3f}", fontsize=9)
        ax.axis("off")
    
    for j in range(n_variants, len(axes)):
        axes[j].axis("off")
    
    fig.tight_layout()
    fig.savefig(output_dir / "color_dimension_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_superpixels(rgb: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        convert2lab=False,
        enforce_connectivity=True,
        channel_axis=-1,
        slic_zero=True,
    )
    return segments


def compute_color_merged_superpixels(rgb: np.ndarray, n_segments: int, compactness: float, n_clusters: int) -> np.ndarray:
    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        convert2lab=False,
        enforce_connectivity=True,
        channel_axis=-1,
        slic_zero=True,
    )

    image_lab = rgb2lab(rgb)
    labels = np.unique(segments)
    feats = []
    for label in labels:
        mask = segments == label
        feats.append(image_lab[mask].mean(axis=0))
    feats = np.vstack(feats)

    k = min(n_clusters, len(labels))
    clustering = AgglomerativeClustering(n_clusters=k, linkage="average")
    cluster_ids = clustering.fit_predict(feats)

    max_label = int(labels.max())
    sp_to_cluster = np.zeros(max_label + 1, dtype=int)
    for label, cluster_id in zip(labels, cluster_ids):
        sp_to_cluster[int(label)] = int(cluster_id)

    merged_segments = sp_to_cluster[segments]
    return merged_segments


def run_hybrid_superpixel_ablation(
    output_dir: Path,
    model,
    device,
    rgb: np.ndarray,
    lab: np.ndarray,
    head: str,
    target_class: int | None,
    va_index: int | None,
    n_segments: int,
    compactness: float,
    topk: int,
    mode: str,
    n_clusters: int,
):
    # Use SuperpixelPaletteExtractor for sophisticated segmentation and posterization
    if mode == "color_merge":
        extractor = SuperpixelPaletteExtractor(
            n_segments=n_segments,
            n_clusters=n_clusters,
            compactness=compactness,
        )
        # Convert rgb to tensor format (C,H,W) for extractor
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # (3,H,W) in [0,1]
        lab_palette, hsv_palette, rgb_palette, impacts, segments, posterized_rgb = extractor.extract_palette(rgb_tensor)
    else:
        segments = compute_superpixels(rgb, n_segments=n_segments, compactness=compactness)
        # For simple SLIC mode, create posterized version for visualization
        posterized_rgb = np.zeros_like(rgb)
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            posterized_rgb[mask] = rgb[mask].mean(axis=0)
    
    unique_ids = np.unique(segments)
    total_pixels = segments.size

    tensor_tf = build_tensor_transform()
    base_tensor = tensor_tf(Image.fromarray((rgb * 255).astype(np.uint8))).unsqueeze(0)
    baseline_score, target_or_idx = evaluate_model(
        model,
        device,
        base_tensor,
        head,
        target_class,
        va_index,
    )

    drops = []
    areas = []
    for seg_id in unique_ids:
        mask = segments == seg_id
        areas.append(float(mask.sum() / total_pixels))

        lab_masked = lab.copy()
        lab_masked[mask, 1] = 0
        lab_masked[mask, 2] = 0
        rgb_masked = np.clip(lab2rgb(lab_masked), 0, 1)

        masked_tensor = tensor_tf(Image.fromarray((rgb_masked * 255).astype(np.uint8))).unsqueeze(0)
        masked_score, _ = evaluate_model(
            model,
            device,
            masked_tensor,
            head,
            target_or_idx if head == "emotion" else None,
            target_or_idx if head == "va" else None,
        )
        drops.append(baseline_score - masked_score)

    # Save CSV
    hybrid_dir = output_dir / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    # Save posterized segmentation visualization
    seg_path = hybrid_dir / "superpixel_segments.png"
    plt.imsave(seg_path, posterized_rgb)
    csv_path = hybrid_dir / "superpixel_drops.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id,area,drop\n")
        for seg_id, area, drop in zip(unique_ids, areas, drops):
            f.write(f"{seg_id},{area:.6f},{drop:.6f}\n")

    # Bar plot (sorted by drop magnitude)
    order = np.argsort(-np.abs(drops))
    show_ids = order[:topk]
    show_drops = [drops[i] for i in show_ids]
    show_labels = [str(int(unique_ids[i])) for i in show_ids]
    colors = ["#1f77b4" if d >= 0 else "#d62728" for d in show_drops]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(show_labels, show_drops, color=colors)
    ax.set_title(f"Hybrid Superpixel Color Ablation ({mode}, Top-K by |drop|)")
    ax.set_xlabel("Superpixel ID")
    ax.set_ylabel("Drop (baseline - masked)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(hybrid_dir / "superpixel_drop_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overlay grid for top-K (red overlay on posterized background)
    cols = min(5, topk)
    rows = (topk + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, seg_idx in enumerate(show_ids):
        ax = axes[idx]
        seg_id = unique_ids[seg_idx]
        mask = segments == seg_id
        # Blend posterized (90%) with original image (10%) for transparency, then add red overlay
        overlay = posterized_rgb * 0.7 + rgb * 0.3
        overlay[mask] = overlay[mask] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7
        ax.imshow(overlay)
        ax.set_title(f"id={int(seg_id)} drop={drops[seg_idx]:+.3f}\narea={areas[seg_idx]*100:.1f}%")
        ax.axis("off")
    for j in range(topk, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(hybrid_dir / "superpixel_overlays.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default="/home/rdas/color_transfer/Emotion6_new/joy/3.jpg",
        help="Path to input image",
    )
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/emotion6/attention/final_model.pt",
        help="Path to model checkpoint",
    )
    ap.add_argument("--k", type=int, default=5, help="Palette size")
    ap.add_argument("--baseline", choices=["mean_rgb", "gray", "lab_mean", "lab_chroma"], default="lab_chroma")
    ap.add_argument("--head", choices=["emotion", "va"], default="emotion")
    ap.add_argument("--target_class", type=int, default=None, help="Emotion class index")
    ap.add_argument("--va_index", type=int, default=None, help="VA index (0/1/2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_size", type=int, default=8000, help="Pixels sampled for kmeans")
    ap.add_argument("--superpixels", type=int, default=50)
    ap.add_argument("--superpixel_compactness", type=float, default=5.0)
    ap.add_argument("--hybrid_topk", type=int, default=20)
    ap.add_argument("--hybrid_mode", choices=["slic", "color_merge"], default="color_merge")
    ap.add_argument("--hybrid_clusters", type=int, default=20, help="Merged region count for color_merge")
    ap.add_argument("--no-hybrid", action="store_false", dest="hybrid", default=True, help="Disable hybrid superpixel ablation (enabled by default)")
    args = ap.parse_args()

    cfg = AttentionModelConfig()
    device = cfg.device

    checkpoint_path = args.checkpoint

    model = create_model(
        model_type=cfg.model_type,
        backbone_name=cfg.backbone_name,
        dropout=cfg.dropout,
        use_cls_plus_patchmean=True,
        num_queries=cfg.num_queries,
        num_heads=cfg.num_attention_heads,
        num_emotions=cfg.num_emotions,
        va_dims=cfg.va_dims,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_path = Path(args.image)
    img = Image.open(image_path).convert("RGB")

    pre_tf = build_eval_transform(cfg.image_size)
    img_proc = pre_tf(img)

    rgb = np.asarray(img_proc).astype(np.float32) / 255.0
    lab = rgb2lab(rgb)

    labels, centers_lab = compute_palette_kmeans(lab, k=args.k, seed=args.seed, sample_size=args.sample_size)
    palette_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb, 0, 1)

    base_rgb = make_baseline_rgb(rgb, lab, args.baseline)

    tensor_tf = build_tensor_transform()
    base_tensor = tensor_tf(Image.fromarray((rgb * 255).astype(np.uint8))).unsqueeze(0)

    baseline_score, target_or_idx = evaluate_model(
        model,
        device,
        base_tensor,
        args.head,
        args.target_class,
        args.va_index,
    )

    drops = []
    proportions = []
    total_pixels = labels.size
    for i in range(args.k):
        rgb_masked = rgb.copy()
        mask = labels == i
        proportions.append(float(mask.sum() / total_pixels))
        if args.baseline == "lab_chroma":
            lab_masked = lab.copy()
            lab_masked[mask, 1] = 0
            lab_masked[mask, 2] = 0
            rgb_masked = np.clip(lab2rgb(lab_masked), 0, 1)
        else:
            rgb_masked[mask] = base_rgb

        masked_tensor = tensor_tf(Image.fromarray((rgb_masked * 255).astype(np.uint8))).unsqueeze(0)
        masked_score, _ = evaluate_model(
            model,
            device,
            masked_tensor,
            args.head,
            target_or_idx if args.head == "emotion" else None,
            target_or_idx if args.head == "va" else None,
        )
        drops.append(baseline_score - masked_score)

    out_dir = get_output_dir("color_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hybrid:
        run_hybrid_superpixel_ablation(
            output_dir=out_dir,
            model=model,
            device=device,
            rgb=rgb,
            lab=lab,
            head=args.head,
            target_class=args.target_class,
            va_index=args.va_index,
            n_segments=args.superpixels,
            compactness=args.superpixel_compactness,
            topk=args.hybrid_topk,
            mode=args.hybrid_mode,
            n_clusters=args.hybrid_clusters,
        )

    title = f"Palette Importance (head={args.head}, baseline={args.baseline})"
    save_palette_figure(out_dir, palette_rgb, drops, title, proportions)
    save_drop_bar(out_dir, drops)
    save_masks_grid(out_dir, rgb, labels, palette_rgb, drops, base_rgb if base_rgb is not None else np.array([0.5, 0.5, 0.5], dtype=np.float32))
    run_color_causality_checks(model, device, img, cfg.image_size, args.head, args.target_class, args.va_index, out_dir)

    print("Saved:")
    print(f"  {out_dir / 'palette_importance.png'}")
    print(f"  {out_dir / 'color_drop_bar.png'}")
    print(f"  {out_dir / 'palette_masks.png'}")
    print(f"  {out_dir / 'color_causality.txt'}")
    print(f"  {out_dir / 'color_dimension_analysis.png'}")
    if args.hybrid:
        print(f"  {out_dir / 'hybrid' / 'superpixel_segments.png'}")
        print(f"  {out_dir / 'hybrid' / 'superpixel_drops.csv'}")
        print(f"  {out_dir / 'hybrid' / 'superpixel_overlays.png'}")


if __name__ == "__main__":
    main()
