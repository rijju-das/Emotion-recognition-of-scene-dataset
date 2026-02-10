import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import torchvision.transforms.functional as TF

from .attention_config import AttentionModelConfig
from .config import get_paths, get_output_dir
from .dataset_registry import get_dataset_info
from .models.dinov2_multitask_extended import create_model
from .superpixel_extractor import SuperpixelPaletteExtractor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PrototypePalette:
    centers_lab: np.ndarray  # (K, 3)
    weights: np.ndarray      # (K,)
    cov_diag: np.ndarray     # (K, 3)


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


def load_model(cfg: AttentionModelConfig, checkpoint_path: str | Path):
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()
    return model


def eval_emotion_logit(model, device, tensor: torch.Tensor, target_class: int) -> float:
    with torch.no_grad():
        logits, _ = model(tensor.to(device))
    return float(logits[0, target_class].item())


def compute_region_drops(
    model,
    device,
    rgb01: np.ndarray,
    lab: np.ndarray,
    segments: np.ndarray,
    target_class: int,
    tensor_tf,
) -> tuple[list[float], list[float]]:
    base_tensor = tensor_tf(Image.fromarray((rgb01 * 255).astype(np.uint8))).unsqueeze(0)
    baseline_score = eval_emotion_logit(model, device, base_tensor, target_class)

    unique_ids = np.unique(segments)
    total_pixels = segments.size
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
        masked_score = eval_emotion_logit(model, device, masked_tensor, target_class)
        drops.append(baseline_score - masked_score)

    return drops, areas


def collect_lab_samples(
    lab: np.ndarray,
    segments: np.ndarray,
    top_ids: list[int],
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.isin(segments, top_ids)
    samples = lab[mask].reshape(-1, 3)
    if samples.shape[0] > max_pixels:
        idx = rng.choice(samples.shape[0], size=max_pixels, replace=False)
        samples = samples[idx]
    return samples


def build_prototypes(
    model,
    cfg: AttentionModelConfig,
    dataset_name: str,
    split: str,
    output_path: Path,
    n_segments: int,
    n_clusters: int,
    compactness: float,
    topk_regions: int,
    palette_k: int,
    max_images_per_class: int,
    max_pixels_per_class: int,
    seed: int,
):
    info = get_dataset_info(dataset_name)
    paths = get_paths()
    csv_path = paths.train_csv if split == "train" else paths.test_csv

    eval_tf = build_eval_transform(cfg.image_size)
    tensor_tf = build_tensor_transform()

    dataset = info.dataset_cls(
        csv_path=str(csv_path),
        img_root=str(paths.img_root),
        transform=None,
        use_scaled_va=True,
    )

    extractor = SuperpixelPaletteExtractor(
        n_segments=n_segments,
        n_clusters=n_clusters,
        compactness=compactness,
    )

    rng = np.random.default_rng(seed)
    class_names = info.class_names
    class_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    class_counts: dict[int, int] = {i: 0 for i in range(len(class_names))}

    for idx in range(len(dataset)):
        img, y, _ = dataset[idx]
        y = int(y.item())
        if class_counts[y] >= max_images_per_class:
            continue

        img_proc = eval_tf(img)
        rgb01 = np.asarray(img_proc).astype(np.float32) / 255.0
        lab = rgb2lab(rgb01)
        rgb_tensor = torch.from_numpy(rgb01.transpose(2, 0, 1)).float()
        _, _, _, _, segments, _ = extractor.extract_palette(rgb_tensor)

        drops, _ = compute_region_drops(
            model,
            cfg.device,
            rgb01,
            lab,
            segments,
            target_class=y,
            tensor_tf=tensor_tf,
        )

        unique_ids = np.unique(segments)
        order = np.argsort(-np.asarray(drops))
        top_ids = [int(unique_ids[i]) for i in order[:topk_regions]]

        samples = collect_lab_samples(
            lab,
            segments,
            top_ids=top_ids,
            max_pixels=max_pixels_per_class,
            rng=rng,
        )
        if samples.size > 0:
            class_samples[y].append(samples)
            class_counts[y] += 1

    palettes = {}
    for class_id, name in enumerate(class_names):
        if len(class_samples[class_id]) == 0:
            continue
        samples = np.vstack(class_samples[class_id])
        if samples.shape[0] > max_pixels_per_class:
            idx = rng.choice(samples.shape[0], size=max_pixels_per_class, replace=False)
            samples = samples[idx]

        k = min(palette_k, samples.shape[0])
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(samples)
        centers = kmeans.cluster_centers_

        weights = np.zeros(k, dtype=np.float32)
        cov_diag = np.zeros((k, 3), dtype=np.float32)
        for i in range(k):
            cluster_samples = samples[labels == i]
            if cluster_samples.shape[0] == 0:
                continue
            weights[i] = float(cluster_samples.shape[0]) / float(samples.shape[0])
            cov = np.var(cluster_samples, axis=0)
            cov_diag[i] = cov

        palettes[name] = PrototypePalette(
            centers_lab=centers.astype(np.float32),
            weights=weights.astype(np.float32),
            cov_diag=cov_diag.astype(np.float32),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    npz_data = {
        "class_names": np.array(class_names, dtype=object),
        "dataset_name": dataset_name,
    }
    for name, palette in palettes.items():
        npz_data[f"{name}_centers"] = palette.centers_lab
        npz_data[f"{name}_weights"] = palette.weights
        npz_data[f"{name}_cov_diag"] = palette.cov_diag

    np.savez(output_path, **npz_data)
    meta = {
        "dataset_name": dataset_name,
        "split": split,
        "n_segments": n_segments,
        "n_clusters": n_clusters,
        "compactness": compactness,
        "topk_regions": topk_regions,
        "palette_k": palette_k,
        "max_images_per_class": max_images_per_class,
        "max_pixels_per_class": max_pixels_per_class,
        "seed": seed,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_prototypes(path: Path) -> dict[str, PrototypePalette]:
    data = np.load(path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"]]
    palettes = {}
    for name in class_names:
        centers_key = f"{name}_centers"
        weights_key = f"{name}_weights"
        cov_key = f"{name}_cov_diag"
        if centers_key not in data:
            continue
        palettes[name] = PrototypePalette(
            centers_lab=data[centers_key],
            weights=data[weights_key],
            cov_diag=data[cov_key],
        )
    return palettes


def transfer_image(
    rgb01: np.ndarray,
    segments: np.ndarray,
    palette: PrototypePalette,
    ab_strength: float,
    chroma_scale: float,
) -> np.ndarray:
    lab = rgb2lab(rgb01)
    out = lab.copy()

    centers_ab = palette.centers_lab[:, 1:3]
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        region_lab = lab[mask]
        if region_lab.shape[0] == 0:
            continue
        region_ab = region_lab[:, 1:3].mean(axis=0)
        dists = np.linalg.norm(centers_ab - region_ab, axis=1)
        target_idx = int(np.argmin(dists))
        target_ab = centers_ab[target_idx]

        new_ab = (1.0 - ab_strength) * region_lab[:, 1:3] + ab_strength * target_ab

        if chroma_scale != 1.0:
            src_chroma = np.sqrt(region_lab[:, 1] ** 2 + region_lab[:, 2] ** 2) + 1e-6
            tgt_chroma = np.sqrt(target_ab[0] ** 2 + target_ab[1] ** 2) + 1e-6
            scale = (tgt_chroma / src_chroma) * chroma_scale
            new_ab = new_ab * scale[:, None]

        out_region = out[mask]
        out_region[:, 1:3] = new_ab
        out[mask] = out_region

    rgb_out = np.clip(lab2rgb(out), 0, 1)
    return rgb_out


def run_transfer(
    cfg: AttentionModelConfig,
    model,
    image_path: Path,
    prototype_path: Path,
    target_emotion: str,
    output_dir: Path,
    n_segments: int,
    n_clusters: int,
    compactness: float,
    ab_strength: float,
    chroma_scale: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    palettes = load_prototypes(prototype_path)
    if target_emotion not in palettes:
        raise ValueError(f"Target emotion '{target_emotion}' not found in prototypes.")

    img = Image.open(image_path).convert("RGB")
    pre_tf = build_eval_transform(cfg.image_size)
    img_proc = pre_tf(img)
    rgb01 = np.asarray(img_proc).astype(np.float32) / 255.0

    extractor = SuperpixelPaletteExtractor(
        n_segments=n_segments,
        n_clusters=n_clusters,
        compactness=compactness,
    )
    segments = extractor.extract_segments(rgb01)

    rgb_out = transfer_image(
        rgb01,
        segments,
        palette=palettes[target_emotion],
        ab_strength=ab_strength,
        chroma_scale=chroma_scale,
    )

    Image.fromarray((rgb01 * 255).astype(np.uint8)).save(output_dir / "original.png")
    Image.fromarray((rgb_out * 255).astype(np.uint8)).save(output_dir / "transferred.png")

    tensor_tf = build_tensor_transform()
    base_tensor = tensor_tf(Image.fromarray((rgb01 * 255).astype(np.uint8))).unsqueeze(0)
    out_tensor = tensor_tf(Image.fromarray((rgb_out * 255).astype(np.uint8))).unsqueeze(0)

    with torch.no_grad():
        base_logits, _ = model(base_tensor.to(cfg.device))
        out_logits, _ = model(out_tensor.to(cfg.device))

    base_pred = int(base_logits.argmax(dim=1).item())
    out_pred = int(out_logits.argmax(dim=1).item())

    report = {
        "target_emotion": target_emotion,
        "base_pred": base_pred,
        "out_pred": out_pred,
        "base_target_logit": float(base_logits[0, base_pred].item()),
        "out_target_logit": float(out_logits[0, out_pred].item()),
    }
    with open(output_dir / "transfer_report.json", "w") as f:
        json.dump(report, f, indent=2)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--dataset", default="emotion6")
    build.add_argument("--split", default="train", choices=["train", "test"])
    build.add_argument("--output", default=None)
    build.add_argument("--n_segments", type=int, default=80)
    build.add_argument("--n_clusters", type=int, default=12)
    build.add_argument("--compactness", type=float, default=8.0)
    build.add_argument("--topk_regions", type=int, default=5)
    build.add_argument("--palette_k", type=int, default=8)
    build.add_argument("--max_images_per_class", type=int, default=200)
    build.add_argument("--max_pixels_per_class", type=int, default=20000)
    build.add_argument("--seed", type=int, default=42)

    transfer = sub.add_parser("transfer")
    transfer.add_argument("--checkpoint", required=True)
    transfer.add_argument("--prototype", required=True)
    transfer.add_argument("--image", required=True)
    transfer.add_argument("--target_emotion", required=True)
    transfer.add_argument("--output_dir", default=None)
    transfer.add_argument("--n_segments", type=int, default=80)
    transfer.add_argument("--n_clusters", type=int, default=12)
    transfer.add_argument("--compactness", type=float, default=8.0)
    transfer.add_argument("--ab_strength", type=float, default=0.6)
    transfer.add_argument("--chroma_scale", type=float, default=1.0)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = AttentionModelConfig()
    if args.command == "build":
        model = load_model(cfg, args.checkpoint)
        output_path = Path(args.output) if args.output else get_output_dir("palette_prototypes") / "prototypes.npz"
        build_prototypes(
            model=model,
            cfg=cfg,
            dataset_name=args.dataset,
            split=args.split,
            output_path=output_path,
            n_segments=args.n_segments,
            n_clusters=args.n_clusters,
            compactness=args.compactness,
            topk_regions=args.topk_regions,
            palette_k=args.palette_k,
            max_images_per_class=args.max_images_per_class,
            max_pixels_per_class=args.max_pixels_per_class,
            seed=args.seed,
        )
        print(f"Saved prototypes to {output_path}")
        print(f"Metadata: {output_path.with_suffix('.json')}")
    else:
        model = load_model(cfg, args.checkpoint)
        output_dir = Path(args.output_dir) if args.output_dir else get_output_dir("palette_transfer")
        run_transfer(
            cfg=cfg,
            model=model,
            image_path=Path(args.image),
            prototype_path=Path(args.prototype),
            target_emotion=args.target_emotion,
            output_dir=output_dir,
            n_segments=args.n_segments,
            n_clusters=args.n_clusters,
            compactness=args.compactness,
            ab_strength=args.ab_strength,
            chroma_scale=args.chroma_scale,
        )
        print(f"Saved transfer outputs to {output_dir}")


if __name__ == "__main__":
    main()
