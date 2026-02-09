import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import csv
import matplotlib.pyplot as plt

from .config import get_paths, get_checkpoint_dir, get_output_dir, DATASET_NAME
from .attention_config import AttentionModelConfig
from .dataset_registry import get_dataset_info
from .models.dinov2_multitask_extended import create_model
from .training.multitask_trainer import MultiTaskTrainer
from .eval.interventions import GrayscaleIntervention, SaturationIntervention, HueShiftIntervention

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_test_tf(image_size: int, intervention=None):
    tf_list = []
    if intervention is not None:
        tf_list.append(transforms.Lambda(lambda img: intervention.apply(img)))
    tf_list += [
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(tf_list)

def main():
    paths = get_paths()
    cfg = AttentionModelConfig()
    ds_info = get_dataset_info(DATASET_NAME)
    output_dir = get_output_dir("color_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model.load_state_dict(torch.load(get_checkpoint_dir("attention") / "final_model.pt", map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    interventions = [
        ("baseline", None),
        ("grayscale", GrayscaleIntervention()),
        ("hue+0.08", HueShiftIntervention(0.08)),
        ("hue-0.08", HueShiftIntervention(-0.08)),
        ("sat0.5", SaturationIntervention(0.5)),
        ("sat1.5", SaturationIntervention(1.5)),
    ]

    results = []
    for name, intv in interventions:
        test_tf = make_test_tf(cfg.image_size, intervention=intv)
        test_ds = ds_info.dataset_cls(str(paths.test_csv), str(paths.img_root), transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        trainer = MultiTaskTrainer(
            model,
            train_loader=test_loader,
            test_loader=test_loader,
            device=cfg.device,
            lam_va=cfg.lam_va,
        )
        metrics = trainer.evaluate()
        results.append({
            "name": name,
            "loss": metrics["loss"],
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "rmse_va": metrics["rmse_va"],
        })
        print(f"{name:10s}: loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} rmse_va={metrics['rmse_va']:.4f}")

    if results:
        baseline = next((r for r in results if r["name"] == "baseline"), results[0])
        csv_path = output_dir / "color_eval_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "loss",
                    "acc",
                    "f1",
                    "rmse_va",
                    "acc_delta",
                    "f1_delta",
                    "rmse_va_delta",
                ],
            )
            writer.writeheader()
            for r in results:
                row = {
                    "name": r["name"],
                    "loss": r["loss"],
                    "acc": r["acc"],
                    "f1": r["f1"],
                    "rmse_va": r["rmse_va"],
                    "acc_delta": r["acc"] - baseline["acc"],
                    "f1_delta": r["f1"] - baseline["f1"],
                    "rmse_va_delta": r["rmse_va"] - baseline["rmse_va"],
                }
                writer.writerow(row)

        names = [r["name"] for r in results]
        accs = [r["acc"] for r in results]
        f1s = [r["f1"] for r in results]
        rmses = [r["rmse_va"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, accs, color="#4c78a8")
        ax.set_title("Accuracy by Color Intervention")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_by_intervention.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, f1s, color="#f58518")
        ax.set_title("Macro F1 by Color Intervention")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / "f1_by_intervention.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, rmses, color="#54a24b")
        ax.set_title("RMSE VA by Color Intervention")
        ax.set_ylabel("RMSE VA")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / "rmse_va_by_intervention.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {csv_path}")
        print(f"Saved: {output_dir / 'accuracy_by_intervention.png'}")
        print(f"Saved: {output_dir / 'f1_by_intervention.png'}")
        print(f"Saved: {output_dir / 'rmse_va_by_intervention.png'}")

if __name__ == "__main__":
    main()
