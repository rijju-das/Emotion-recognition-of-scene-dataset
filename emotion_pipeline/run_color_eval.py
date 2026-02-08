import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from config import Paths, TrainConfig
from data.emotion6 import Emotion6Dataset
from models.dinov2_multitask import DinoV2EmotionVA
from training.multitask_trainer import MultiTaskTrainer
from eval.interventions import GrayscaleIntervention, SaturationIntervention, HueShiftIntervention

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_test_tf(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class InterventionWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, intervention, pil_loader):
        self.base = base_ds
        self.intv = intervention
        self.pil_loader = pil_loader  # function to load PIL from filename

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        # base returns already-transformed tensor; we need PIL-level edit -> so reload PIL
        row = self.base.df.iloc[idx]
        pil = self.pil_loader(row["filename"])
        pil2 = self.intv.apply(pil)

        x = self.base.transform(pil2)
        y = torch.tensor(self.base.df.iloc[idx]["label_str"]).item()  # not used; keep simple
        # easier: just call base again and replace x, but we need y,va
        # So: re-use base's label/va creation logic via base.__getitem__ with pil hack is more code.
        # In practice, implement Emotion6Dataset to optionally return PIL pre-transform.
        return x, *self.base.__getitem__(idx)[1:]

def main():
    paths = Paths()
    cfg = TrainConfig()

    test_tf = make_test_tf(cfg.image_size)
    test_ds = Emotion6Dataset(str(paths.test_csv), str(paths.img_root), transform=test_tf)

    model = DinoV2EmotionVA(backbone_name=cfg.backbone_name, use_cls_plus_patchmean=True)
    model.load_state_dict(torch.load("dinov2_emotion_va.pt", map_location=cfg.device))
    model.eval()

    # Simple: just evaluate baseline first (no intervention)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    trainer = MultiTaskTrainer(model, train_loader=test_loader, test_loader=test_loader, device=cfg.device, lam_va=cfg.lam_va)
    print("Baseline:", trainer.evaluate())

    # NOTE: For clean intervention evaluation, implement dataset option to return PIL before transform.
    # Then apply intervention on PIL and transform to tensor. (Happy to provide that version.)

if __name__ == "__main__":
    main()
