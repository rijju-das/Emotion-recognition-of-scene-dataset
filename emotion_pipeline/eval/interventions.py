from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms.functional as TF

class BaseIntervention(ABC):
    @abstractmethod
    def apply(self, img: Image.Image) -> Image.Image:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

class GrayscaleIntervention(BaseIntervention):
    def apply(self, img: Image.Image) -> Image.Image:
        return TF.to_grayscale(img, num_output_channels=3)

    def name(self) -> str:
        return "grayscale"

class SaturationIntervention(BaseIntervention):
    def __init__(self, scale: float):
        self.scale = scale

    def apply(self, img: Image.Image) -> Image.Image:
        return TF.adjust_saturation(img, self.scale)

    def name(self) -> str:
        return f"sat_{self.scale:.2f}"

class HueShiftIntervention(BaseIntervention):
    def __init__(self, delta: float):
        self.delta = delta  # [-0.5, 0.5], where 0.5 = 180 degrees

    def apply(self, img: Image.Image) -> Image.Image:
        return TF.adjust_hue(img, self.delta)

    def name(self) -> str:
        return f"hue_{self.delta:+.2f}"
