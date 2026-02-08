from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @abstractmethod
    def class_names(self) -> list[str]:
        ...