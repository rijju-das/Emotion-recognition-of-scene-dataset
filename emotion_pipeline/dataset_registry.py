from dataclasses import dataclass
from typing import Type

from .data.emotion6 import Emotion6Dataset, EMOTIONS
from .data.dvisa import DVisaDataset, DVISA_EMOTIONS
from .data.emoset_new import EmoSetNewDataset, EMOSET_NEW_EMOTIONS


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    dataset_cls: Type
    class_names: list[str]


_DATASETS = {
    "emotion6": DatasetInfo(
        name="emotion6",
        dataset_cls=Emotion6Dataset,
        class_names=EMOTIONS,
    ),
    "emotion_ori": DatasetInfo(
        name="emotion_ori",
        dataset_cls=Emotion6Dataset,
        class_names=EMOTIONS,
    ),
    "dvisa": DatasetInfo(
        name="dvisa",
        dataset_cls=DVisaDataset,
        class_names=DVISA_EMOTIONS,
    ),
    "emoset_new": DatasetInfo(
        name="emoset_new",
        dataset_cls=EmoSetNewDataset,
        class_names=EMOSET_NEW_EMOTIONS,
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {sorted(_DATASETS)}")
    return _DATASETS[name]


def infer_label_field(header: list[str], dataset_name: str) -> str:
    if dataset_name == "dvisa":
        if "emotion" in header:
            return "emotion"
        if "final_emo" in header:
            return "final_emo"
    if "label" in header:
        return "label"
    if "label_str" in header:
        return "label_str"
    return "label"
