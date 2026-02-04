import os
from enum import Enum
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2


def get_class_names() -> Dict[int, str]:
    class_names = {
        0: "unlabeled",
        1: "building",
        2: "fence",
        3: "other",
        4: "pedestrian",
        5: "pole",
        6: "roadline",
        7: "road",
        8: "sidewalk",
        9: "vegetation",
        10: "car",
        11: "wall",
        12: "traffic sign",
    }
    return class_names


class Mode(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class RoadDataset(Dataset):
    def __init__(self, mode: Mode, root_dir: str):
        self.mode = mode

        self.images_dir = os.path.join(root_dir, mode.value, "images")
        self.image_files = sorted(os.listdir(self.images_dir))

        self.labels_dir = os.path.join(root_dir, mode.value, "labels")
        self.label_files = sorted(os.listdir(self.labels_dir))

        assert len(self.image_files) == len(
            self.label_files
        ), "number of images does not match the number of labels"
        assert (
            len(self.image_files) > 0
        ), "no image files found in the training directory"

        print(f"{mode.value} - loaded {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.image_files[index])
        label_path = os.path.join(self.labels_dir, self.label_files[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image)
        assert type(image_tensor) is torch.Tensor

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = label[..., 0]
        label_tensor = torch.from_numpy(label).long()

        return image_tensor, label_tensor


def get_dataloader(data_dir: str, mode: Mode, **kwargs) -> DataLoader[RoadDataset]:
    dataset = RoadDataset(mode, data_dir)
    return DataLoader(dataset, **kwargs)
