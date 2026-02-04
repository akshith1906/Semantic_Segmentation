"""
a script to validate image and mask loading methods
"""

import os
import random
from typing import Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def visualize_mask_classes(
    dataset_dir,
    train_images_folder="train/images",
    train_labels_folder="train/labels",
    sample_index=None,
):
    images_path = os.path.join(dataset_dir, train_images_folder)
    labels_path = os.path.join(dataset_dir, train_labels_folder)

    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))

    if len(image_files) == 0:
        raise ValueError("bi image files found in the training directory.")
    if sample_index is None:
        sample_index = random.randint(0, len(image_files) - 1)

    image_file = image_files[sample_index]
    label_file = label_files[sample_index]

    image = cv2.imread(os.path.join(images_path, image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(labels_path, label_file))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask[..., 0]

    class_names = get_class_names()
    num_classes = len(class_names)
    total_plots = num_classes + 1
    cols = 7
    rows = (total_plots // cols) + (total_plots % cols > 0)

    _, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for class_id in range(num_classes):
        ax = axes[class_id + 1]
        region_mask = mask == class_id

        region_image = np.zeros_like(mask, dtype=np.uint8)
        region_image[region_mask] = mask[region_mask]

        ax.imshow(region_image, cmap="gray")
        class_name = class_names.get(class_id, f"Class {class_id}")
        ax.set_title(f"{class_id}: {class_name}")
        ax.axis("off")

    for ax in axes[total_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_directory = os.path.join("..", "data", "dataset_224")
    visualize_mask_classes(dataset_directory)
