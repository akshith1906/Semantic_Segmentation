import os
import random


def split_train_val(dataset_dir, val_ratio=0.15, seed=42):
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")

    val_images_dir = os.path.join(dataset_dir, "valid", "images")
    val_labels_dir = os.path.join(dataset_dir, "valid", "labels")

    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = sorted(os.listdir(train_images_dir))
    label_files = sorted(os.listdir(train_labels_dir))

    if len(image_files) != len(label_files):
        raise ValueError("The number of images does not match the number of labels.")

    total_files = len(image_files)
    num_val = int(total_files * val_ratio)

    random.seed(seed)
    indices = list(range(total_files))
    random.shuffle(indices)
    val_indices = set(indices[:num_val])

    for idx, (img_file, label_file) in enumerate(zip(image_files, label_files)):
        if idx in val_indices:
            src_img = os.path.join(train_images_dir, img_file)
            src_label = os.path.join(train_labels_dir, label_file)
            dst_img = os.path.join(val_images_dir, img_file)
            dst_label = os.path.join(val_labels_dir, label_file)

            os.rename(src_img, dst_img)
            os.rename(src_label, dst_label)
            print(f"moved: {img_file} and {label_file} to validation set")


if __name__ == "__main__":
    dataset_directory = "../data/dataset_224"
    split_train_val(dataset_directory)
