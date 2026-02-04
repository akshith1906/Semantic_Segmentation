import os
import argparse

import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

from src.dataset import Mode, get_class_names, get_dataloader
from src.model import FCN, FCNVariant, get_project_name


def visualize_predictions(
    model: FCN, dataloader: DataLoader, device: torch.device, res_dir: str
):
    model.eval()
    model.to(device)

    # get a batch from the dataloader
    images, ground_truth_masks = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    predictions = torch.argmax(outputs, dim=1)

    images = images.cpu()
    ground_truth_masks = ground_truth_masks.cpu()
    predictions = predictions.cpu()

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0)

        true_mask = ground_truth_masks[i]
        pred_mask = predictions[i]

        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(true_mask, cmap="viridis")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap="viridis")
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(res_dir, f"pred_{i}.png")
        plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("road segmentation")
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join("..", "data", "dataset_224")
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--variant", type=str, default="fcn8s")

    args = parser.parse_args()

    classes = get_class_names()
    variant = FCNVariant(args.variant)
    model = FCN(variant=variant, num_classes=len(classes))

    proj_name = get_project_name(variant, args.freeze_backbone)
    ckpt_path = os.path.join("ckpts", f"{proj_name}.pth")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    test_dataloader = get_dataloader(
        args.data_dir, Mode.TEST, batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_dir = os.path.join("ckpts", proj_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    visualize_predictions(model, test_dataloader, device, res_dir)
