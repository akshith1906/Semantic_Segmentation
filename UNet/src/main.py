import os
import argparse

import torch
import wandb

from src.visualise import visualize_predictions
from src.unet.factory import fetch_unet, Variant, get_project_name
from src.loops import train_model, test_model
from src.dataset import get_class_names, get_dataloader, Mode


def main(
    data_dir: str,
    batch_size: int,
    variant: str,
    num_epochs: int,
    lr: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(data_dir, Mode.TRAIN, batch_size=batch_size)
    valid_dataloader = get_dataloader(data_dir, Mode.VALID, batch_size=batch_size)

    classes = get_class_names()

    unet = fetch_unet(Variant(variant))
    model = unet(in_channels=3, out_channels=len(classes)).to(device)
    proj_name = get_project_name(variant)

    ckpts_dir = "ckpts"
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)
    ckpt_path = os.path.join(ckpts_dir, f"{proj_name}.pth")

    run = wandb.init(
        project="cv-a4-unet",
        name=proj_name,
        config={
            "learning rate": lr,
        },
    )
    train_model(
        model,
        train_dataloader,
        valid_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        ckpt_path=ckpt_path,
    )

    test_dataloader = get_dataloader(data_dir, Mode.TEST, batch_size=batch_size)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_model(model, test_dataloader, device)

    run.finish()

    res_dir = os.path.join(ckpts_dir, proj_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    visualize_predictions(model, test_dataloader, device, res_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("road segmentation")
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join("..", "data", "dataset_256")
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--variant", type=str, default="vanilla")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    main(
        args.data_dir,
        args.batch_size,
        args.variant,
        args.num_epochs,
        args.lr,
    )
