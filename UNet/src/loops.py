import time

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU

from src.unet.unet import UNet
from src.dataset import get_class_names


def _get_loss_criterion() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()


def _compute_miou(model, dataloader, device, num_classes):
    model.eval()

    miou = MeanIoU(num_classes=num_classes)
    miou = miou.to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="computing mIoU"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            miou.update(preds, labels)

    return miou.compute().item()


def _test_epoch(
    model: UNet,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()  # type: ignore

    num_items = len(dataloader.dataset)  # type: ignore
    return test_loss / num_items


def _train_epoch(
    model: UNet,
    optimizer,
    dataloader: DataLoader,
    criterion: torch.nn.Module,  # TODO: fix type
    device: torch.device,
) -> float:
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(dataloader, desc="training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()

        train_loss += loss.item()  # type: ignore

    num_items = len(dataloader.dataset)  # type: ignore
    return train_loss / num_items


def train_model(
    model: UNet,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    ckpt_path: str,
) -> None:
    print("\tTRAINING")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = _get_loss_criterion()

    num_classes = len(get_class_names())
    for epoch in range(num_epochs):
        start_time = time.time()

        # run epoch
        train_loss = _train_epoch(model, optimizer, train_dataloader, criterion, device)
        val_loss = _test_epoch(model, val_dataloader, criterion, device)
        miou = _compute_miou(model, val_dataloader, device, num_classes=num_classes)

        # log losses
        wandb.log(
            {"train/loss": train_loss, "val/loss": val_loss, "val/mean_iou": miou}
        )

        # checkpoint model
        torch.save(model.state_dict(), ckpt_path)

        time_taken = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} : train Loss: {train_loss:.4f}"
            + f" - val loss: {val_loss:.4f} - val mIoU: {miou:.4f} -- time: {time_taken:.2f}s"
        )


def test_model(
    model: UNet,
    test_dataloader: DataLoader,
    device: torch.device,
) -> None:
    print("\tTESTING")

    criterion = _get_loss_criterion()

    num_classes = len(get_class_names())
    start_time = time.time()

    test_loss = _test_epoch(model, test_dataloader, criterion, device)
    miou = _compute_miou(model, test_dataloader, device, num_classes=num_classes)

    wandb.log({"test/loss": test_loss, "test/mean_iou": miou})

    time_taken = time.time() - start_time
    print(
        f"test results: test loss: {test_loss:.4f}"
        + f" val mIoU: {miou:.4f} -- time: {time_taken:.2f}s"
    )
