
from pathlib import Path

import torch
import pandas as pd


def calculate_metrics(logits, masks, threshold=0.5):
    """
    Calculate IoU, Dice/F1, precision, and recall.
    """

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(-1)
    masks = masks.view(-1)

    true_positive = (predictions * masks).sum()
    false_positive = (predictions * (1 - masks)).sum()
    false_negative = ((1 - predictions) * masks).sum()

    eps = 1e-7

    iou = true_positive / (
        true_positive + false_positive + false_negative + eps
    )

    dice = 2 * true_positive / (
        2 * true_positive + false_positive + false_negative + eps
    )

    precision = true_positive / (
        true_positive + false_positive + eps
    )

    recall = true_positive / (
        true_positive + false_negative + eps
    )

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    threshold=0.5,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0

    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_function(logits, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()

        true_positive += (
            predictions * masks
        ).sum().item()

        false_positive += (
            predictions * (1 - masks)
        ).sum().item()

        false_negative += (
            (1 - predictions) * masks
        ).sum().item()

    eps = 1e-7

    iou = true_positive / (
        true_positive + false_positive + false_negative + eps
    )

    dice = 2 * true_positive / (
        2 * true_positive + false_positive + false_negative + eps
    )

    precision = true_positive / (
        true_positive + false_positive + eps
    )

    recall = true_positive / (
        true_positive + false_negative + eps
    )

    return {
        "loss": total_loss / len(loader),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_function,
    device,
    threshold=0.5,
):
    """
    Evaluate the model for one validation epoch.
    """

    model.eval()

    total_loss = 0.0

    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = loss_function(logits, masks)

        total_loss += loss.item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()

        true_positive += (
            predictions * masks
        ).sum().item()

        false_positive += (
            predictions * (1 - masks)
        ).sum().item()

        false_negative += (
            (1 - predictions) * masks
        ).sum().item()

    eps = 1e-7

    iou = true_positive / (
        true_positive + false_positive + false_negative + eps
    )

    dice = 2 * true_positive / (
        2 * true_positive + false_positive + false_negative + eps
    )

    precision = true_positive / (
        true_positive + false_positive + eps
    )

    recall = true_positive / (
        true_positive + false_negative + eps
    )

    return {
        "loss": total_loss / len(loader),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    device,
    epochs,
    checkpoint_dir,
    metrics_dir,
    experiment_name="experiment",
    threshold=0.5,
):
    """
    Train a segmentation model and save the best checkpoint.

    The best model is selected using validation IoU.
    """

    checkpoint_dir = Path(checkpoint_dir)
    metrics_dir = Path(metrics_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_iou = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            threshold=threshold,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            threshold=threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]
            best_epoch = epoch

            checkpoint_path = checkpoint_dir / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": best_iou,
                    "experiment": experiment_name,
                },
                checkpoint_path,
            )

            print(
                f"  Best checkpoint saved "
                f"(epoch {epoch}, IoU {best_iou:.4f})"
            )

    history_df = pd.DataFrame(history)

    history_path = (
        metrics_dir / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", round(best_iou, 4))
    print("Checkpoint:", checkpoint_dir / "best_model.pth")
    print("History:", history_path)

    return history_df, best_epoch
