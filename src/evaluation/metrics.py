
import numpy as np


def calculate_confusion_counts(
    prediction,
    target,
):
    """
    Calculate pixel-level confusion counts.

    Parameters
    ----------
    prediction : array-like
        Binary predicted mask.
    target : array-like
        Binary ground-truth mask.

    Returns
    -------
    dict
        TP, FP, FN, and TN counts.
    """

    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)

    true_positive = np.logical_and(
        prediction,
        target,
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~target,
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        target,
    ).sum()

    true_negative = np.logical_and(
        ~prediction,
        ~target,
    ).sum()

    return {
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
    }


def calculate_segmentation_metrics(
    prediction,
    target,
):
    """
    Calculate segmentation metrics from binary masks.
    """

    counts = calculate_confusion_counts(
        prediction,
        target,
    )

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]

    eps = 1e-7

    iou = tp / (
        tp + fp + fn + eps
    )

    dice = 2 * tp / (
        2 * tp + fp + fn + eps
    )

    precision = tp / (
        tp + fp + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    return {
        **counts,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def calculate_error_rates(
    prediction,
    target,
):
    """
    Calculate false-positive and false-negative rates.
    """

    counts = calculate_confusion_counts(
        prediction,
        target,
    )

    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tn = counts["true_negative"]
    tp = counts["true_positive"]

    eps = 1e-7

    false_positive_rate = fp / (
        fp + tn + eps
    )

    false_negative_rate = fn / (
        fn + tp + eps
    )

    return {
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }
