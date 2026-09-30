
import torch
import segmentation_models_pytorch as smp


def bce_dice_loss(logits, targets):
    """
    Baseline BCE + Dice loss.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    dice = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    return bce + dice(logits, targets)


def focal_dice_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    Focal + Dice loss.
    """
    focal = smp.losses.FocalLoss(
        mode="binary",
        alpha=alpha,
        gamma=gamma,
    )

    dice = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    return focal(logits, targets) + dice(logits, targets)


def weighted_bce_dice_loss(
    logits,
    targets,
    pos_weight=2.0,
):
    """
    Weighted BCE + Dice loss.

    pos_weight increases the contribution of building pixels
    relative to background pixels.
    """
    weight = torch.tensor(
        [pos_weight],
        dtype=logits.dtype,
        device=logits.device,
    )

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight,
    )

    dice = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    return bce + dice(logits, targets)
