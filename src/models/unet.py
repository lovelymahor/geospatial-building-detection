
import segmentation_models_pytorch as smp


def create_unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
):
    """
    Create a U-Net model for binary building segmentation.
    """

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )

    return model
