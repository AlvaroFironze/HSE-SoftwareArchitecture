import segmentation_models_pytorch as smp
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
class SegmentationModel(nn.Module):
    def __init__(self,model_name,encoder,weights):
        super(SegmentationModel, self).__init__()

        match model_name:
            case "UnetPlusPlus":
                self.model = smp.UnetPlusPlus(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "Unet":
                self.model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "MAnet":
                self.model = smp.MAnet(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "Linknet":
                self.model = smp.Linknet(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "FPN":
                self.model = smp.FPN(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "PSPNet":
                self.model = smp.PSPNet(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "DeepLabV3":
                self.model = smp.DeepLabV3(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "DeepLabV3Plus":
                self.model = smp.DeepLabV3Plus(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
            case "PAN":
                self.model = smp.PAN(
                    encoder_name=encoder,
                    encoder_weights=weights,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )

    def forward(self, images, masks=None):
        logits = self.model(images)

        if masks != None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            dice1 = dice(logits, masks)
            return logits, loss1 + loss2, dice1

        return logits
def dice(pred, label):
    pred = (pred > 0).float()
    return 2. * (pred*label).sum() / (pred+label).sum()