# Simple Segmentation Head
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
class SegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

class BaseDinov2SegmentationModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, frozen_backbone=False):
        if frozen_backbone:
            with torch.no_grad():
                features = self.backbone_forward(x)
        else:
            features = self.backbone_forward(x)
        out = self.head(features)
        out_upscaled = torch_F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return out_upscaled
    
    def backbone_forward(self, imgs):
        return self.backbone.get_intermediate_layers(imgs, n=1, reshape=True)[0]  # shape: [B, C, H, W]

class BaseDinov2SegmentationModelPreUpscaled(BaseDinov2SegmentationModel):
    """
    Base Dinov2 Segmentation Model with pre-upscaled features before passing to the head.
    """

    def forward(self, x, frozen_backbone=False):
        if frozen_backbone:
            with torch.no_grad():
                features = self.backbone_forward(x)
        else:
            features = self.backbone_forward(x)
        features_upscaled = torch_F.interpolate(features, size=x.shape[-2:], mode='bilinear', align_corners=False)

        out = self.head(features_upscaled)

        return out