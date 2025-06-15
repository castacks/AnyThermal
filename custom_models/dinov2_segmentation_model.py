# Simple Segmentation Head
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

def transform_images(images,patch_size=14):
    # Resize the images to the required input size
    # import pdb; pdb.set_trace()
    width = images.shape[-2]
    height = images.shape[-1]
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size
    images = torch_F.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
    return images

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

class BaseDinov2SegmentationModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    

    def forward(self, x, frozen_backbone=False):
        w_new,h_new = (x.shape[-2] // 14) * 14, (x.shape[-1] // 14) * 14
        if frozen_backbone:
            with torch.no_grad():
                features = self.backbone_forward(transform_images(x))
        else:
            features = self.backbone_forward(transform_images(x))
        out = self.head(features)

        out_upscaled = torch_F.interpolate(torch_F.interpolate(out, size=(w_new,h_new), mode='bilinear', align_corners=False)
                                                , size=x.shape[-2:], mode='bilinear', align_corners=False)

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