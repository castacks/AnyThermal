# Simple Segmentation Head
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

import sys
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/loftup/upsamplers')
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/loftup')
from upsamplers import load_loftup_checkpoint
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
def transform_images(images,patch_size=14):
    # Resize the images to the required input size
    # import pdb; pdb.set_trace()
    width = images.shape[-2]
    height = images.shape[-1]
    new_width = (width // patch_size) * patch_size
    new_height = (height // patch_size) * patch_size
    images = torch_F.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
    import pdb; pdb.set_trace()
    return images

def pad_to_multiple(x, multiple):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)  # pad (left, right, top, bottom)
    x_padded = torch_F.pad(x, padding, mode='reflect')
    return x_padded, padding

def crop_to_shape(x, target_h, target_w):
    return x[:, :, :target_h, :target_w]

class DPTSegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.upscale = False

    def forward(self, x):
        # Input: (B, 768, H/14, W/14)
        x = self.up1(x)
        x = torch_F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = torch_F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.up3(x)
        x = torch_F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.up4(x)
        x = torch_F.interpolate(x, scale_factor=1.75, mode='bilinear', align_corners=False)  # 1.75 × 8 = 14

        x = self.final(x)
        return x

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
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead128(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead64(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead256(nn.Module):
    def __init__(self, in_channels=768, num_classes=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class BaseDinov2SegmentationModel(nn.Module):
    def __init__(self, backbone, head,upscale_method):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.upscale_method = upscale_method
        if self.upscale_method == "bilinear":
            self.upscale_fn = self.upscale_bilinear
        elif self.upscale_method == "loftup":
            model_path = hf_hub_download(repo_id="haiwen/loftup-dinov2b", filename="pytorch_model.bin")
            self.upscale_model = load_loftup_checkpoint(model_path, n_dim=768, lr_pe_type="sine", lr_size=14)
            self.upscale_model.eval()
            self.upscale_fn = self.loft_upscale
        else:
            raise ValueError(f"Unsupported upscale method: {self.upscale_method}")
    

    def forward(self, x):
        x_padded = pad_to_multiple(x, 14)[0]  # Pad to multiple of 14
        w_new, h_new = x_padded.shape[-2], x_padded.shape[-1]

        features = self.backbone_forward(x_padded)
        if self.upscale_method =="loftup":
            features_scaled = self.upscale_fn(features,x, size=(w_new,h_new))
            out_upscaled = self.head(features_scaled)
        elif self.upscale_method == "bilinear":
            out = self.head(features)
            out_upscaled = self.upscale_fn(out,x, size=(w_new,h_new))
        # if self.head.upscale:
        #     out_upscaled = self.upscale_fn(out,x, size=(w_new,h_new))
        # else:
        #     out_upscaled = out

        final_scale = crop_to_shape(out_upscaled, x.shape[-2], x.shape[-1])  # Crop to original size

        return final_scale
    
    def upscale_bilinear(self,out,x,size):
        return torch_F.interpolate(out, size=size, mode='bilinear', align_corners=False)

    def loft_upscale(self,out,x,size):
        return self.upscale_model(out, x)
    
    def backbone_forward(self, imgs):
        return self.backbone.forward_train(imgs, preprocess = True,return_local_features= True)[0] #B,C,H,W

seg_head_str_to_dict = {
    'dpt': DPTSegmentationHead,
    'base': SegmentationHead,
    'linear': LinearSegmentationHead,
    'non_linear_128': NonLinearSegmentationHead128,
    'non_linear_64': NonLinearSegmentationHead64,
    'non_linear_256': NonLinearSegmentationHead256,
}