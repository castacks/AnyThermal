# Simple Segmentation Head
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

import sys
from .base_model import *
from .mmdistill_dinov2_model import MMDistillDinov2
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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def interpolate_batchwise(self, x, scale_factor, mode='bilinear', align_corners=False):
        """
        Perform interpolation in mini-batches to avoid int32 overflow.

        Args:
            x: (B, C, H, W) tensor
            scale_factor: float or tuple of (scale_h, scale_w)
            mode: interpolation mode
            align_corners: see torch.nn.functional.interpolate
        """
        max_elements = torch.iinfo(torch.int32).max
        b, c, h, w = x.shape
        target_h = int(h * scale_factor if isinstance(scale_factor, float) else h * scale_factor[0])
        target_w = int(w * scale_factor if isinstance(scale_factor, float) else w * scale_factor[1])

        # Compute max batch size
        max_batch_size = max(max_elements // (c * target_h * target_w), 1)

        chunks = []
        for start in range(0, b, max_batch_size):
            end = min(start + max_batch_size, b)
            x_chunk = x[start:end]
            up_chunk = F.interpolate(x_chunk, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
            chunks.append(up_chunk)
        return torch.cat(chunks, dim=0)

    def forward(self, x):
        # Input: (B, 768, H/14, W/14)
        x = self.up1(x)
        x = self.interpolate_batchwise(x, scale_factor=2., mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = self.interpolate_batchwise(x, scale_factor=2., mode='bilinear', align_corners=False)

        x = self.up3(x)
        x = self.interpolate_batchwise(x, scale_factor=2., mode='bilinear', align_corners=False)

        x = self.up4(x)
        x = self.interpolate_batchwise(x, scale_factor=1.75, mode='bilinear', align_corners=False)  # 1.75 × 8 = 14

        x = self.final(x)
        return x

class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=12,dropout_prob= 0.):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead128(nn.Module):
    def __init__(self, in_channels=768, num_classes=12,dropout_prob= 0.):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),  # Dropout here for regularization
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead64(nn.Module):
    def __init__(self, in_channels=768, num_classes=12,dropout_prob= 0.):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),  # Dropout here for regularization
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.upscale = True

    def forward(self, x):
        return self.model(x)

class NonLinearSegmentationHead64GELU(nn.Module):
    def __init__(self, in_channels=768, num_classes=12,dropout_prob= 0.):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_prob),  # Dropout here for regularization
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
        if "bilinear" in self.upscale_method:
            self.upscale_fn = self.upscale_bilinear
        else:
            raise ValueError(f"Unsupported upscale method: {self.upscale_method}")
    
    def train(self):
        self.backbone.train()
        self.head.train()
    
    def eval(self):
        self.backbone.eval()
        self.head.eval()

    def forward(self, x):
        x_padded = pad_to_multiple(x, 14)[0]  # Pad to multiple of 14
        w_new, h_new = x_padded.shape[-2], x_padded.shape[-1]
        x_shape = x.shape[-2:]
        features = self.backbone_forward(x_padded)
        del x

        if "bilinear" in self.upscale_method:
            del x_padded
            if self.head.upscale:
                if self.upscale_method == "pre_bilinear":
                    with torch.no_grad() if self.backbone.un_frozen_layer_index == [] else contextlib.nullcontext():
                        feat_upscaled = self.upscale_fn(features,None, size=(w_new,h_new))
                    out_upscaled = self.head(feat_upscaled)
                elif self.upscale_method == "bilinear":
                    out = self.head(features)
                    out_upscaled = self.upscale_fn(out,None, size=(w_new,h_new))
            else:
                out_upscaled = self.head(features)
        else:
            raise ValueError(f"Unsupported upscale method: {self.upscale_method}")

        final_scale = crop_to_shape(out_upscaled, x_shape[0],x_shape[1])  # Crop to original size
        return final_scale
    
    def upscale_bilinear(self,out,x,size):
        output = []
        for i in range(out.shape[0]):
            img = out[i:i+1]
            img = torch_F.interpolate(img, size=size, mode='bilinear', align_corners=False)
            output.append(img)
        return torch.cat(output, dim=0)
    
    def backbone_forward(self, imgs):
        with torch.no_grad() if self.backbone.un_frozen_layer_index == [] else contextlib.nullcontext():
            return self.backbone.forward_train(imgs, preprocess = True)['block_final_output'][0]

seg_head_str_to_dict = {
    'dpt': DPTSegmentationHead,
    'linear': LinearSegmentationHead,
    'non_linear_128': NonLinearSegmentationHead128,
    'non_linear_64': NonLinearSegmentationHead64,
    'non_linear_256': NonLinearSegmentationHead256,
    'non_linear_64_gelu': NonLinearSegmentationHead64GELU,
}

class MMDistillSegmentationModel(BaseSegmentationModel):
    def __init__(self,args,backbone_model_type,head_model, un_frozen_layer_index,frozen_head,modality,device,num_classes,upscale_method="bilinear",backbone_path="",model_path="",   **kwargs):
        self.head_model = head_model
        self.un_frozen_layer_index = un_frozen_layer_index
        self.dropout_prob = args.dropout_prob if hasattr(args,'dropout_prob') else 0.0
        
        self.backbone_path = backbone_path
        self.frozen_head = frozen_head
        self.model_path = model_path
        self.modality = modality
        self.upscale_method = upscale_method
        self.backbone_model_type = backbone_model_type
        super().__init__(device, num_classes)

    def build_model(self):
        assert self.head_model in seg_head_str_to_dict, f"Unsupported head model: {self.head_model}. Supported models are: {list(seg_head_str_to_dict.keys())}"
        head_type  = seg_head_str_to_dict[self.head_model]
        backbone_path = self.backbone_path
        backbone_model_type = self.backbone_model_type
        head_state_dict = None
        if self.backbone_path!= "" and self.model_path != "":
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")
        if self.backbone_path!= "":
            print(f"Loading backbone from {self.backbone_path}")
            backbone_model_type = torch.load(backbone_path, map_location=self.device,weights_only=True)["student_model_type"]
        elif self.model_path != "":
            print(f"Loading backbone and head model from {self.model_path}")
            head_state_dict = torch.load(self.model_path, map_location=self.device,weights_only=True)["seg_head"]
            backbone_path = torch.load(self.model_path, map_location=self.device,weights_only=True)["backbone_path"]
            if backbone_path != "":
                backbone_model_type = torch.load(backbone_path, map_location=self.device,weights_only=True)["student_model_type"]

        backbone = MMDistillDinov2(backbone_model_type, self.modality, backbone_path=backbone_path,un_frozen_layer_index=self.un_frozen_layer_index,layers_to_hook=['final'])

        head = head_type(in_channels=backbone.model.embed_dim, num_classes=self.num_classes,dropout_prob=self.dropout_prob).to(self.device)
        if head_state_dict is not None:
            head.load_state_dict(head_state_dict)

        model = BaseDinov2SegmentationModel(backbone, head,self.upscale_method).to(self.device)
        if self.frozen_head:
            model.head.eval()
        return model
    
    def preprocess(self, images):
        return None
    
    def forward(self, x):
        return self.model(x)
    
    def unfrozen_parameters(self):
        output = self.model.backbone.unfrozen_parameters()
        if not self.frozen_head:
            output.extend(list(self.model.head.parameters()))
        return output
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def parameters(self):
        output = list(self.model.backbone.model.parameters())
        output.extend(list(self.model.head.parameters()))
        return output