import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_F
from torchvision import transforms as T

def label_to_rgb(mask_tensor,ID_TO_RGB):
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.cpu().numpy()
    elif isinstance(mask_tensor, np.ndarray):
        mask_np = mask_tensor
    else:
        raise ValueError("Input must be a torch.Tensor or numpy.ndarray")
    h, w = mask_np.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in ID_TO_RGB.items():
        rgb_img[mask_np == k] = color
    return Image.fromarray(rgb_img)



# def transform_images(images,mask=False,patch_size=14):
#     # Resize the images to the required input size
#     # import pdb; pdb.set_trace()
#     width = images.shape[-2]
#     height = images.shape[-1]
#     new_width = (width // patch_size) * patch_size
#     new_height = (height // patch_size) * patch_size
#     if mask:
#         images = torch_F.interpolate(images, size=(new_width, new_height), mode='nearest')
#     else:
#         images = torch_F.interpolate(images, size=(new_width, new_height), mode='bilinear', align_corners=False)
#     return images
# Training pipeline