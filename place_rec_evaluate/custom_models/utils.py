from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF


normalise_model_mean_std = {
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'vit': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    }
}

def mono_to_rgb(img_np: np.ndarray):
    """
    Converts a grayscale image (H x W or H x W x 1) to RGB by replicating channels.
    """
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=-1)
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    return img_np
# def to_np(img, dtype=np.uint8):
#     """
#     Converts a torch tensor or PIL image to a numpy array.
#     - Tensor should be [C, H, W] and in [0, 1] or [0, 255] range.
#     """
#     if isinstance(img, torch.Tensor):
#         img = img.detach().cpu()
#         if img.dim() == 3:
#             img = img.permute(1, 2, 0)  # [H, W, C]
#         img = img.numpy()
#     elif isinstance(img, Image.Image):
#         img = np.array(img)
#     else:
#         raise TypeError(f"Unsupported image type: {type(img)}")

#     if img.dtype != dtype:
#         img = img.astype(dtype)

#     return img

def to_np(x, ret_type=float) -> np.ndarray:
    """
        Converts 'x' to numpy object of `dtype` as 'ret_type'
        
        Parameters:
        - x:    An object
        
        Returns:
        - x_np:     A numpy array of dtype `ret_type`
    """
    x_np: np.ndarray = None
    if type(x) == torch.Tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)
    x_np = x_np.astype(ret_type)
    return x_np

def normalise_img(img_np: np.ndarray):
    """
    Normalises the image to [0, 1] range.
    """
    norm_img_np = (img_np - img_np.min())/(img_np.max() - img_np.min())
    norm_img_np = (norm_img_np * 255)
    return norm_img_np
# def pad_img(img_np: np.ndarray, padding: int = 10, color=(255, 0, 0)):
#     """
#     Pads a numpy image with a given RGB color border.
#     """
#     if img_np.ndim == 2:
#         img_np = np.expand_dims(img_np, axis=-1)
#     if img_np.shape[-1] == 1:
#         img_np = np.repeat(img_np, 3, axis=-1)
#     h, w, c = img_np.shape
#     padded = np.ones((h + 2 * padding, w + 2 * padding, c), dtype=img_np.dtype) * np.array(color, dtype=img_np.dtype)
#     padded[padding:padding + h, padding:padding + w] = img_np
#     return padded
def pad_img(img: np.ndarray, padding:int, color:tuple=(0, 0, 0)) \
        -> np.ndarray:
    """
        Pad an image with 'padding' along each side (height and width)
        and fill the padding with 'color'.
        
        Parameters:
        - img:  Image of shape [H, W, C=3] with channels as RGB (same
                as the 'color' channels)
        - padding:      Padding 'P' (int) for each dimension (applied 
                        on both ends of axis)
        - color:    The RGB color of the padding
        
        Returns:
        - _img:     Image of shape [H+2P, W+2P, C=3]
    """
    if type(color) == list:
        color = tuple(color)
    assert len(color) == 3, "Color should be (R, G, B) value"
    color = np.array(color)
    # ret_img = np.pad(img, [(padding, padding), (padding, padding), 
    #             (0, 0)], constant_values=[(color, color), 
    #                         (color, color), (0, 0)])
    ret_img = np.ones((img.shape[0] + 2*padding, 
                img.shape[1] + 2*padding, 3), np.uint8) * color
    ret_img[padding:-padding, padding:-padding] = img
    return ret_img.astype(img.dtype)

def default_preprocess(image_path, size=224):
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image)

import torch
import torch.nn.functional as F

def default_preprocess_tensor(images: torch.Tensor, normalise_model,size=(224,224),normalise=True,resize=True):
    """
    Preprocess a batch of image tensors. Input is a tensor of shape [B, C, H, W].
    Resizes, normalizes using ImageNet mean/std.
    """
    assert normalise_model in normalise_model_mean_std, f"Model {normalise_model} not supported. Choose from {list(normalise_model_mean_std.keys())}"
    if images.dtype != torch.float32:
        images = images.float() / 255.0  # Scale if originally uint8

    if isinstance(size, int):
        size = (size, size)

    # Resize using bilinear interpolation
    if resize:
        images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)

    # Normalize using ImageNet mean/std
    if normalise:
        mean = torch.tensor(normalise_model_mean_std[normalise_model]['mean'], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(normalise_model_mean_std[normalise_model]['std'], device=images.device).view(1, 3, 1, 1)
        
        images = (images - mean) / std
    return images

def default_preprocess_tensor_keep_ratio_center_crop(images: torch.Tensor,normalise_model, size=224,normalise=True,resize=True):
    """
    Resizes the shortest edge of input images to `size` while maintaining aspect ratio,
    then center-crops to `crop_size` × `crop_size`, and normalizes with ImageNet stats.

    Args:
        images (Tensor): Input tensor of shape [B, C, H, W], assumed RGB and in [0, 255] or [0, 1]
        size (int): Target size for shortest side during resizing
        crop_size (int): Output size after center cropping

    Returns:
        Tensor: Preprocessed tensor of shape [B, C, crop_size, crop_size]
    """
    if images.dtype != torch.float32:
        images = images.float() / 255.0  # Scale to [0, 1] if uint8

    B, C, H, W = images.shape

    # Compute new size maintaining aspect ratio
    if resize:
        scale = size / min(H, W)
        new_H, new_W = int(round(H * scale)), int(round(W * scale))
        images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # Center crop
        top = (new_H - size) // 2
        left = (new_W - size) // 2
        images = images[:, :, top:top+size, left:left+size]

    # Normalize with CLIP/ViT stats
    if normalise:
        mean = torch.tensor(normalise_model_mean_std[normalise_model]['mean'], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(normalise_model_mean_std[normalise_model]['std'], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

    return images

def preprocess_dinov2(images: torch.Tensor, normalise_model, patch_size=14, normalise=True):
    #resize to w and H such that it is divisible by patch_size
    B, C, H, W = images.shape
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size
    images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)

    if normalise:
        mean = torch.tensor(normalise_model_mean_std[normalise_model]['mean'], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(normalise_model_mean_std[normalise_model]['std'], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
    
    return images


def compute_similarity(query_feat, db_feats, top_k=5):
    similarities = [F.cosine_similarity(query_feat, db_feat, dim=0).item() for db_feat in db_feats]
    indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return indices, [similarities[i] for i in indices]
