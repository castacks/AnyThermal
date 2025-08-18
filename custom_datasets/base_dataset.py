import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

import os
import numpy as np
import cv2
import torch
import torch.utils.data 
from typing import List, Union
from natsort import natsorted
from configs import prog_args
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean

from torch.utils.data import DataLoader

import os
import torch
import faiss
import numpy as np
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from typing import Tuple
from torchvision.transforms import functional as F
import random
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFilter

base_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # normalization is done in the model since it can be different for each modal
])


class BaseDataset(Dataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 25,rescale_during_crop=False,crop_during_vpr_test=False):
        self.args = args
        self.augment = augment
        self.crop_during_vpr_test = crop_during_vpr_test
        super().__init__()
        self.crop_images = crop_images
        self.vpr_test = vpr_test
        self.vpr_train = vpr_train
        self.datasets_folder = datasets_folder
        self.rescale_during_crop = rescale_during_crop
        if seq == []:
            raise ValueError("Please provide a sequence name. Input is a list")
        if not isinstance(seq,list):
            raise ValueError("Sequence name(s) should be a list")

        if self.vpr_test:
            if self.augment or self.rescale_during_crop:
                raise ValueError("For VPR test, augmentations and rescale_during_crop should be False")
        self.check_seq_list(seq)
        self.seq = seq

        
        self.dist_thresh = dist_thresh
        self.db_modality = db_modality
        self.q_modality = q_modality
        print("seq: ",self.seq)
        print("db_modality: ",self.db_modality)
        print("q_modality: ",self.q_modality)
        self.db_abs_paths =[]
        self.q_abs_paths = []

        self.db_abs_paths,self.q_abs_paths = self.generate_image_paths(self.db_abs_paths,self.q_abs_paths)
        
        self.database_num = len(self.db_abs_paths)
        self.queries_num = len(self.q_abs_paths)
        if self.vpr_test or self.vpr_train:
            self.form_db_qu_coords()
            assert len(self.db_coords) == self.database_num, f"Database coordinates length {len(self.db_coords)} does not match database number {self.database_num}"
            assert len(self.q_coords) == self.queries_num, f"Queries coordinates length {len(self.q_coords)} does not match queries number {self.queries_num}"

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

        self.read_fn=self.generate_read_fn()
        self.semantic_classes , self.semantic_id_to_rgb= self.semantic_classes_num_and_map_to_rgb()

    @abstractmethod
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        """
        Generates image paths for the dataset. Return the updated db_abs_paths and q_abs_paths
        """
        pass
    
    def __len__(self):
        if self.vpr_test:
            return self.database_num + self.queries_num
        else:
            return self.database_num

    @abstractmethod
    def generate_read_fn(self):
        """
        Generates read function for the dataset. in the below example, self.read_rgb and self.read_thermal are the read functions for rgb and thermal images respectively.
        e.g. read_fn = {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }
        """
        pass
    
    @abstractmethod
    def check_seq_list(self,seq):
        """
        Checks if the sequence list is valid.
        """
        pass

    @abstractmethod
    def form_db_qu_coords(self):
        """
        Returns ground truth positives for the dataset.
        """
        pass
    
    @abstractmethod
    def semantic_classes_num_and_map_to_rgb(self):
        """
        Return num of sematic classes and dict for the mapping between semantic class and RGB in the dataset. If not a semantic dataset return -1,{}
        """
        pass


    def augment_function(self, modality1: str, modality2: str, img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if modality1 == "thr" and modality2 == "rgb":
            img2, img1 = self.rgb_thermal_augment(img2, img1)
        elif modality1 == "rgb" and modality2 == "thr":
            img1, img2 = self.rgb_thermal_augment(img1, img2)
        elif (modality1 == "thr_seg" or modality1 == "thr") and modality2 == "seg_mask":
            img1, img2 = self.thermal_seg_augment(img1,img2)
        elif modality1 == "rgb" and modality2 == "seg_mask":
            img1, img2 = self.thermal_seg_augment(img1,img2)
        else:
            raise ValueError(f"Unsupported modality combination: {modality1}, {modality2}")
        return img1, img2  # No augmentation if modalities are not recognized
    


    def thermal_seg_augment(self, img1: torch.Tensor, img2: torch.Tensor, 
                            crop_scale_range: Tuple[float, float]=(0.5, 1.0)) -> Tuple[torch.Tensor, torch.Tensor]:
        
        resize_target = self.dataset_shape
        aug_list = set(self.args.thermal_segmentation_augmentation)

        if img1.ndim == 2:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 2:
            img2 = img2.unsqueeze(0)

        _, h, w = img1.shape

        if "hflip" in aug_list and random.random() > 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)


        if "brightness_contrast" in aug_list:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            img1 = torch.clamp(img1 * brightness_factor * contrast_factor, 0, 1)

        if "noise" in aug_list and random.random() < 0.3:
            noise = torch.randn_like(img1) * 0.02
            img1 = torch.clamp(img1 + noise, 0, 1)

        if "gamma" in aug_list:
            gamma = random.uniform(0.9, 1.1)
            img1 = torch.pow(img1, gamma)

        if "crop_with_random_ratio" in aug_list and random.random() < 0.5:
            crop_scale = random.uniform(*crop_scale_range)
            crop_h = int(h * crop_scale)
            crop_w = int(w * crop_scale)
            min_crop = min(crop_h, crop_w)
            crop_h, crop_w = min_crop, min_crop
            if crop_h < h and crop_w < w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                img1 = img1[:, top:top+crop_h, left:left+crop_w]
                img2 = img2[:, top:top+crop_h, left:left+crop_w]

            img1 = F.resize(img1, resize_target, interpolation=InterpolationMode.BILINEAR, antialias=True)
            img2 = F.resize(img2, resize_target, interpolation=InterpolationMode.NEAREST, antialias=True)
        
        if "crop_with_fixed_ratio" in aug_list:
            target_h, target_w = resize_target
            assert target_h == target_w, "resize_target must be square for aspect ratio preservation"

            # Case 1: crop to center region of target size if large enough
            if h >= target_h and w >= target_w:
                top = (h - target_h) // 2
                left = (w - target_w) // 2
                img1 = img1[:, top:top+target_h, left:left+target_w]
                img2 = img2[:, top:top+target_h, left:left+target_w]

            else:
                # Case 2: Take center square crop of max possible size
                side = min(h, w)
                top = (h - side) // 2
                left = (w - side) // 2
                img1 = img1[:, top:top+side, left:left+side]
                img2 = img2[:, top:top+side, left:left+side]

                # Resize to target
                img1 = F.resize(img1, resize_target, interpolation=InterpolationMode.BILINEAR, antialias=True)
                img2 = F.resize(img2, resize_target, interpolation=InterpolationMode.NEAREST, antialias=True)

        return img1, img2

    def crop_and_resize(self, img1: Image.Image,img2: Image.Image,size: Tuple[int, int] = (308,504)) -> Image.Image:
        """
        Center crops and resizes the image to the specified size.
        """
        i, j, h, w = T.RandomResizedCrop.get_params(img1, scale=(0.8, 1.0), ratio=(0.75, 1.33))

 
    def crop_pair_fixed_size(self, img1: Image.Image, img2: Image.Image, size: Tuple[int, int] = (308,504)) -> Tuple[Image.Image, Image.Image]:
        """
        Applies the same fixed-size random crop to both images without rescaling.
        Args:
            img1: PIL Image (e.g., RGB)
            img2: PIL Image (e.g., thermal)
            size: (height, width) of crop
        Returns:
            Tuple of cropped img1 and img2
        """
        original_height, original_width = img1.shape[-2:]  # Assuming img1 is a PIL Image
        target_height, target_width = size

        if original_height < target_height or original_width < target_width:
            return img1, img2  # Return original images if they are smaller than the target size

        top = random.randint(0, original_height - target_height)
        left = random.randint(0, original_width - target_width)

        cropped1 = F.crop(img1, top, left, target_height, target_width)
        cropped2 = F.crop(img2, top, left, target_height, target_width)

        return cropped1, cropped2
    
    def crop_width_resize_fixed_size(self,img1: torch.Tensor, img2: torch.Tensor, target_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the same fixed-size random crop to two tensors (C, H, W).
        Args:
            img1: Tensor (C, H, W), e.g., RGB
            img2: Tensor (C, H, W), e.g., thermal
            target_size: final size to resize to (target_size x target_size)
        Returns:
            Tuple of cropped and resized tensors
        """
        _, H, W = img1.shape

        # Ensure height <= width for horizontal crop
        assert H <= W, "Image height should be <= width for this crop logic"

        crop_size = H  # we take a square crop: height x height
        max_left = W - crop_size

        # Randomly choose horizontal crop start
        left = random.randint(0, max_left)
        top = 0  # no vertical crop

        # Apply the crop
        img1_cropped = F.crop(img1, top, left, crop_size, crop_size)
        img2_cropped = F.crop(img2, top, left, crop_size, crop_size)

        # Resize to target size
        img1_resized = F.resize(img1_cropped, [target_size, target_size], interpolation=F.InterpolationMode.BILINEAR,antialias=True)
         # Use bilinear interpolation for RGB images
        img2_resized = F.resize(img2_cropped, [target_size, target_size], interpolation=F.InterpolationMode.BILINEAR,antialias=True)

        return img1_resized, img2_resized

    def rgb_thermal_augment(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aug_list = self.args.aug_list

        # ----- Random Parameters -----
        rgb_brightness_factor = random.uniform(0.7, 1.3)
        rgb_contrast_factor = random.uniform(0.8, 1.2)
        rgb_gamma = random.uniform(0.7, 1.5)

        thermal_brightness_factor = random.uniform(0.7, 1.3)
        thermal_contrast_factor = random.uniform(0.8, 1.2)
        thermal_gamma = random.uniform(0.7, 1.5)

        saturation_factor = random.uniform(0.2, 1.2)
        hue_factor = random.uniform(-0.05, 0.05)
        do_flip = random.random() > 0.5

        rgb_shape = rgb.shape[-2:]

        # ----- Synchronized Affine -----
        if "affine" in aug_list:
            angle = random.uniform(-10, 10)
            translate = [random.uniform(-0.02, 0.02) * rgb_shape[1],
                        random.uniform(-0.02, 0.02) * rgb_shape[0]]
            scale = random.uniform(0.95, 1.05)
            shear = random.uniform(-5, 5)
            rgb = F.affine(rgb, angle=angle, translate=translate, scale=scale, shear=[shear], interpolation=F.InterpolationMode.BILINEAR)
            thermal = F.affine(thermal, angle=angle, translate=translate, scale=scale, shear=[shear], interpolation=F.InterpolationMode.BILINEAR)

        # ----- Horizontal Flip -----
        if do_flip:
            rgb = F.hflip(rgb)
            thermal = F.hflip(thermal)

        # ----- Brightness -----
        if "brightness" in aug_list:
            rgb = F.adjust_brightness(rgb, rgb_brightness_factor)
            thermal = F.adjust_brightness(thermal, thermal_brightness_factor)

        # ----- Contrast -----
        if "contrast" in aug_list:
            rgb = F.adjust_contrast(rgb, rgb_contrast_factor)
            thermal = F.adjust_contrast(thermal, thermal_contrast_factor)

        # ----- Gamma Correction -----
        if "gamma" in aug_list:
            rgb = F.adjust_gamma(rgb, gamma=rgb_gamma)
            thermal = F.adjust_gamma(thermal, gamma=thermal_gamma)

        # ----- RGB-only: Saturation & Hue -----
        if "color_jitter" in aug_list:
            rgb = F.adjust_saturation(rgb, saturation_factor)
            rgb = F.adjust_hue(rgb, hue_factor)

        # ----- CLAHE for Thermal -----
        if "clahe" in aug_list:
            thermal_np = np.array(F.to_pil_image(thermal).convert("L"))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            thermal_np = clahe.apply(thermal_np)
            thermal = F.to_tensor(Image.fromarray(thermal_np))

        # ----- Gaussian Blur for Thermal -----
        if "blur" in aug_list:
            thermal_pil = F.to_pil_image(thermal)
            thermal = F.to_tensor(thermal_pil.filter(ImageFilter.GaussianBlur(radius=1.0)))

        # ----- Cutout (synchronized) -----
        if "cutout" in aug_list:
            _, H, W = rgb.shape
            cutout_size = int(0.1 * min(H, W))
            x0 = random.randint(0, W - cutout_size)
            y0 = random.randint(0, H - cutout_size)
            rgb[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0
            thermal[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0

        return rgb, thermal
    def __getitem__(self, index):

        if self.vpr_test:
            if index>=self.database_num:
                img = self.read_fn[self.q_modality](self.images_paths[index])

            elif index<self.database_num:
                img = self.read_fn[self.db_modality](self.images_paths[index])
            
            if self.crop_during_vpr_test:
                raise ValueError("crop_during_vpr_test is not implemented yet")
            # print("Image shape:", img.shape)

            return img, index

        else:
            db_img = self.read_fn[self.db_modality](self.images_paths[index])
            q_img = self.read_fn[self.q_modality](self.images_paths[self.database_num+index])
            if self.crop_images:
                if self.rescale_during_crop:
                    # Apply crop and resize with fixed size
                    db_img, q_img = self.crop_width_resize_fixed_size(db_img, q_img)
                else:
                    # Apply fixed size crop without rescaling
                    db_img, q_img = self.crop_pair_fixed_size(db_img, q_img)

            if self.augment:
                # Apply augmentations if training mode and augmentations are enabled
                db_img,q_img = self.augment_function(self.db_modality, self.q_modality, db_img, q_img)
            return {self.db_modality:db_img,self.q_modality:q_img}, index
