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
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from typing import Tuple
from torchvision.transforms import functional as F
import random
# from utilities import CustomDataset
def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

base_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # normalization is done in the model since it can be different for each modal
])


class BaseDataset(Dataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 25,rescale_during_crop=False,crop_during_vpr_test=False):
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
            self.dist, self.soft_positives_per_query = self.form_gt_positives()
            assert len(self.soft_positives_per_query) == self.queries_num, f"Soft positives per query length {len(self.soft_positives_per_query)} does not match queries number {self.queries_num}"
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
    def form_gt_positives(self):
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
        else:
            raise ValueError(f"Unsupported modality combination: {modality1}, {modality2}")
        return img1, img2  # No augmentation if modalities are not recognized
    

    def crop_and_resize(self, img1: Image.Image,img2: Image.Image,size: Tuple[int, int] = (210,504)) -> Image.Image:
        """
        Center crops and resizes the image to the specified size.
        """
        i, j, h, w = T.RandomResizedCrop.get_params(img1, scale=(0.8, 1.0), ratio=(0.75, 1.33))

 
    def crop_pair_fixed_size(self, img1: Image.Image, img2: Image.Image, size: Tuple[int, int] = (210,504)) -> Tuple[Image.Image, Image.Image]:
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
            print("Warning: Image is smaller than target size, returning original images without cropping, original size:", (original_height, original_width), "target size:", (target_height, target_width))
            return img1, img2  # Return original images if they are smaller than the target size

        top = random.randint(0, original_height - target_height)
        left = random.randint(0, original_width - target_width)

        cropped1 = F.crop(img1, top, left, target_height, target_width)
        cropped2 = F.crop(img2, top, left, target_height, target_width)

        return cropped1, cropped2

    def rgb_thermal_augment(self,rgb: Image.Image, thermal: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # ----- Random Parameters -----
        rgb_brightness_factor = random.uniform(0.7, 1.3)
        rgb_contrast_factor = random.uniform(0.8, 1.2)

        thermal_brightness_factor = random.uniform(0.7, 1.3) 
        thermal_contrast_factor = random.uniform(0.8, 1.2)


        saturation_factor = random.uniform(0.2, 1.2)
        hue_factor = random.uniform(-0.05, 0.05)
        do_flip = random.random() > 0.5

        # # ----- Center Crop and Resize (300x450) -----
        # rgb, thermal = self.crop_and_resize(rgb, thermal)

        # ----- Horizontal Flip -----
        if do_flip:
            rgb = F.hflip(rgb)
            thermal = F.hflip(thermal)

        # ----- Brightness & Contrast (synchronized) -----
        rgb = F.adjust_brightness(rgb, rgb_brightness_factor)
        thermal = F.adjust_brightness(thermal, thermal_brightness_factor)

        rgb = F.adjust_contrast(rgb, rgb_contrast_factor)
        thermal = F.adjust_contrast(thermal, rgb_contrast_factor)

        # ----- RGB-only: Saturation & Hue -----
        rgb = F.adjust_saturation(rgb, saturation_factor)
        rgb = F.adjust_hue(rgb, hue_factor)

        return rgb, thermal
    def __getitem__(self, index):

        if self.vpr_test:
            if index>=self.database_num:
                img = self.read_fn[self.q_modality](self.images_paths[index])

            elif index<self.database_num:
                img = self.read_fn[self.db_modality](self.images_paths[index])
            
            if self.crop_during_vpr_test:
                img = F.resize(img, size=(210,560))

            # print("Image shape:", img.shape)

            return img, index

        else:
            db_img = self.read_fn[self.db_modality](self.images_paths[index])
            q_img = self.read_fn[self.q_modality](self.images_paths[self.database_num+index])
            if self.crop_images:
                if self.rescale_during_crop:
                    if random.random() < 0.5:
                        # Random resized crop (augmentation)
                        db_img, q_img = self.crop_and_resize(db_img, q_img)
                    else:
                        # Realistic crop (no resize)
                        db_img, q_img = self.crop_pair_fixed_size(db_img, q_img)
                else:
                    db_img, q_img = self.crop_pair_fixed_size(db_img, q_img)

            if self.augment:
                # Apply augmentations if training mode and augmentations are enabled
                db_img,q_img = self.augment_function(self.db_modality, self.q_modality, db_img, q_img)
            return {self.db_modality:db_img,self.q_modality:q_img}, index

if __name__ == "__main__":
    args = None
    dataset = Thermal_day_night_MS2()
    print(dataset[0][0].shape)
    