import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import random

from natsort import natsorted
from datasets.ms2_utils import enhance_image, hist_99, align_contrast, load_as_float_img, process_one_image
# from ms2_utils import enhance_image, hist_99, align_contrast, load_as_float_img, process_one_image

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
])

def sparse_to_dense(sparse, max_depth=100.):
    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    ## dilate
    custom_kernel = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
    sparse = cv2.dilate(sparse, custom_kernel)

    ## close
    custom_kernel = np.ones((5, 5), np.uint8)
    sparse = cv2.morphologyEx(sparse, cv2.MORPH_CLOSE, custom_kernel)

    ## fill
    invalid = sparse < 0.1
    custom_kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(sparse, custom_kernel)
    sparse[invalid] = dilated[invalid]

    ## invert
    valid = sparse > 0.1
    sparse[valid] = max_depth - sparse[valid]

    return sparse

class Wisard_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data_dir_list = os.listdir(self.data_dir)

        self.thermal_list = []
        self.rgb_list = []

        for seq in self.data_dir_list:
            if "IR" in seq:
                self.thermal_list.append(seq)
            elif "VIS" in seq:
                self.rgb_list.append(seq)

        self.thermal_list = natsorted(self.thermal_list)
        self.rgb_list = natsorted(self.rgb_list)

        self.thermal_image_paths = []
        self.rgb_image_paths = []

        for thermal_seq, rgb_seq in zip(self.thermal_list, self.rgb_list):

            thermal_seq_list = natsorted(os.listdir(os.path.join(self.data_dir, thermal_seq)))
            rgb_seq_list = natsorted(os.listdir(os.path.join(self.data_dir, rgb_seq)))

            for thermal_img in thermal_seq_list:
                if "jpeg" and "jpg" not in thermal_img:
                    continue
                self.thermal_image_paths.append(os.path.join(self.data_dir, thermal_seq, thermal_img))

            for rgb_img in rgb_seq_list:
                if "jpeg" and "jpg" not in rgb_img:
                    continue
                self.rgb_image_paths.append(os.path.join(self.data_dir, rgb_seq, rgb_img))

        # Shuffle and subsample to avoid redundant samples
        # random.shuffle(self.thermal_image_paths)
        # random.shuffle(self.rgb_image_paths)
        # self.thermal_image_paths = self.thermal_image_paths[::2]
        # self.rgb_image_paths = self.rgb_image_paths[::2]

        print(len(self.thermal_image_paths), len(self.rgb_image_paths))

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        
        idx1 = idx

        thermal_image_path1 = self.thermal_image_paths[idx1]
        thermal_image1 = load_as_float_img(thermal_image_path1)
        h = thermal_image1.shape[0]
        w = thermal_image1.shape[1]
        thermal_image1 = cv2.resize(thermal_image1, ((w//14)*14, (h//14)*14))
        thermal_image1 = base_transform(thermal_image1)

        rgb_image_path1 = self.rgb_image_paths[idx1]
        rgb_image1 = cv2.imread(rgb_image_path1)
        h  = rgb_image1.shape[0]
        w = rgb_image1.shape[1]
        rgb_image1 = cv2.resize(rgb_image1, ((w//14)*14, (h//14)*14))
        rgb_image1 = base_transform(rgb_image1)

        images_dict = {"rgb1": rgb_image1, "thermal1": thermal_image1}

        return images_dict

if __name__ == "__main__":
    dataset = Wisard_Dataset("/storage2/datasets/thermal_wisard/")

    dict1 = dataset[0]
    dict2 = dataset[0]
    rgb_image1 = dict1['rgb1']
    thermal_image1 = dict1['thermal1']

    rgb_image2 = dict2['rgb1']
    thermal_image2 = dict2['thermal1']

    # Save to file
    cv2.imwrite("rgb1.png", rgb_image1.permute(1, 2, 0).numpy()*255)
    cv2.imwrite("thermal1.png", thermal_image1.permute(1, 2, 0).numpy())

    # Save to file
    cv2.imwrite("rgb2.png", rgb_image1.permute(1, 2, 0).numpy()*255)
    cv2.imwrite("thermal2.png", thermal_image1.permute(1, 2, 0).numpy())
