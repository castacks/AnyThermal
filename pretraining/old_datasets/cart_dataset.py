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
])

class CART_dataset(Dataset):
    def __init__(self, data_dir,model_type="dinov2"):
        self.data_dir = data_dir

        self.rgb_image_paths = []
        self.thermal_image_paths = []

        self.model_type = model_type

        # Convention: For now, left images only for training
        cur_seq_rgb_image_paths = natsorted(os.listdir(os.path.join(self.data_dir,"color")))
        cur_seq_thermal_image_paths = natsorted(os.listdir(os.path.join(self.data_dir,"thermal8")))

        for i in range(len(cur_seq_rgb_image_paths)):

            if 'big_bear' in cur_seq_rgb_image_paths[i] or 'caltech_duck' in cur_seq_rgb_image_paths[i] or 'caltech-coregistered-nature-dataset_ONR_2022-04-03-12-20-57' in cur_seq_rgb_image_paths[i] or 'caltech-coregistered-nature-dataset_ONR_2022-04-03-12-16-33' in cur_seq_rgb_image_paths[i]:
                continue

            self.rgb_image_paths.append(os.path.join(self.data_dir,"color", cur_seq_rgb_image_paths[i]))
            self.thermal_image_paths.append(os.path.join(self.data_dir,"thermal8", cur_seq_thermal_image_paths[i]))

        print(len(self.rgb_image_paths), len(self.thermal_image_paths))

        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        idx1 = idx

        thermal_image_path1 = self.thermal_image_paths[idx1]
        # thermal_image1 = load_as_float_img(thermal_image_path1)
        thermal_image1 = cv2.imread(thermal_image_path1)
        h = thermal_image1.shape[0]
        w = thermal_image1.shape[1]
        if self.model_type=="clip":
            thermal_image1 = cv2.resize(thermal_image1, (224, 224))
        else:
            thermal_image1 = cv2.resize(thermal_image1, ((w//14)*14, (h//14)*14))
        thermal_image1 = base_transform(thermal_image1)

        rgb_image_path1 = self.rgb_image_paths[idx1]
        rgb_image1 = cv2.imread(rgb_image_path1)
        h  = rgb_image1.shape[0]
        w = rgb_image1.shape[1]
        if self.model_type=="clip":
            rgb_image1 = cv2.resize(rgb_image1, (224, 224))
        else:
            rgb_image1 = cv2.resize(rgb_image1, ((w//14)*14, (h//14)*14))
        rgb_image1 = base_transform(rgb_image1)

        images_dict = {"rgb1": rgb_image1, "thermal1": thermal_image1}
        # print(rgb_image1.shape, thermal_image1.shape)
        return images_dict

if __name__ == "__main__":
    dataset = CART_dataset("/storage2/datasets/CART_dataset/labeled_rgbt_pairs/")

    dict1 = dataset[0]
    dict2 = dataset[100]
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
