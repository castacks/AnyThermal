import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from natsort import natsorted
from datasets.ms2_utils import enhance_image, hist_99, align_contrast, load_as_float_img, process_one_image

base_transform = T.Compose([
    T.ToTensor(),
    # tvf.CenterCrop([(h//14)*14, (w//14)*14]),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

thermal_base_transform = T.Compose([
    T.ToTensor(),
    # tvf.CenterCrop([(h//14)*14, (w//14)*14]),
    T.Normalize(mean=[0.45], std=[0.225]),
    ])

class Custom_MS2Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Convention: For now, left images only for training
        self.rgb_image_paths = natsorted(os.listdir(os.path.join(self.data_dir,"sync_data/_2021-08-06-10-59-33", "rgb", "img_left")))
        self.thermal_image_paths = natsorted(os.listdir(os.path.join(self.data_dir,"sync_data/_2021-08-06-10-59-33", "thr", "img_left")))
        self.lidar_image_paths = natsorted(os.listdir(os.path.join(self.data_dir,"proj_depth/_2021-08-06-10-59-33", "rgb", "depth_filtered")))

        for i in range(len(self.rgb_image_paths)):
            self.rgb_image_paths[i] = os.path.join(self.data_dir,"sync_data/_2021-08-06-10-59-33", "rgb", "img_left", self.rgb_image_paths[i])
            self.thermal_image_paths[i] = os.path.join(self.data_dir,"sync_data/_2021-08-06-10-59-33", "thr", "img_left", self.thermal_image_paths[i])
            self.lidar_image_paths[i] = os.path.join(self.data_dir,"proj_depth/_2021-08-06-10-59-33","rgb", "depth_filtered", self.lidar_image_paths[i])

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        
        idx1 = idx
        # idx2 = np.random.randint(0, len(self.rgb_image_paths))

        thermal_image_path1 = self.thermal_image_paths[idx1]
        thermal_image1 = load_as_float_img(thermal_image_path1)
        thermal_image1 = process_one_image(thermal_image1,type="hist_99")
        thermal_image1 = cv2.cvtColor(thermal_image1, cv2.COLOR_GRAY2RGB)
        h = thermal_image1.shape[0]
        w = thermal_image1.shape[1]
        # print("thermal", h, w)
        thermal_image1 = cv2.resize(thermal_image1, ((w//14)*14, (h//14)*14))
        thermal_image1 = base_transform(thermal_image1)

        rgb_image_path1 = self.rgb_image_paths[idx1]
        rgb_image1 = cv2.imread(rgb_image_path1)
        h  = rgb_image1.shape[0]
        w = rgb_image1.shape[1]
        # print("rgb", h, w)
        rgb_image1 = cv2.resize(rgb_image1, ((w//14)*14, (h//14)*14))
        rgb_image1 = base_transform(rgb_image1)

        lidar_image_path1 = self.lidar_image_paths[idx1]
        lidar1 = cv2.imread(lidar_image_path1)
        h = lidar1.shape[0]
        w = lidar1.shape[1]
        # print("lidar", h, w)
        lidar1 = cv2.resize(lidar1, ((w//14)*14, (h//14)*14))
        lidar1 = base_transform(lidar1)

        # print(rgb_image1.shape, thermal_image1.shape, lidar1.shape)
        images_dict = {"rgb1": rgb_image1, "thermal1": thermal_image1,"lidar1": lidar1}

        return images_dict

if __name__ == "__main__":
    dataset = Custom_MS2Dataset("/ocean/projects/cis220039p/shared/datasets/MS2_full/")
    thermal_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    thermal_model.eval()

    x = dataset[0]["thermal1"].unsqueeze(0)
    y = thermal_model(x)
    z = thermal_model(x)