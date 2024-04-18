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

from utilities import CustomDataset
from custom_datasets.ms2_utils import *

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mixVPR_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    T.Resize((320,320))
])


class Thermal_day_night_MS2(CustomDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,datasets_folder='/ocean/projects/cis220039p/shared/datasets/MS2_full',dataset_name="sync_data",split="train",use_ang_positives=False,dist_thresh = 10,ang_thresh=20,use_mixVPR=False,use_SAM=False):
        super().__init__()

        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.split = split
        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM
        self.seq = "_2021-08-06-10-59-33"
        # self.seq = "_2021-08-06-11-23-45"
        # self.seq = "_2021-08-06-11-23-45"
        print("seq: ",self.seq)
        # self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"rgb/img_left")))[::20]
        self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thr/img_left")))[::20]
        # self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"proj_depth/",self.seq,"rgb", "depth_filtered")))[::20]
        # self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"rgb/img_left")))[::20]
        # self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thr/img_left")))[::20]
        self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"proj_depth/",self.seq,"rgb", "depth_filtered")))[::20]
        self.db_abs_paths = []
        self.q_abs_paths = []

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thr/img_left",p))
            # self.db_abs_paths.append(os.path.join(self.datasets_folder,"proj_depth/",self.seq,"rgb", "depth_filtered",p))
            # self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"rgb/img_left",p))
        for q in self.q_paths:
            # self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"rgb/img_left",q))
            self.q_abs_paths.append(os.path.join(self.datasets_folder,"proj_depth/",self.seq,"rgb", "depth_filtered",q))
            # self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thr/img_left",q))
        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.form_gt_positives()

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """

        # load files for the coordinates
        self.db_coord_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"odom",self.seq,"thr")))[::20]
        self.q_coord_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"odom",self.seq,"thr")))[::20]
        self.db_coords = []
        self.q_coords = []

        for p in self.db_coord_paths:          
            with open(os.path.join(self.datasets_folder,"odom",self.seq,"thr",p)) as f:
                lines = f.readline()
                elements = lines.split()

                matrix = []
                for i in range(0, len(elements), 4):
                    row = [float(elements[i]), float(elements[i+1]), float(elements[i+2]), float(elements[i+3])]
                    matrix.append(row)
                matrix = np.asarray(matrix)

                coord_x = float(matrix[0,3])
                coord_y = float(matrix[1,3])
                coord_z = float(matrix[2,3])

                self.db_coords.append([coord_x,coord_y,coord_z])

        for q in self.q_coord_paths:
            with open(os.path.join(self.datasets_folder,"odom",self.seq,"thr",q)) as f:
                lines = f.readline()
                elements = lines.split()

                matrix = []
                for i in range(0, len(elements), 4):
                    row = [float(elements[i]), float(elements[i+1]), float(elements[i+2]), float(elements[i+3])]
                    matrix.append(row)
                matrix = np.asarray(matrix)

                coord_x = float(matrix[0,3])
                coord_y = float(matrix[1,3])
                coord_z = float(matrix[2,3])

                self.q_coords.append([coord_x,coord_y,coord_z])

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        self.dist,self.soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= 25,
                                                            return_distance=True)            

        # for i in range(len(self.soft_positives_per_query)):
        #     print("query: ",self.soft_positives_per_query[i],self.q_paths[i])
            # for j in range(len(self.soft_positives_per_query[i])):
            #     print("database: ",self.db_paths[self.soft_positives_per_query[i][j]],self.dist[i][j])

    def __getitem__(self, index):

        if index>=self.database_num:
            # # img = cv2.imread(self.images_paths[index])
            # # img = base_transform(img)
            # img = cv2.imread(self.images_paths[index])
            # h  = img.shape[0]
            # w = img.shape[1]
            # img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            # img = base_transform(img)

            # thermal_image_path1 = self.images_paths[index]
            # img = load_as_float_img(thermal_image_path1)
            # img = process_one_image(img,type="hist_99")
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # h = img.shape[0]
            # w = img.shape[1]
            # img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            # img = base_transform(img)

            lidar_image_path1 = self.images_paths[index]
            img = cv2.imread(lidar_image_path1)
            h = img.shape[0]
            w = img.shape[1]
            img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            img = base_transform(img)

        elif index<self.database_num:
            thermal_image_path1 = self.images_paths[index]
            img = load_as_float_img(thermal_image_path1)
            img = process_one_image(img,type="hist_99")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h = img.shape[0]
            w = img.shape[1]
            img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            img = base_transform(img)

            # img = cv2.imread(self.images_paths[index])
            # h  = img.shape[0]
            # w = img.shape[1]
            # img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            # img = base_transform(img)

            # lidar_image_path1 = self.images_paths[index]
            # img = cv2.imread(lidar_image_path1)
            # h = img.shape[0]
            # w = img.shape[1]
            # img = cv2.resize(img, ((w//14)*14, (h//14)*14))
            # img = base_transform(img)

        return img, index

if __name__ == "__main__":
    args = None
    dataset = Thermal_day_night_MS2(args,split="train",use_mixVPR=False)
    print(dataset[0][0].shape)
    