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

class CartDataloader(CustomDataset):
    def __init__(self,args,seq,db_modality,q_modality,datasets_folder='/storage2/datasets/jkarhade',dataset_name="CART_place_recognition",use_ang_positives=False,dist_thresh = 10,ang_thresh=20):
        # super().__init()

        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.db_modality = db_modality
        self.q_modality = q_modality
        self.seq = seq #"Idyll_wild"
        self.subsample = 1

        self.model_type = "clip"

        print("seq: ",self.seq)
        print("db_modality: ",self.db_modality)
        print("q_modality: ",self.q_modality)

        if self.db_modality == "rgb":
            self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"color")))[::self.subsample]
        elif self.db_modality == "thr":
            self.db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thermal")))[::self.subsample]

        if self.q_modality == "rgb":
            self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"color")))[::self.subsample]
        elif self.q_modality == "thr":
            self.q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thermal")))[::self.subsample]

        self.db_abs_paths = []
        self.q_abs_paths = []

        for db_path in self.db_paths:
            if self.db_modality == "rgb":
                self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"color",db_path))
            elif self.db_modality == "thr":
                self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thermal",db_path))

        for q_path in self.q_paths:
            if self.q_modality == "rgb":
                self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"color",q_path))
            elif self.q_modality == "thr":
                self.q_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"thermal",q_path))

        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.images_paths = self.db_abs_paths + self.q_abs_paths

        self.gt_positives = np.load(os.path.join(self.datasets_folder,self.dataset_name,self.seq,"soft_positives_per_query.npy"),allow_pickle=True)
        self.soft_positives_per_query = []

        for i in range(len(self.gt_positives)):
            self.soft_positives_per_query.append(self.gt_positives[i])

    def __getitem__(self,index):

        if index>=self.database_num:
            if self.q_modality == "thr":
                # img = load_as_float_img(self.images_paths[index])
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

            elif self.q_modality == "rgb":
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

        elif index < self.database_num:
            if self.db_modality == "thr":
                # img = load_as_float_img(self.images_paths[index])
                # import pdb;pdb.set_trace()
                img = cv2.imread(self.images_paths[index])                
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

            elif self.db_modality == "rgb":
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                    # cv2.imwrite("test.png",img)
                    # import pdb;pdb.set_trace()
                img = base_transform(img)

        return img, index

if __name__ == "__main__":

    args = None
    dataset = CartDataloader(args,db_modality="rgb",q_modality="thr",seq="Idyll_wild")

    # cv2.imwrite("test.png",dataset[0+dataset.db_num][0])
