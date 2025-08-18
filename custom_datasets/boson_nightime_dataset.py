import os
import torch
import faiss
import faiss.contrib.torch_utils
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disables DecompressionBombError
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as F
import h5py
import time
import random

from base_dataset import *

class BosonNightimeBaseDataset(BaseDataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self,args,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 35, rescale_during_crop=False,crop_during_vpr_test=False,dataset_name="satellite_0_thermalmapping_135_train",G_contrast="none",force_ce=False,val_positive_dist_threshold=50, neg_ring_outer_radius = 100):
        self.dataset_name = dataset_name
        self.extended = "extended" in seq #PARV_TODO : how to use the extended data confirm in STHN
        self.resize = [512, 512]
        self.resize_smaller = 224
        self.G_contrast = G_contrast
        self.force_ce = force_ce
        self.val_positive_dist_threshold = val_positive_dist_threshold
        self.neg_ring_outer_radius = neg_ring_outer_radius
        self.location_type = 'gps'

        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop,crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)

        for i in range(len(self.db_abs_paths)):
            self.db_abs_paths[i] = "database_" + self.db_abs_paths[i]
        for i in range(len(self.q_abs_paths)):
            self.q_abs_paths[i] = "queries_" + self.q_abs_paths[i]
        
        self.images_paths = list(self.db_abs_paths) + \
            list(self.q_abs_paths)

        # Close h5 and initialize for h5 reading in __getitem__
        self.database_folder_h5_df = None
        self.queries_folder_h5_df = None

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                base_transform
            ]
        )
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        # Redirect datafolder path to h5
        split = self.seq[0]

        self.database_folder_h5_path = join(
            self.datasets_folder, self.dataset_name, split + "_database.h5"
        )
        self.queries_folder_h5_path = join(
            self.datasets_folder, self.dataset_name, split + "_queries.h5"
        )

        self.database_folder_map_path = join(
            self.datasets_folder, "maps/satellite/20201117_BingSatellite.png"
        )

        database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r", swmr=True)
        self.database_folder_map_df = F.to_tensor(Image.open(self.database_folder_map_path).convert("RGB"))
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r", swmr=True)

        # Map name to index
        self.database_name_dict = {}
        self.queries_name_dict = {}

        # Duplicated elements are added
        for index, database_image_name in enumerate(database_folder_h5_df["image_name"]):
            database_image_name_decoded = database_image_name.decode("UTF-8")
            while database_image_name_decoded in self.database_name_dict:
                northing = [str(float(database_image_name_decoded.split("@")[2])+0.00001)]
                database_image_name_decoded = "@".join(list(database_image_name_decoded.split("@")[:2]) + northing + list(database_image_name_decoded.split("@")[3:]))
            self.database_name_dict[database_image_name_decoded] = index
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            queries_image_name_decoded = queries_image_name.decode("UTF-8")
            while queries_image_name_decoded in self.queries_name_dict:
                northing = [str(float(queries_image_name_decoded.split("@")[2])+0.00001)]
                queries_image_name_decoded = "@".join(list(queries_image_name_decoded.split("@")[:2]) + northing + list(queries_image_name_decoded.split("@")[3:]))
            self.queries_name_dict[queries_image_name_decoded] = index

        # Read paths and UTM coordinates for all images.
        # database_folder = join(self.dataset_folder, "database")
        # queries_folder  = join(self.dataset_folder, "queries")
        # if not os.path.exists(database_folder): raise FileNotFoundError(f"Folder {database_folder} does not exist")
        # if not os.path.exists(queries_folder) : raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        db_abs_paths = sorted(self.database_name_dict)
        q_abs_paths = sorted(self.queries_name_dict)

        db_abs_paths = db_abs_paths
        q_abs_paths = q_abs_paths

        print("#images in database: ", len(db_abs_paths), "from ", self.database_folder_h5_path)
        print("#images in queries: ", len(q_abs_paths), "from ", self.queries_folder_h5_path)
        
        
        # import pdb;pdb.set_trace()

        database_folder_h5_df.close()
        queries_folder_h5_df.close()

        return db_abs_paths,q_abs_paths

    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }
    
    def read_rgb(self, path):
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r", swmr=True)
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r", swmr=True)
        # print(f"Reading RGB image from {path}")
        image_name ="_".join(path.split("_")[1:])
        # database_queries_split = path.split("_")[0]
        # assert database_queries_split == "database"
        # import pdb;pdb.set_trace()
        # img = Image.fromarray(
        #         self.database_folder_h5_df["image_data"][
        #             self.database_name_dict[image_name]
        #         ]
        #     )
        if self.extended:
            img = self.extended_database_folder_map_df
        else:
            img = self.database_folder_map_df
        center_cood = (float(image_name.split("@")[1]), float(image_name.split("@")[2]))
        area = (int(center_cood[1]) - self.resize[1]//2, int(center_cood[0]) - self.resize[0]//2,
                int(center_cood[1]) + self.resize[1]//2, int(center_cood[0]) + self.resize[0]//2)
        img = F.crop(img=img, top=area[1], left=area[0], height=area[3]-area[1], width=area[2]-area[0])
        img = F.resize(img, [self.resize_smaller,self.resize_smaller], interpolation=F.InterpolationMode.BILINEAR,antialias=True)
        return img
    
    def read_thermal(self, path):
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r", swmr=True)
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r", swmr=True)
        # print(f"Reading thermal image from {path}")
        image_name = "_".join(path.split("_")[1:])
        database_queries_split = path.split("_")[0]
        assert database_queries_split == "queries"
        img = Image.fromarray(
                self.queries_folder_h5_df["image_data"][
                    self.queries_name_dict[image_name]
                ]
            )
        #center crop of size self.resize
        # print(f"Image shape before center crop: {img.size}")
        img = F.center_crop(img, self.resize)
        # print(f"Image shape after center crop: {img.size}")
        if self.G_contrast!="none" and (self.force_ce or self.split!="extended"):
            if self.G_contrast == "manual":
                img = self.query_transform(
                    transforms.functional.adjust_contrast(img, contrast_factor=3))
            elif self.G_contrast == "autocontrast":
                img = self.query_transform(
                    transforms.functional.autocontrast(img))
            elif self.G_contrast == "equalize":
                img = self.query_transform(
                    transforms.functional.equalize(img))
            else:
                raise NotImplementedError()
        else:
            img = self.query_transform(img)

        img = transforms.functional.resize(img, self.resize)
        img = F.resize(img, [self.resize_smaller,self.resize_smaller], interpolation=F.InterpolationMode.BILINEAR,antialias=True)

        return img

    def get_positives(self):
        return self.hard_positives_per_query

    def get_hard_negatives(self):
        return self.hard_negatives_per_query

    def __del__(self):
        if (
            hasattr(self, "database_folder_h5_df")
            and self.database_folder_h5_df is not None
        ):
            self.database_folder_h5_df.close()
            self.queries_folder_h5_df.close()
        
    def find_black_region(self):
        # Only for thermal
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r", swmr=True)
        queries_with_black_region = []
        for index, path in enumerate(self.queries_paths):
            real_path = path[len("queries_"):]
            image_index = self.queries_name_dict[real_path]
            image = queries_folder_h5_df["image_data"][image_index]
            if np.count_nonzero(image==0) > 50:
                queries_with_black_region.append(index)
        queries_folder_h5_df.close()
        return queries_with_black_region
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}
    
    def form_db_qu_coords(self):
        self.db_coords = []
        self.q_coords = []

        self.db_coords = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.db_abs_paths]
        ).astype(float).tolist()
        self.q_coords = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.q_abs_paths]
        ).astype(float).tolist()

    
    def check_seq_list(self,seq):
        assert isinstance(seq, list), "seq should be a list of sequences"
        assert len(seq) == 1, "seq should contain only one sequence for BosonNightimeBaseDataset"

        split = seq[0]

        self.database_folder_h5_path = join(
            self.datasets_folder, self.dataset_name, split + "_database.h5"
        )
        self.queries_folder_h5_path = join(
            self.datasets_folder, self.dataset_name, split + "_queries.h5"
        )

        self.database_folder_map_path = join(
            self.datasets_folder, "maps/satellite/20201117_BingSatellite.png"
        )

        if not os.path.exists(self.database_folder_h5_path):
            raise FileNotFoundError(f"Folder {self.database_folder_h5_path} does not exist")
        if not os.path.exists(self.queries_folder_h5_path):
            raise FileNotFoundError(f"Folder {self.queries_folder_h5_path} does not exist")
        if not os.path.exists(self.database_folder_map_path):
            raise FileNotFoundError(f"Folder {self.database_folder_map_path} does not exist")

