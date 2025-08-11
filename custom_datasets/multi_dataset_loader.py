from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader, Subset, RandomSampler, BatchSampler
from collections import defaultdict
from ms2_dataset import MS2, return_ms2_split, return_ms2_split_debug
from cart_dataset import CART, HandheldCART,return_cart_split_segmentation_geographic,return_cart_split, return_cart_split_debug,return_handheld_cart_split
from freiburg_dataset import Freiburg, return_freiburg_split
from vivid_dataset import Vivid, return_vivid_split
from sthereo_dataset import STHEREO, return_sthereo_split
from boson_nightime_dataset import BosonNightimeBaseDataset
from m2p2_dataset import M2P2, return_m2p2_split
from mfnet_dataset import MFNet, return_mfnet_split
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import os
import math

import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
import faiss
from torchvision.transforms.functional import to_pil_image
from joblib import Parallel, delayed
import multiprocessing as mp

from functools import partial

from torchvision import transforms
import time
class IdentityTransform:
    def __call__(self, x):
        return x

def apply_clahe(img):
    """Apply CLAHE to a single-channel tensor image (1, H, W)"""
    img_np = img[0].cpu().numpy() #taking the first channel
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_eq = self.clahe.apply(img_uint8)
    img_eq = torch.tensor(img_eq, dtype=torch.float32, device=img.device) / 255.0
    img_eq = img_eq.repeat(3,1,1)
    return img_eq


class RAMEfficient2DMatrixGPU:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=torch.float32, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.matrix = [None] * shape[0]

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.type(self.dtype).to(self.device)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return torch.stack([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class ThermalAugmentations:
    def __init__(self,
                 crop_scale=(0.6, 1.0),
                 brightness=0.2,
                 contrast=0.2,
                 rotation=15,
                 flip_prob=0.5,
                 blur_prob=0.2,
                 blur_kernel=3,
                 clahe_prob=0.2,
                 enabled_transforms=None):
        self.crop_scale = crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.rotation = rotation
        self.flip_prob = flip_prob
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel
        self.clahe_prob = clahe_prob

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        all_transforms = {"crop", "flip", "rotate", "brightness", "contrast", "clahe", "blur"}
        if enabled_transforms is None:
            self.enabled = all_transforms
        else:
            self.enabled = set(enabled_transforms) & all_transforms

    def random_resized_crop_preserve_shape(self, img, scale):
        """Crop a region of the image and resize back to original shape"""
        C, H, W = img.shape
        target_area = random.uniform(*scale) * H * W
        aspect_ratio = W / H

        new_h = int(round((target_area / aspect_ratio) ** 0.5))
        new_w = int(round(new_h * aspect_ratio))
        new_h = min(new_h, H)
        new_w = min(new_w, W)

        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)

        return TF.resized_crop(img, top, left, new_h, new_w, size=(H, W),
                               interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

    def __call__(self, img):

        """Apply augmentations to a single image (C, H, W)"""
        if "crop" in self.enabled:
            img = self.random_resized_crop_preserve_shape(img, self.crop_scale)

        if "flip" in self.enabled and random.random() < self.flip_prob:
            img = TF.hflip(img)

        if "rotate" in self.enabled and self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            fill_val = img.mean().item()
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=fill_val, antialias=True)

        if "brightness" in self.enabled and self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = img * factor

        if "contrast" in self.enabled and self.contrast > 0:
            mean = img.mean()
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = (img - mean) * factor + mean

        if "clahe" in self.enabled and random.random() < self.clahe_prob:
            img = apply_clahe(img)

        if "blur" in self.enabled and random.random() < self.blur_prob:
            img = TF.gaussian_blur(img, kernel_size=self.blur_kernel)

        img = torch.clamp(img, 0.0, 1.0)

        return img


class IntraDatasetBatchSampler:
    def __init__(self, dataset_to_indices, batch_size, shuffle=True,subsample=1, equal_samples=False):
        if isinstance(dataset_to_indices, dict):
            self.dataset_to_indices = dataset_to_indices
        elif isinstance(dataset_to_indices, np.ndarray):
            self.dataset_to_indices = defaultdict(list)
            for idx, d_idx in enumerate(dataset_to_indices):
                self.dataset_to_indices[d_idx].append(idx)
        elif isinstance(dataset_to_indices, list):
            assert isinstance(dataset_to_indices[0], tuple) and len(dataset_to_indices[0]) == 2, \
                "dataset_to_indices should be a list of tuples (dataset_id, index)"
            self.dataset_to_indices = defaultdict(list)
            for idx, (d_idx, _) in enumerate(dataset_to_indices):
                self.dataset_to_indices[d_idx].append(idx)
        else:
            raise TypeError("dataset_to_indices should be a dict, np.ndarray, or list of tuples.")

        print("dataset_to_indices keys:", self.dataset_to_indices.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subsample = subsample
        self.equal_samples = equal_samples

    def __iter__(self):
        
        return iter(self.calculate_batches())

    def calculate_batches(self):
        dataset_keys = sorted(self.dataset_to_indices.keys())

        all_batches = []
        all_batch_list = []

        for d_idx in dataset_keys:
            indices = self.dataset_to_indices[d_idx][:]
            if self.shuffle:
                random.shuffle(indices)
            else:
                indices.sort()  # Deterministic order within dataset

            # Split indices into batches
            temp_batch_list = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                temp_batch_list.append(batch)
            
            all_batch_list.append(temp_batch_list)
        
        if self.equal_samples:
            min_num_batches = min(len(batch_list) for batch_list in all_batch_list)
        
        for i in range(len(all_batch_list)):
            if self.equal_samples:
                random_idx = np.random.choice(len(all_batch_list[i]), size=min_num_batches, replace=False)
            else:
                random_idx = np.arange(len(all_batch_list[i]))
            sampled_batches_size = len(random_idx)

            all_batches.extend([all_batch_list[i][j] for j in random_idx])
            print("Dataset", i, "has", len(all_batch_list[i]), "batches, sampled", sampled_batches_size, "batches", "all_batches length:", len(all_batches))
        
        
        if self.shuffle:
            random.shuffle(all_batches)  # Shuffle batches across datasets
        # Subsample batches if needed
        if self.subsample > 1:
            all_batches = all_batches[::self.subsample]
        
        return all_batches

    def __len__(self):
        """Total number of batches across all datasets"""
        output = sum(
            math.ceil(len(idxs) / self.batch_size)
            for idxs in self.dataset_to_indices.values()
        )
        if self.subsample > 1:
            output = math.ceil(output / self.subsample)
        return len(self.calculate_batches())

class MultiDatasetWrapper(Dataset):
    def __init__(self, args,datasets, dataset_names, mode, use_odom=False, dist_thresh=25,build_common_dataset = False):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.mapping = []
        self.mode = mode
        self.use_odom = use_odom
        self.args = args
        for d_idx, d in enumerate(datasets):
            self.mapping.extend([(d_idx, i) for i in range(len(d))])
        self.idx_to_dataset = np.array([d_idx for d_idx, _ in self.mapping])
        
        if self.use_odom:
            if hasattr(datasets[0], 'soft_positives_per_query'):
                print("building combined soft_positives for validation set")
                self.soft_positives = []
                self.db_coords = []
                self.q_coords = []
                global_index = 0
                for d_idx, d in enumerate(datasets):
                    if hasattr(d, 'soft_positives_per_query'):
                        self.soft_positives.extend([[global_index+i for i in d.soft_positives_per_query[idx]] for idx in range(len(d.soft_positives_per_query))])
                        assert d.database_num == len(d.soft_positives_per_query), "Soft positives length mismatch"
                        global_index += len(d.soft_positives_per_query)
                    else:
                        raise ValueError(f"Dataset {dataset_names[d_idx]} does not have soft_positives_per_query attribute")

                    self.db_coords.extend(d.db_coords)
                    self.q_coords.extend(d.q_coords)

                for i in range(len(self.db_coords)):
                    if self.db_coords[i] is not None:
                        self.db_coords[i] = self.db_coords[i][:2]
                        self.q_coords[i] = self.q_coords[i][:2]

                # self.db_coords = np.array(self.db_coords)
                # self.q_coords = np.array(self.q_coords)

                # if build_common_dataset:
                #     print("Finding common soft positives in the database and queries")

                #     knn = NearestNeighbors(n_jobs=-1)
                #     knn.fit(self.db_coords)
                #     _, self.common_soft_positives = knn.radius_neighbors(self.q_coords, radius=dist_thresh, return_distance=True)
                #     assert len(self.common_soft_positives) == len(self.soft_positives), "Mismatch between soft positives"
                #     # IMP self.common_soft_positives can be differnt from self.soft_positives as it only compares 2d coordinates whereas self.soft_positives contains the indices of the soft positives in the database which can compare 3d also if the data is available.                
                #     print("Found common soft positives in the database and queries")
            else:
                raise ValueError("Soft positives not available. Check dataset classes.")

            self.extra_margin_soft_positives = self.knn_neighbours("soft_positives_per_query", n_jobs=-1)

        if hasattr(args, 'student_modality_dual') and args.student_modality_dual:
            self.thermal_augmentations = ThermalAugmentations(enabled_transforms=args.thermal_aug_list)
        else:
            self.thermal_augmentations = None
        print("self.thermal_augmentations:", self.thermal_augmentations)

        self.db_dataset, self.qu_dataset = self.concat_datasets_separately()

        self.db_mapping =[]
        self.q_mapping = []

        db_len=0
        q_len=0

        for d_idx, d in enumerate(self.datasets):
            self.db_mapping.extend([i+db_len+q_len for i in range(d.database_num)])
            db_len += d.database_num
            self.q_mapping.extend([i+db_len+q_len for i in range(d.queries_num)])
            q_len += d.queries_num
        
        self.db_mapping = np.array(self.db_mapping, dtype=np.int32)
        self.q_mapping = np.array(self.q_mapping, dtype=np.int32)

        self.queries_num = len(self.q_mapping)
        self.database_num = len(self.db_mapping)
        if self.use_odom:
            assert self.database_num == len(self.db_coords), "Database dataset length does not match coordinates length"
            assert self.queries_num == len(self.q_coords), "Query dataset length does not match coordinates length"
    
    def knn_neighbours(self, og_radius, n_jobs=-1):
        final_output_list = []
        running_total_datase_len = 0
        for d_idx, d in enumerate(self.datasets):
            if og_radius == 'hard_positives_per_query':
                radius = d.dist_thresh
            elif og_radius == 'soft_positives_per_query':
                if hasattr(d, 'val_positive_dist_threshold') and d.val_positive_dist_threshold >0:
                    radius = d.val_positive_dist_threshold
                else:
                    radius = None
            elif og_radius == 'hard_negatives_per_query':
                radius = d.prior_location_threshold
            else:
                raise ValueError(f"Unknown radius type: {og_radius}. Use 'hard_positives_per_query', 'soft_positives_per_query' or 'hard_negatives_per_query'.")
            if hasattr(d, 'db_coords') and radius is not None:
                knn = NearestNeighbors(n_jobs=n_jobs)
                knn.fit(d.db_coords)

                neighbours = knn.radius_neighbors(
                    d.q_coords, radius=radius, return_distance=False
                )
                neighbours = np.array(neighbours, dtype=object) + running_total_datase_len
                final_output_list.append(neighbours)
                running_total_datase_len += len(d.db_coords)
            elif og_radius == 'soft_positives_per_query' and hasattr(d,'val_extra_margin_positive_radius_index') and d.val_extra_margin_positive_radius_index is not None:
                temp_neighbours = d.val_extra_margin_positives_per_query
                temp_neighbours = np.array(temp_neighbours, dtype=object) + running_total_datase_len
                final_output_list.append(temp_neighbours)
                running_total_datase_len += len(d.val_extra_margin_positives_per_query)
            else:
                raise ValueError(f"Dataset {self.dataset_names[d_idx]} does not have db_coords attribute for radius mode '{og_radius}'. ")
        
        final_output_list = np.concatenate(final_output_list, axis=0)
        return final_output_list

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.mapping[idx]
        item = {}
        item["item"] = self.datasets[dataset_idx][local_idx]
        if self.thermal_augmentations:
            if "thr" in item["item"][0]:
                item["item"][0]["thr_dual"] = self.thermal_augmentations(item["item"][0]["thr"])
        item["dataset_name"] = self.dataset_names[dataset_idx]
        item["batch_id"] = idx
        item["dataset_id"] = dataset_idx  # NEW: used to group batch by dataset
        return item

    def concat_datasets_separately(self):
        db_dataset_list = []
        q_dataset_list = []
        db_idx_to_dataset = []
        q_idx_to_dataset = []
        running_db_id = 0
        running_q_id = 0
        for d_idx,d in enumerate(self.datasets):
            db_dataset = Subset(d, range(d.database_num))
            q_dataset = Subset(d, range(d.database_num, len(d)))
            db_dataset_list.append(db_dataset)
            q_dataset_list.append(q_dataset)
            for i in range(d.database_num):
                db_idx_to_dataset.append(d_idx)
            for i in range(d.queries_num):
                q_idx_to_dataset.append(d_idx)
            running_db_id += d.database_num
            running_q_id += d.queries_num
        concat_db,concat_q= ConcatDataset(db_dataset_list), ConcatDataset(q_dataset_list)
        concat_db.idx_to_dataset = np.array(db_idx_to_dataset)
        concat_q.idx_to_dataset = np.array(q_idx_to_dataset)
        return concat_db, concat_q


def triplet_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(
        zip(triplets_local_indexes, triplets_global_indexes)
    ):
        local_indexes += (
            len(global_indexes) * i
        )  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


import faiss
import os

# Global FAISS GPU resource for each worker process
global_faiss_res = None
global_gpu_id = None

def faiss_worker_init(use_gpu=True, gpu_id=0):
    """
    Called once per worker process to initialize FAISS GPU resources.
    """
    global global_faiss_res, global_gpu_id

    if use_gpu:
        print(f"[Worker {os.getpid()}] Initializing FAISS GPU resources on GPU {gpu_id}")
        global_gpu_id = gpu_id
        res = faiss.StandardGpuResources()
        res.setTempMemory(200 * 1024 * 1024)  # Optional: preallocate 200MB temp memory
        global_faiss_res = res
    else:
        print(f"[Worker {os.getpid()}] Using FAISS CPU")
        global_faiss_res = None


def triplet_worker_fn(random_query_idx, sampled_queries_indexes_local, args, cache, db_mapping, 
                      hard_positives_per_query, hard_negatives_per_query, soft_positives_per_query, 
                      database_indexes_local, negs_num_per_query, use_faiss_gpu, gpu_res_id, 
                      db_idx_to_dataset, qu_idx_to_dataset, global_index_to_dataset):

    global global_faiss_res, global_gpu_id

    st = time.time()

    query_index = sampled_queries_indexes_local[random_query_idx]

    idx_to_db_dict = defaultdict(list)
    for i, db_idx in enumerate(db_idx_to_dataset):
        idx_to_db_dict[db_idx].append(i)

    qu_dataset_id = qu_idx_to_dataset[random_query_idx]
    local_db_for_given_qu = idx_to_db_dict[qu_dataset_id]

    global_index = db_mapping['q_mapping'][query_index]
    query_features = cache[global_index]
    if query_features is None:
        raise RuntimeError(f"Features for query index {query_index} are None.")

    # Random positive selection
    best_positive_index = np.random.choice(hard_positives_per_query[query_index],
                                           size=1,
                                           replace=False)[0]

    # Filter negatives
    soft_positives = soft_positives_per_query[query_index]
    neg_indexes = np.setdiff1d(database_indexes_local[local_db_for_given_qu], soft_positives, assume_unique=True)

    if args.prior_location_threshold != -1:
        hard_negatives = hard_negatives_per_query[query_index]
        neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)

    neg_features = cache[db_mapping['db_mapping'][neg_indexes]]

    print("time to get query features:", time.time() - st)

    if use_faiss_gpu:
        print(f"[Worker {os.getpid()}] Using FAISS GPU {global_gpu_id} for query {query_index} with {len(neg_features)} negatives")
        faiss_index = faiss.GpuIndexFlatL2(global_faiss_res, neg_features.shape[-1])
    else:
        print(f"[Worker {os.getpid()}] Using FAISS CPU for query {query_index} with {len(neg_features)} negatives")
        faiss_index = faiss.IndexFlatL2(neg_features.shape[-1])

    faiss_index.add(neg_features)
    _, neg_nums = faiss_index.search(query_features.reshape(1, -1), negs_num_per_query)
    neg_nums = neg_nums.reshape(-1).cpu().numpy()
    final_neg_indexes = neg_indexes[neg_nums.astype(np.int32)]

    del faiss_index

    print(f"[Worker {os.getpid()}] Query {query_index} processed in {time.time() - st:.2f} seconds")

    return (query_index, best_positive_index, *final_neg_indexes), qu_dataset_id



class TripletsDataset(MultiDatasetWrapper):
    def __init__(self, args,datasets, dataset_names, mode, use_odom=False, dist_thresh=25,build_common_dataset = False):
        super().__init__(args,datasets, dataset_names, mode, use_odom, dist_thresh,build_common_dataset)
        self.args = args
        self.neg_samples_num = (
            args.neg_samples_num
        )  # Number of negatives to randomly sample
        self.negs_num_per_query = (
            args.negs_num_per_query  # Number of negatives per query in each batch
        )
        self.is_inference = False
        identity_transform = IdentityTransform()
        self.resize = args.resize if hasattr(args, 'resize') else None
        base_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    #                     0.229, 0.224, 0.225]),
                ]
            )
        self.resized_transform = transforms.Compose(
            [
                transforms.Resize(self.resize)
                if self.resize is not None
                else identity_transform,
                base_transform,
            ]
        )

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3)
                if self.args.G_gray
                else identity_transform,
                transforms.ColorJitter(brightness=args.brightness)
                if args.brightness != None
                else identity_transform,
                transforms.ColorJitter(contrast=args.contrast)
                if args.contrast != None
                else identity_transform,
                transforms.ColorJitter(saturation=args.saturation)
                if args.saturation != None
                else identity_transform,
                transforms.ColorJitter(hue=args.hue)
                if args.hue != None
                else identity_transform,
                transforms.RandomPerspective(args.rand_perspective)
                if args.rand_perspective != None
                else identity_transform,
                transforms.RandomResizedCrop(
                    size=self.resize, scale=(1 - args.random_resized_crop, 1)
                )
                if args.random_resized_crop != None
                else identity_transform,
                transforms.RandomRotation(degrees=args.random_rotation)
                if args.random_rotation != None
                else identity_transform,
                self.resized_transform,
            ]
        )

        self.soft_positives_per_query = self.knn_neighbours(
                og_radius="soft_positives_per_query", n_jobs=-1
            )

        
        # Find hard_negatives_per_query. Hard negative is out of prior position threshold and we don't care
        if args.prior_location_threshold != -1:
            # knn = NearestNeighbors(n_jobs=-1)
            # knn.fit(self.db_coords)
            # self.hard_negatives_per_query = knn.radius_neighbors(
            #     self.q_coords,
            #     radius=args.prior_location_threshold,
            #     return_distance=False,
            # )
            self.hard_negatives_per_query = self.knn_neighbours(
                og_radius="hard_negatives_per_query", n_jobs=-1
            )
        else:
            self.hard_negatives_per_query = []

        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        self.hard_positives_per_query = self.knn_neighbours(
                og_radius="hard_positives_per_query", n_jobs=-1
            )


        # Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(
            np.array([len(p)
                     for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        # self.hard_positives_per_query = np.delete(
        #     self.hard_positives_per_query, queries_without_any_hard_positive
        # )
        self.local_q_mapping = np.arange(len(self.q_coords))
        self.local_q_mapping = np.delete(
            self.local_q_mapping, queries_without_any_hard_positive
        )
        # Recompute queries_num because some queries might have been removed
        self.queries_num = len(self.local_q_mapping)
        # self.gpu_resources = []
        # for i in range(2):
        #     # 2 gpu resource for positive
        #     res = faiss.StandardGpuResources()
        #     res.setTempMemory(200 * 1024 * 1024)  # 200 MB
        #     self.gpu_resources.append(res)
    
    

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_local_indexes[index], (1,
                                                  1, self.negs_num_per_query)
        )

        # if self.args.G_contrast:
        #     query = self.query_transform(
        #         transforms.functional.adjust_contrast(self._find_img_in_h5(query_index, "queries"), contrast_factor=3))
        # else:

        assert np.equal(self.qu_dataset[query_index][0], super().__getitem__(self.q_mapping[query_index])["item"][0]).all(), "Query image should be the same as the one in the dataset"
        query = self.query_transform(to_pil_image(self.qu_dataset[query_index][0]))
        positive = self.resized_transform(to_pil_image(self.db_dataset[best_positive_index][0]))
        negatives = [self.resized_transform(to_pil_image(self.db_dataset[i][0])) for i in neg_indexes]

        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (
                    triplets_local_indexes,
                    torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3),
                )
            )
        return images, triplets_local_indexes,self.triplets_local_indexes[index]

    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            assert super().__len__() == len(self.qu_dataset) + len(self.db_dataset), "The length of the dataset should be equal to the sum of the lengths of the query and database datasets."
            return super().__len__()
        else:
            return len(self.triplets_local_indexes)
    
    def compute_triplets(self, args, model, model_db=None):
        self.is_inference = True
        self.compute_triplets_partial(args, model, model_db)
    
    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape, cache=None):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""

        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        if cache is None:
            if args.use_faiss_gpu:
                cache = RAMEfficient2DMatrixGPU(cache_shape, dtype=torch.float32, device=args.device)
            else:
                cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        sampler = IntraDatasetBatchSampler(subset_ds.idx_to_dataset, args.infer_batch_size)
        # import pdb; pdb.set_trace()
        subset_dl = DataLoader(
            dataset=subset_ds,
            # num_workers=args.num_workers,
            # batch_size=args.infer_batch_size,
            # shuffle=False,
            batch_sampler = sampler,
        )
        model.eval()

        data_iter = iter(subset_dl)
        
        from torch.cuda.amp import autocast

        
        with torch.no_grad(), autocast():
            for _ in tqdm(range(len(subset_dl)), ncols=100):
                torch.cuda.empty_cache()
                try:
                    batch_item = next(data_iter)
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import pdb; pdb.set_trace()
                indexes = batch_item["batch_id"]
                images = batch_item["item"][0].to(args.device)
                features = model.extract_feature(images,test=False)
                if args.use_faiss_gpu:
                    for temp_idx in indexes:
                        assert cache[temp_idx.item()] is None, f"Cache for index {temp_idx} is not None, but should be!"
                    cache[indexes] = features
                else:
                    raise NotImplementedError("FAISS GPU is required for this implementation.")
                    cache[indexes.numpy()] = features.cpu().numpy()
        del data_iter, subset_dl, images, features

        return cache
    
    def get_query_features(self, query_index, cache):
        """Get the features of the query image from the cache. The input shoudl be the local index of the query image where the indexes are not deleted based on queries_without_any_hard_positive."""
        global_index = self.q_mapping[query_index]
        query_features = cache[global_index]
        if query_features is None:
            mapping_dataset_idx , mapping_local_index = self.mapping[global_index]
            raise RuntimeError(
                f"For query {self.datasets[mapping_dataset_idx].images_paths[mapping_local_index]} "
                + f"with local index {query_index} and global index {global_index} features have not been computed!\n"
                + "There might be some bug with caching"
            )
        return query_features
    
    def get_best_positive_index(self, args, query_index, cache, query_features):
        # Get the best positive index (local) for the query image.
        local_db_id = self.hard_positives_per_query[query_index]
        global_db_id = self.db_mapping[local_db_id]
        positives_features = cache[global_db_id]
        if args.use_faiss_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(200 * 1024 * 1024)  # 200 MB
            faiss_index = faiss.GpuIndexFlatL2(
                res, positives_features.shape[1])
        else:
            faiss_index = faiss.IndexFlatL2(positives_features.shape[1])
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(
            query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item(
        )
        if args.use_faiss_gpu:
            del res
        return best_positive_index
    
    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        """Get the hardest negatives indexes (local) for the query image."""
        neg_features = cache[self.db_mapping[neg_samples]]
        if args.use_faiss_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(200 * 1024 * 1024)  # 200 MB
            faiss_index = faiss.GpuIndexFlatL2(res, neg_features.shape[-1])
        else:
            faiss_index = faiss.IndexFlatL2(neg_features.shape[-1])
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(
            query_features.reshape(1, -1), self.negs_num_per_query
        )
        if args.use_faiss_gpu:
            neg_nums = neg_nums.reshape(-1).cpu()
        else:
            neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        if not hasattr(neg_indexes, "__len__"):
            neg_indexes = np.expand_dims(neg_indexes, 0)
        if args.use_faiss_gpu:
            del res
        return neg_indexes
    
    def compute_triplets_partial(self, args, model, model_db=None):

        self.triplets_local_indexes = []

        sampled_queries_indexes_post_deletion = np.random.choice(
            self.queries_num, min(args.cache_refresh_rate, self.queries_num), replace=False
        )
        sampled_queries_indexes_local = self.local_q_mapping[sampled_queries_indexes_post_deletion]
        sampled_queries_indexes_global = self.q_mapping[sampled_queries_indexes_local.tolist()].tolist()

        sampled_database_indexes_local = np.random.choice(
            self.database_num, self.neg_samples_num, replace=False
        )

        positives_indexes_local = [
            self.hard_positives_per_query[i] for i in sampled_queries_indexes_local
        ]
        positives_indexes_local = [p for pos in positives_indexes_local for p in pos]
        database_indexes_local = np.unique(list(sampled_database_indexes_local) + positives_indexes_local)

        database_indexes_global = self.db_mapping[database_indexes_local].tolist()

        subset_ds = Subset(self, database_indexes_global + sampled_queries_indexes_global)
        db_subset_ds = Subset(self, database_indexes_global)
        qu_subset_ds = Subset(self, sampled_queries_indexes_global)

        db_subset_ds.idx_to_dataset = np.array([self.mapping[i][0] for i in database_indexes_global])
        qu_subset_ds.idx_to_dataset = np.array([self.mapping[i][0] for i in sampled_queries_indexes_global])
        subset_ds.idx_to_dataset = np.concatenate((db_subset_ds.idx_to_dataset, qu_subset_ds.idx_to_dataset), axis=0)

        # import pdb; pdb.set_trace()

        if model_db is None:
            cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        else:
            cache = self.compute_cache(args, model_db, db_subset_ds, (len(self), args.features_dim))
            cache = self.compute_cache(args, model, qu_subset_ds, (len(self), args.features_dim), cache)

        torch.cuda.empty_cache()
        if args.use_faiss_gpu:
            print("Warming up FAISS GPU resources...")
            # Your vector dimension
            dim = cache.shape[-1]  # 4096

            # Warmup FAISS GPU
            start = time.time()

            # Create FAISS GPU resources
            res = faiss.StandardGpuResources()

            # Create GPU index
            gpu_index = faiss.GpuIndexFlatL2(res, dim)

            # Add a dummy vector
            dummy_data = np.random.rand(10, dim).astype('float32')
            gpu_index.add(dummy_data)

            # Run a dummy search
            dummy_query = np.random.rand(1, dim).astype('float32')
            gpu_index.search(dummy_query, k=1)

            end = time.time()
            print(f"FAISS warmup took {end - start:.3f}s")
            
        
        # # Process queries
        results = []
        dataset_id = []
        global_index_to_dataset=[self.mapping[i][0] for i in range(len(self.mapping))]

        
        print(f"[INFO] Starting multiprocessing mining on {len(sampled_queries_indexes_local)} queries...")
        num_workers = 8
        use_faiss_gpu = args.use_faiss_gpu
        available_gpus = [0]  # Set your GPU IDs here

        # Create the Pool
        pool = mp.Pool(
            processes=num_workers,
            initializer=faiss_worker_init,
            initargs=(use_faiss_gpu, 0)  # Default GPU 0 for now
        )

        # # Partial function for worker
        worker = partial(
            triplet_worker_fn,
            sampled_queries_indexes_local=sampled_queries_indexes_local,
            args=args,
            cache=cache,
            db_mapping={'db_mapping': self.db_mapping, 'q_mapping': self.q_mapping},
            hard_positives_per_query=self.hard_positives_per_query,
            hard_negatives_per_query=self.hard_negatives_per_query,
            soft_positives_per_query=self.soft_positives_per_query,
            database_indexes_local=database_indexes_local,
            negs_num_per_query=self.negs_num_per_query,
            use_faiss_gpu=use_faiss_gpu,
            gpu_res_id=None,  # No need anymore; GPU ID handled inside initializer
            db_idx_to_dataset=db_subset_ds.idx_to_dataset,
            qu_idx_to_dataset=qu_subset_ds.idx_to_dataset,
            global_index_to_dataset=global_index_to_dataset
        )

        
        for r in tqdm(pool.imap_unordered(worker, range(len(sampled_queries_indexes_local))),
                    total=len(sampled_queries_indexes_local), ncols=100):
            if r is not None:
                results.append(r[0])
                dataset_id.append(r[1])

        pool.close()
        pool.join()

        self.triplets_local_indexes = torch.tensor(results, dtype=torch.int32)
        self.triplet_idx_to_dataset = np.array(dataset_id, dtype=np.int32)
        del cache

def str_to_dataset(name):
    if name == "ms2":
        return MS2
    elif name == "cart":
        return CART
    elif name =="handheld_cart":
        return HandheldCART
    elif name == "freiburg":
        return Freiburg
    elif name == "vivid":
        return Vivid
    elif name == "sthereo":
        return STHEREO
    elif name == "boson":
        return BosonNightimeBaseDataset
    elif name == "m2p2":
        return M2P2
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def build_dataset(args,return_dataloader=True,m2p2_rgb_only=False, build_triplets=False):

    if args.train:
        mode_list = ["train", "val"]
    else:
        mode_list = [args.dataset_split_for_eval]
        if args.dataset_split_for_eval not in ["train", "val","test"]:
            raise ValueError("ddataset_idataset_split_for_eval must be either 'train' or 'val'")
    
    if not return_dataloader and len(mode_list) > 1:
        raise ValueError("If return_dataloader is False, only one mode can be specified (either 'train' or 'val').")
    combined_datasets = {mode: [] for mode in mode_list}
    combined_dataloader = {mode: None for mode in mode_list}

    dataset_names = {}

    dataset_names[mode_list[0]] = args.dataset
    if len(mode_list) > 1:
        assert args.train
        assert "val" == mode_list[1], "If there are two modes, the second one must be 'val'."
        dataset_names["val"] = args.eval_dataset if (hasattr(args, 'eval_dataset') and args.eval_dataset) else args.dataset

    print("Dataset names:", dataset_names)
    teacher_modality = args.teacher_modality
    student_modality = args.student_modality


    for mode in mode_list:
        for ds_name in dataset_names[mode]:
            ds_instace = str_to_dataset(ds_name)
            dataset_init_dict ={}
            if ds_name == "ms2":
                print("Using MS2 dataset")
                if args.debug:
                    print("Using MS2 dataset for debugging")
                    seq_list = return_ms2_split_debug(mode)
                else:
                    seq_list = return_ms2_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/MS2_full"
            elif ds_name == "cart" or ds_name == "handheld_cart":
                print("Using CART dataset")
                if args.cart_split =='segmentation':
                    raise ValueError("CART segmentation split is not supported yet. tio support see which mode to use - thermal or rgbt")
                    if mode == "train":
                        seq_list = return_cart_split_segmentation_geographic(split=mode,area="socal",mode="rgbt")
                    else:
                        seq_list = return_cart_split_segmentation_geographic(split=mode,area="northcarolina",mode="rgbt") + return_cart_split_segmentation_geographic(split=mode,area="kentucky",mode="rgbt")
                    dataset_init_dict["seq_as_txt"]="rgbt"
                    data_root = None
                    root_frame_dir = None

                    print("Using CART dataset for segmentation task")
                else:
                    if args.debug:
                        print("Using CART dataset for debugging")
                        seq_list = return_cart_split_debug(mode)
                    else:
                        seq_list = return_cart_split(mode)
                    data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
                    root_frame_dir = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
                
                dataset_init_dict["root_frame_dir"]= root_frame_dir
                dataset_init_dict["cart_split"] = args.cart_split
            elif ds_name == "freiburg":
                print("Using Freiburg dataset")
                seq_list = return_freiburg_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/freiburg"
            elif ds_name == "vivid":
                print("Using VIVID++ dataset")
                seq_list = return_vivid_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/VIVID++/extracted_data"
            elif ds_name == "sthereo":
                print("Using STHEREO dataset")
                seq_list = return_sthereo_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/STHEREO/sequences"
            elif ds_name == "boson":
                print("Using Boson Nightime dataset")
                seq_list = [mode]
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/boson_nightime"
            elif ds_name == "m2p2":
                print("Using M2P2 dataset")
                seq_list = return_m2p2_split(mode, rgb_only=m2p2_rgb_only)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/M2P2/extracted_data_new"
            elif ds_name == "mfnet":
                print("Using MFNet dataset")
                seq_list = return_mfnet_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/MFNet"
            else:
                raise ValueError(f"Unknown dataset name: {ds_name}")
            augment = False
            if args.train and mode == 'train' and args.augment:
                augment = True

            crop_during_vpr_test = False

            vpr_test = args.vpr_test
            if vpr_test and args.crop_images:
                crop_during_vpr_test = True


            dataset_init_dict.update({
                "args": args,
                "db_modality": teacher_modality,
                "q_modality": student_modality,
                "seq": seq_list,
                "datasets_folder": data_root,
                "augment": augment,
                "vpr_train": args.use_odom,
                "rescale_during_crop": args.rescale_during_crop if mode == "train" else False,
                "vpr_test": vpr_test,
                "crop_during_vpr_test": crop_during_vpr_test,
                "crop_images": args.crop_images if mode == "train" else False,
            })

            if hasattr(args, 'val_positive_dist_threshold') and args.val_positive_dist_threshold >0:
                dataset_init_dict["val_positive_dist_threshold"] = args.val_positive_dist_threshold



            if hasattr(args, 'dist_thresh'):
                dataset_init_dict["dist_thresh"] = args.dist_thresh

            print(f"For mode {mode} Dataset init dict:", dataset_init_dict)
            combined_datasets[mode].append(ds_instace(**dataset_init_dict))
    
    build_common_dataset = args.common_database if hasattr(args, 'common_database') else False

    if build_common_dataset:
        print("Building combined datasets for gps_coords ")
    else:
        print("NOT Building combined datasets for gps_coords ")
    for mode in mode_list:
        if build_triplets and mode == "train":
            dataset_cls = TripletsDataset
        else:
            dataset_cls = MultiDatasetWrapper
        wrapped_dataset = dataset_cls(args,combined_datasets[mode], dataset_names[mode], mode=mode, use_odom=args.use_odom,build_common_dataset=build_common_dataset)

        if not return_dataloader:
            return wrapped_dataset

        subsample = args.subsample_val if hasattr(args, 'subsample_val') and mode == 'val' else 1
    
        
        if args.sampling_weight =="equal":
            sample_weights = [1.0] * len(wrapped_dataset)
        elif args.sampling_weight == "inverse_length":
            dataset_lengths = [len(d) for d in combined_datasets[mode]]
            temperature = getattr(args, 'sampling_temperature', 1.0)
            weights = [(1.0 / l) ** temperature for l in dataset_lengths]
            normalized_weights = [w / sum(weights) for w in weights]

            sample_weights = []
            for weight, dataset in zip(normalized_weights, combined_datasets[mode]):
                sample_weights.extend([weight] * len(dataset))
        else:
            raise ValueError(f"Unknown sampling weight type: {args.sampling_weight}")
        

        num_workers = args.train_num_workers if mode == 'train' else args.eval_num_workers
        batch_size = args.batch_size if mode == 'train' else args.eval_batch_size

        if getattr(args, 'intra_dataset_batch', False):
            # Group indices by dataset
            dataset_to_indices = defaultdict(list)
            for idx, (d_idx, _) in enumerate(wrapped_dataset.mapping):
                dataset_to_indices[d_idx].append(idx)

            if not getattr(args, 'no_shuffle', False):
                print(f"Using IntraDatasetBatchSampler for {mode} mode")
                batch_sampler = IntraDatasetBatchSampler(dataset_to_indices, batch_size,subsample=subsample,equal_samples=getattr(args, 'equal_samples', False))
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            if not getattr(args, 'no_shuffle', False):
                print(f"Using WeightedRandomSampler for {mode} mode")
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights)//subsample, replacement=False)
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,drop_last=False)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    if args.train:
        return combined_dataloader["train"], combined_dataloader["val"]
    else:
        return combined_dataloader[args.dataset_split_for_eval]
