from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader, Subset, RandomSampler, BatchSampler
from collections import defaultdict
from .ms2_dataset import MS2, return_ms2_split, return_ms2_split_debug
from .cart_dataset import CART, HandheldCART,return_cart_split, return_cart_split_debug,return_handheld_cart_split
from .freiburg_dataset import Freiburg, return_freiburg_split
from .vivid_dataset import Vivid, return_vivid_split
from .sthereo_dataset import STHEREO, return_sthereo_split
from .boson_nightime_dataset import BosonNightimeBaseDataset
from .m2p2_dataset import M2P2, return_m2p2_split
from .mfnet_dataset import MFNet, return_mfnet_split
from .tartanrgbt_dataset import TartanRGBT, return_tartanrgbt_split
from .odombeyondvision import OdomBeyondVision, return_odom_beyond_vision_split
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

def seed_worker(worker_id: int):
    # PyTorch already seeds torch’s RNG for each worker.
    # Sync numpy & python.random to that same seed so your augmentations match.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Optional: re-assert torch seed if you want (not required):
    # torch.manual_seed(worker_seed)

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
            self.db_coords = []
            self.q_coords = []
            for d_idx, d in enumerate(datasets):
                self.db_coords.extend(d.db_coords)
                self.q_coords.extend(d.q_coords)

            for i in range(len(self.db_coords)):
                if self.db_coords[i] is not None:
                    self.db_coords[i] = self.db_coords[i][:2]
                    self.q_coords[i] = self.q_coords[i][:2]

            
            self.hard_positives_per_query = self.knn_neighbours("hard_positives_per_query", n_jobs=-1)
            self.extra_margin_soft_positives = self.knn_neighbours("soft_positives_per_query", n_jobs=-1)
            self.ring_negatives = self.knn_neighbours("ring_negatives_per_query", n_jobs=-1)

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
    
    
    def get_hard_positives_per_query(self):
        return self.hard_positives_per_query
    def get_extra_margin_soft_positives(self):
        return self.extra_margin_soft_positives
    def get_ring_negatives(self):
        return self.ring_negatives
    def knn_neighbours(self, og_radius, n_jobs=-1):
        final_output_list = []
        running_total_datase_len = 0
        radius = None
        for d_idx, d in enumerate(self.datasets):
            d_location_type = getattr(d, 'location_type')
            assert d_location_type in ['gps', 'time'], f"Dataset {self.dataset_names[d_idx]} has unknown location type: {d_location_type}. Use 'gps' or 'time'."

            if og_radius == 'hard_positives_per_query':
                if d_location_type == 'gps':
                    radius = d.dist_thresh
                elif d_location_type == 'time':
                    radius = d.positive_radius_index
            elif og_radius == 'soft_positives_per_query':
                if d_location_type == 'gps':
                    radius = d.val_positive_dist_threshold
                elif d_location_type == 'time':
                    radius = d.val_extra_margin_positive_radius_index
            elif og_radius == 'outer_ring_negatives_per_query':
                if d_location_type == 'gps':
                    radius = d.neg_ring_outer_radius
                elif d_location_type == 'time':
                    radius = d.neg_ring_outer_radius_index
            elif og_radius == 'ring_negatives_per_query':
                outer_circle = self.knn_neighbours('outer_ring_negatives_per_query', n_jobs=n_jobs)
                inner_circle = self.knn_neighbours('soft_positives_per_query', n_jobs=n_jobs)
                #ring is outer_circle - inner_circle
                result = np.array([list(set(a) - set(b)) for a, b in zip(outer_circle, inner_circle)], dtype=object)
                return result
            else:
                raise ValueError(f"Unknown radius type: {og_radius}. Use 'hard_positives_per_query', 'soft_positives_per_query' or 'hard_negatives_per_query'.")
            if d_location_type == 'gps':
                knn = NearestNeighbors(n_jobs=n_jobs)
                knn.fit(d.db_coords)

                neighbours = knn.radius_neighbors(
                    d.q_coords, radius=radius, return_distance=False
                )
                setattr(d, og_radius, neighbours)
                global_neighbours = np.array(neighbours, dtype=object) + running_total_datase_len
                final_output_list.append(global_neighbours)
                running_total_datase_len += len(d.db_coords)
            elif d_location_type == 'time':
                neighbours_index = [None for _ in range(len(d.q_abs_paths))]
                for i in range(len(d.q_abs_paths)):
                    neighbours_index[i] = np.array(list(range(max(0, i - radius), min(len(d.db_abs_paths), i + radius + 1))))
                
                setattr(d, og_radius, neighbours_index)
                global_neighbours_index = np.array(neighbours_index, dtype=object) + running_total_datase_len
                final_output_list.append(global_neighbours_index)
                running_total_datase_len += len(neighbours_index)
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
    elif name =="tartanrgbt":
        return TartanRGBT
    elif name == "odombeyondvision":
        return OdomBeyondVision
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def build_dataset(args,return_dataloader=True,m2p2_rgb_only=False, build_triplets=False):

    if args.train:
        mode_list = ["train", "val"]
    else:
        mode_list = [args.dataset_split_for_eval]
        if args.dataset_split_for_eval not in ["train", "val","test","all","no_split"]:
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
                data_root = os.environ["ANYTHERMAL_MS2_DATA_ROOT"]
            elif ds_name == "cart" or ds_name == "handheld_cart":
                print("Using CART dataset")
                if args.debug:
                    print("Using CART dataset for debugging")
                    seq_list = return_cart_split_debug(mode)
                else:
                    if ds_name == "handheld_cart":
                        seq_list = return_handheld_cart_split(mode)
                    elif ds_name == "cart":
                        seq_list = return_cart_split(mode)
                    else:
                        raise ValueError(f"Unknown CART dataset name: {ds_name}")
                data_root = os.environ["ANYTHERMAL_CART_DATA_ROOT"]
                root_frame_dir = os.path.join(os.environ["ANYTHERMAL_PROJECT_ROOT"],"custom_datasets/splits/CART/static_segments_output/frames")
                
                dataset_init_dict["root_frame_dir"]= root_frame_dir
                dataset_init_dict["cart_split"] = args.cart_split
            elif ds_name == "freiburg":
                print("Using Freiburg dataset")
                seq_list = return_freiburg_split(mode)
                data_root = os.environ["ANYTHERMAL_FREIBURG_DATA_ROOT"]
            elif ds_name == "vivid":
                print("Using VIVID++ dataset")
                seq_list = return_vivid_split(mode)
                data_root = os.environ["ANYTHERMAL_VIVID++_DATA_ROOT"]
                dataset_init_dict["root_frame_dir"] = os.path.join(os.environ["ANYTHERMAL_PROJECT_ROOT"],"custom_datasets/splits/VIVID++/frame_lists")

            elif ds_name == "sthereo":
                print("Using STHEREO dataset")
                seq_list = return_sthereo_split(mode)
                data_root = os.environ["ANYTHERMAL_STHEREO_DATA_ROOT"]
                dataset_init_dict["root_frame_dir"]= os.path.join(os.environ["ANYTHERMAL_PROJECT_ROOT"],"custom_datasets/splits/sthereo/frame_lists")
            elif ds_name == "boson":
                print("Using Boson Nightime dataset")
                seq_list = [mode]
                data_root = os.environ["ANYTHERMAL_BOSON_DATA_ROOT"]
            elif ds_name == "m2p2":
                print("Using M2P2 dataset")
                data_root = os.environ["ANYTHERMAL_M2P2_DATA_ROOT"]
                seq_list = return_m2p2_split(data_root,mode, rgb_only=m2p2_rgb_only)
                
            elif ds_name == "mfnet":
                print("Using MFNet dataset")
                seq_list = return_mfnet_split(mode)
                data_root = os.environ["ANYTHERMAL_MFNET_DATA_ROOT"]
            elif ds_name == "tartanrgbt":
                print("Using TartanRGBT dataset")
                seq_list = return_tartanrgbt_split(mode, debug=args.debug)
                data_root = os.environ["ANYTHERMAL_TARTANRGBT_DATA_ROOT"]
            elif ds_name == "odombeyondvision":
                print("Using OdomBeyondVision dataset")
                if hasattr(args, 'local_seq'):
                    seq_list = [args.local_seq]
                else:
                    seq_list = return_odom_beyond_vision_split(mode)
                data_root = os.environ["ANYTHERMAL_ODOMBEYONDVISION_DATA_ROOT"]
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
            if hasattr(args, 'neg_ring_outer_radius') and args.neg_ring_outer_radius >0:
                dataset_init_dict["neg_ring_outer_radius"] = args.neg_ring_outer_radius
            



            if hasattr(args, 'dist_thresh') and args.dist_thresh >0:
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
        g = torch.Generator()
        g.manual_seed(args.seed)
        if getattr(args, 'intra_dataset_batch', False):
            # Group indices by dataset
            dataset_to_indices = defaultdict(list)
            for idx, (d_idx, _) in enumerate(wrapped_dataset.mapping):
                dataset_to_indices[d_idx].append(idx)

            if not getattr(args, 'no_shuffle', False):
                print(f"Using IntraDatasetBatchSampler for {mode} mode")
                batch_sampler = IntraDatasetBatchSampler(dataset_to_indices, batch_size,subsample=subsample,equal_samples=getattr(args, 'equal_samples', False))
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_sampler=batch_sampler, num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,worker_init_fn=seed_worker,generator=g)
        else:
            if not getattr(args, 'no_shuffle', False):
                print(f"Using WeightedRandomSampler for {mode} mode")
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights)//subsample, replacement=False)
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,drop_last=False,worker_init_fn=seed_worker,generator=g)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False,worker_init_fn=seed_worker,generator=g)
    if args.train:
        return combined_dataloader["train"], combined_dataloader["val"]
    else:
        return combined_dataloader[args.dataset_split_for_eval]
