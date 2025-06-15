from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader, Subset, RandomSampler, BatchSampler
from collections import defaultdict
from .ms2_dataset import MS2, return_ms2_split
from .cart_dataset import CART, return_cart_split_segmentation_geographic
from .freiburg_dataset import Freiburg, return_freiburg_split
from.vivid_dataset import Vivid, return_vivid_split
from.sthereo_dataset import STHEREO, return_sthereo_split
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import os
import math

class IntraDatasetBatchSampler:
    def __init__(self, dataset_to_indices, batch_size):
        self.dataset_to_indices = dataset_to_indices
        self.batch_size = batch_size

    def __iter__(self):
        keys = list(self.dataset_to_indices.keys())
        random.shuffle(keys)
        all_batches = []
        for d_idx in keys:
            indices = self.dataset_to_indices[d_idx][:]
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                # if len(batch) == self.batch_size:
                all_batches.append(batch)
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return sum(math.ceil(len(idxs) / self.batch_size) for idxs in self.dataset_to_indices.values())


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets, dataset_names, mode, use_odom=False, dist_thresh=25,build_common_dataset = False):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.mapping = []
        self.mode = mode
        self.use_odom = use_odom
        for d_idx, d in enumerate(datasets):
            self.mapping.extend([(d_idx, i) for i in range(len(d))])

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
                        global_index += len(d.soft_positives_per_query)
                    else:
                        raise ValueError(f"Dataset {dataset_names[d_idx]} does not have soft_positives_per_query attribute")

                    self.db_coords.extend(d.db_coords)
                    self.q_coords.extend(d.q_coords)

                for i in range(len(self.db_coords)):
                    self.db_coords[i] = self.db_coords[i][:2]
                    self.q_coords[i] = self.q_coords[i][:2]

                self.db_coords = np.array(self.db_coords)
                self.q_coords = np.array(self.q_coords)

                if build_common_dataset:
                    print("Finding common soft positives in the database and queries")

                    knn = NearestNeighbors(n_jobs=-1)
                    knn.fit(self.db_coords)
                    _, self.common_soft_positives = knn.radius_neighbors(self.q_coords, radius=dist_thresh, return_distance=True)
                    assert len(self.common_soft_positives) == len(self.soft_positives), "Mismatch between soft positives"
                    # IMP self.common_soft_positives can be differnt from self.soft_positives as it only compares 2d coordinates whereas self.soft_positives contains the indices of the soft positives in the database which can compare 3d also if the data is available.                
                    print("Found common soft positives in the database and queries")
            else:
                raise ValueError("Soft positives not available. Check dataset classes.")

        self.database_num = len(self.mapping)
        self.db_dataset, self.qu_dataset = self.concat_datasets_separately()

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.mapping[idx]
        item = {}
        item["item"] = self.datasets[dataset_idx][local_idx]
        item["dataset_name"] = self.dataset_names[dataset_idx]
        item["batch_id"] = idx
        item["dataset_id"] = dataset_idx  # NEW: used to group batch by dataset
        return item

    def concat_datasets_separately(self):
        db_dataset_list = []
        q_dataset_list = []
        for d in self.datasets:
            db_dataset = Subset(d, range(d.database_num))
            q_dataset = Subset(d, range(d.database_num, len(d)))
            db_dataset_list.append(db_dataset)
            q_dataset_list.append(q_dataset)
        return ConcatDataset(db_dataset_list), ConcatDataset(q_dataset_list)


def str_to_dataset(name):
    if name == "ms2":
        return MS2
    elif name == "cart":
        return CART
    elif name == "freiburg":
        return Freiburg
    elif name == "vivid":
        from .vivid_dataset import Vivid
        return Vivid
    elif name == "sthereo":
        from .sthereo_dataset import STHEREO
        return STHEREO
    elif name == "boson":
        from .boson_nightime_dataset import BosonNightimeBaseDataset
        return BosonNightimeBaseDataset
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def build_dataset(args,return_dataloader=True):

    if args.train:
        mode_list = ["train", "val"]
    else:
        mode_list = [args.dataset_split_for_eval]
        if args.dataset_split_for_eval not in ["train", "val"]:
            raise ValueError("ddataset_idataset_split_for_eval must be either 'train' or 'val'")
    
    if not return_dataloader and len(mode_list) > 1:
        raise ValueError("If return_dataloader is False, only one mode can be specified (either 'train' or 'val').")
    combined_datasets = {mode: [] for mode in mode_list}
    combined_dataloader = {mode: None for mode in mode_list}


    # dataset_names = args.dataset  # e.g., ["ms2", "cart"]

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
                seq_list = return_ms2_split(mode)
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/MS2_full"
            elif ds_name == "cart":
                print("Using CART dataset")
                if mode =="train" and args.train_easy:
                    raise ValueError("train_easy is not supported for CART dataset")
                else:
                    seq_list = return_cart_split_segmentation_geographic(split=mode,area="socal",mode="rgbt")
                data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
                root_frame_dir = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
                data_root = None
                root_frame_dir = None
                dataset_init_dict["root_frame_dir"]= root_frame_dir
                dataset_init_dict["seq_as_txt"]="rgbt"
                # import pdb;pdb.set_trace()
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
                if mode == "train":
                    dataset_init_dict["subsample"] = args.subsample if hasattr(args, 'subsample') else 1
            augment = False
            if args.train and mode == 'train' and args.augment:
                augment = True
            
            rescale_during_crop = False
            if args.train and mode == 'train' and args.rescale_during_crop:
                rescale_during_crop = True

            crop_during_vpr_test = False

            vpr_test = args.vpr_test
            if vpr_test and args.common_database:
                crop_during_vpr_test = True

                
            dataset_init_dict.update({
                "db_modality": teacher_modality,
                "q_modality": student_modality,
                "seq": seq_list,
                "datasets_folder": data_root,
                "augment": augment,
                "vpr_train": args.use_odom,
                "rescale_during_crop": rescale_during_crop,
                "vpr_test": vpr_test,
                "crop_during_vpr_test": crop_during_vpr_test,
                "crop_images": args.crop_images
            })

            print(f"For mode {mode} Dataset init dict:", dataset_init_dict)
            combined_datasets[mode].append(ds_instace(**dataset_init_dict))
    
    build_common_dataset = args.common_database if hasattr(args, 'common_database') else False

    if build_common_dataset:
        print("Building combined datasets for gps_coords ")
    else:
        print("NOT Building combined datasets for gps_coords ")
    for mode in mode_list:
        wrapped_dataset = MultiDatasetWrapper(combined_datasets[mode], dataset_names[mode], mode=mode, use_odom=args.use_odom,build_common_dataset=build_common_dataset)

        if not return_dataloader:
            return wrapped_dataset

        dataset_lengths = [len(d) for d in combined_datasets[mode]]
        temperature = 0.5
        weights = [(1.0 / l) ** temperature for l in dataset_lengths]
        normalized_weights = [w / sum(weights) for w in weights]

        sample_weights = []
        for weight, dataset in zip(normalized_weights, combined_datasets[mode]):
            sample_weights.extend([weight] * len(dataset))

        num_workers = args.train_num_workers if mode == 'train' else args.eval_num_workers

        if getattr(args, 'intra_dataset_batch', False):
            # Group indices by dataset
            dataset_to_indices = defaultdict(list)
            for idx, (d_idx, _) in enumerate(wrapped_dataset.mapping):
                dataset_to_indices[d_idx].append(idx)

            # Build dataset-wise samplers
            # from torch.utils.data import ChainDataset
            # all_samplers = []
            # for indices in dataset_to_indices.values():
            #     batch_sampler = BatchSampler(RandomSampler(indices), batch_size=args.batch_size, drop_last=True)
            #     all_samplers.extend(list(batch_sampler))
            if not getattr(args, 'no_shuffle', False):
                print(f"Using IntraDatasetBatchSampler for {mode} mode")
                batch_sampler = IntraDatasetBatchSampler(dataset_to_indices, args.batch_size)
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
            # import pdb;pdb.set_trace()
        else:
            if not getattr(args, 'no_shuffle', False):
                print(f"Using WeightedRandomSampler for {mode} mode")
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers,drop_last=False)
            else:
                print(f"Using no shuffle for {mode} mode")
                combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    if args.train:
        return combined_dataloader["train"], combined_dataloader["val"]
    else:
        return combined_dataloader[args.dataset_split_for_eval]
