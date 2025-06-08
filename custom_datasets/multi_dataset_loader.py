from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader
from .ms2_dataset import MS2, return_ms2_split
from .cart_dataset import CART, return_cart_split
from .freiburg_dataset import Freiburg, return_freiburg_split

class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets, dataset_names,mode,use_odom=False):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.mapping = []  # List of tuples (dataset_idx, local_idx)
        self.mode = mode
        self.use_odom = use_odom
        for d_idx, d in enumerate(datasets):    
            self.mapping.extend([(d_idx, i) for i in range(len(d))])
        if self.use_odom:
            if hasattr(datasets[0],'soft_positives_per_query'):
                print("building combined soft_positives for validation set")
                self.soft_positives = []
                global_index = 0
                for d_idx, d in enumerate(datasets):
                    if hasattr(d, 'soft_positives_per_query'):
                        self.soft_positives.extend([[global_index+i for i in d.soft_positives_per_query[idx]] for idx in range(len(d.soft_positives_per_query))])
                        global_index += len(d.soft_positives_per_query)
                    else:
                        raise ValueError(f"Dataset {dataset_names[d_idx]} does not have soft_positives_per_query attribute")
            else:
                raise ValueError("Soft positives are not available for the datasets in the MultiDatasetWrapper. Please check the dataset classes.")
            # import pdb;pdb.set_trace()
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.mapping[idx]
        item = {}

        item["item"] = self.datasets[dataset_idx][local_idx]
        item["dataset_name"] = self.dataset_names[dataset_idx]
        item["batch_id"] = idx
        # if self.mode == 'val':
        #     item["soft_positives_per_query"] = self.soft_positives[idx]

        return item

def str_to_dataset(name):
    if name == "ms2":
        return MS2
    elif name == "cart":
        return CART
    elif name == "freiburg":
        return Freiburg
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def build_dataset(args):

    if args.train:
        mode_list = ["train", "val"]
    else:
        mode_list = ["val"]
    
    combined_datasets = {mode: [] for mode in mode_list}
    combined_dataloader = {mode: None for mode in mode_list}


    dataset_names = args.dataset  # e.g., ["ms2", "cart"]

    teacher_modality = args.teacher_modality
    student_modality = args.student_modality

    if not args.train:
        if not (isinstance(args.seq,list) and len(args.seq) > 0):
            raise ValueError("Please provide a sequence list")

    for ds_name in dataset_names:
        ds_instace = str_to_dataset(ds_name)
        for mode in mode_list:
            dataset_init_dict ={}
            if ds_name == "ms2":
                print("Using MS2 dataset")
                train_seq_list = return_ms2_split("train")
                val_seq_list = return_ms2_split("val")
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/MS2_full"
            elif ds_name == "cart":
                print("Using CART dataset")
                train_seq_list = return_cart_split("train_easy" if args.train_easy else "train")
                val_seq_list = return_cart_split("val")
                data_root = "/ocean/projects/cis220039p/mdt2/shared/CART/bag_files"
                
                root_frame_dir = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/caltech-aerial-rgbt-dataset/splits/parv/filter/static_segments_output/frames"
                dataset_init_dict["root_frame_dir"]= root_frame_dir
            elif ds_name == "freiburg":
                print("Using Freiburg dataset")
                train_seq_list = return_freiburg_split("train")
                val_seq_list = return_freiburg_split("val")
                data_root = "/ocean/projects/cis220039p/mdt2/datasets/freiburg"
            
            if not args.train:
                val_seq_list = args.seq
                
            dataset_init_dict.update({
                "db_modality": teacher_modality,
                "q_modality": student_modality,
                "seq": val_seq_list if mode == 'val' else train_seq_list,
                "datasets_folder": data_root,
                "augment": args.augment if mode == 'train' else False,
                "vpr_train": args.use_odom
            })
            combined_datasets[mode].append(ds_instace(**dataset_init_dict)) 

    for mode in mode_list:
        
        # ---- Compute sampling weights ----
        dataset_lengths = [len(d) for d in combined_datasets[mode]]
        inverse_lengths = [1.0 / l for l in dataset_lengths]
        normalized_weights = [w / sum(inverse_lengths) for w in inverse_lengths]

        sample_weights = []
        for weight, dataset in zip(normalized_weights, combined_datasets[mode]):
            sample_weights.extend([weight] * len(dataset))

        wrapped_dataset = MultiDatasetWrapper(combined_datasets[mode], dataset_names, mode=mode, use_odom=args.use_odom)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        combined_dataloader[mode] = DataLoader(wrapped_dataset, batch_size=args.batch_size,num_workers=4,sampler=sampler)
    
    if args.train:
        return combined_dataloader["train"], combined_dataloader["val"]
    else:
        return combined_dataloader["val"]
