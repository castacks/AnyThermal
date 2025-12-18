from custom_datasets.base_dataset import *
import torchvision.transforms.functional as torch_F
import torchvision
import pandas as pd
import os

# For processing 8-bit thermal?
from custom_datasets.ms2.ms2_utils import *

def return_odom_beyond_vision_split(split):
    """
    Returns the sequence list for the OdomBeyondVision dataset.
    
    Args:
        split (str): One of ["train", "val", "test"]
    
    Returns:
        List[str]: List of sequence names
    """
    
    # All sequences sorted chronologically
    all_sequences = [
        # January 28, 2020 sequences
        "sequence_01_2020-01-28-11-10-12",
        "sequence_02_2020-01-28-11-15-11", 
        "sequence_03_2020-01-28-11-20-29",
        "sequence_04_2020-01-28-11-24-40",
        "sequence_05_2020-01-28-11-28-50",
        "sequence_06_2020-01-28-11-34-05",
        "sequence_07_2020-01-28-11-36-12",
        "sequence_08_2020-01-28-11-38-18",
        "sequence_09_2020-01-28-11-39-07",
        "sequence_10_2020-01-28-11-43-26",
        "sequence_11_2020-01-28-11-45-44",
        "sequence_12_2020-01-28-11-47-09",
        "sequence_13_2020-01-28-11-48-44",
        "sequence_14_2020-01-28-11-50-54",
        "sequence_15_2020-01-28-11-53-26",
        "sequence_16_2020-01-28-11-55-54",
        "sequence_17_2020-01-28-11-57-59",
        "sequence_18_2020-01-28-11-59-42",
        "sequence_19_2020-01-28-12-01-24",
        
        # January 29, 2020 sequences
        "sequence_20_2020-01-29-17-49-34",
        
        # February 6, 2020 sequences
        "sequence_21_2020-02-06-15-03-24",
        "sequence_22_2020-02-06-15-07-25",
        "sequence_23_2020-02-06-15-15-32",
        "sequence_24_2020-02-06-15-24-24",
        "sequence_25_2020-02-06-15-29-15",
        "sequence_26_2020-02-06-15-35-57",
        "sequence_27_2020-02-06-15-48-35",
        "sequence_28_2020-02-06-15-58-41",
        "sequence_29_2020-02-06-16-04-15",
        "sequence_30_2020-02-06-16-09-40",
        "sequence_31_2020-02-06-16-15-45",
        "sequence_32_2020-02-06-16-21-04",
        
        # February 7, 2020 sequences
        "sequence_33_2020-02-07-15-57-43",
        "sequence_34_2020-02-07-16-09-47",
        "sequence_35_2020-02-07-16-25-22",
        "sequence_36_2020-02-07-16-31-44",
        "sequence_37_2020-02-07-16-36-31",
        "sequence_38_2020-02-07-16-41-08",
        "sequence_39_2020-02-07-16-46-50",
        "sequence_40_2020-02-07-17-00-15",
        "sequence_41_2020-02-07-17-01-24",
        "sequence_42_2020-02-07-17-04-08",
        "sequence_43_2020-02-07-17-08-21",
        "sequence_44_2020-02-07-17-11-56"
    ]
    
    # Validation sequences (over 100m and not vicon)
    val_sequences = [
        "sequence_01_2020-01-28-11-10-12",  # H_corridor_circle_bright_1 - 128.2m
        "sequence_02_2020-01-28-11-15-11",  # H_corridor_circle_bright_2 - 133.0m
        "sequence_03_2020-01-28-11-20-29",  # H_corridor_circle_bright_3 - 139.3m
        "sequence_04_2020-01-28-11-24-40",  # H_corridor_circle_bright_4 - 138.6m
        "sequence_05_2020-01-28-11-28-50",  # H_corridor_circle_bright_5 - 141.5m
        "sequence_14_2020-01-28-11-50-54",  # H_viconcorridor_circle_bright_1 - 118.3m
        "sequence_15_2020-01-28-11-53-26",  # H_viconcorridor_circle_bright_2 - 108.7m
        "sequence_16_2020-01-28-11-55-54",  # H_viconcorridor_circle_bright_3 - 104.1m
        "sequence_21_2020-02-06-15-03-24",  # H_atrium_linear_bright_1 - 203.1m
        "sequence_22_2020-02-06-15-07-25",  # H_atrium_linear_bright_2 - 271.9m
        "sequence_23_2020-02-06-15-15-32",  # H_corridor_linear_bright_1 - 315.0m
        "sequence_24_2020-02-06-15-24-24",  # H_corridor_linear_bright_2 - 282.9m
        "sequence_25_2020-02-06-15-29-15",  # H_atrium_linear_bright_3 - 146.6m
        "sequence_26_2020-02-06-15-35-57",  # H_corridor_linear_bright_3 - 417.0m
        "sequence_27_2020-02-06-15-48-35",  # H_corridor_linear_bright_4 - 475.5m
        "sequence_28_2020-02-06-15-58-41",  # H_corridor_linear_bright_5 - 154.3m
        "sequence_29_2020-02-06-16-04-15",  # H_corridor_linear_bright_6 - 151.2m
        "sequence_30_2020-02-06-16-09-40",  # H_multifloor_linear_bright_1 - 318.0m
        "sequence_31_2020-02-06-16-15-45",  # H_multifloor_linear_bright_2 - 320.5m
        "sequence_32_2020-02-06-16-21-04",  # H_multifloor_linear_bright_3 - 226.0m
        "sequence_33_2020-02-07-15-57-43",  # H_hallway_circle_bright_1 - 183.3m
        "sequence_34_2020-02-07-16-09-47",  # H_hallway_circle_bright_2 - 202.2m
        "sequence_35_2020-02-07-16-25-22",  # H_glassy_circle_bright_1 - 231.5m
        "sequence_36_2020-02-07-16-31-44",  # H_hallway_circle_bright_3 - 177.5m
        "sequence_38_2020-02-07-16-41-08",  # H_corridor_linear_bright_7 - 292.2m
        "sequence_39_2020-02-07-16-46-50",  # H_canteen_circle_bright_1 - 153.4m
        "sequence_41_2020-02-07-17-01-24",  # H_office_circle_bright_4 - 115.1m
        "sequence_42_2020-02-07-17-04-08",  # H_atrium_circle_bright_1 - 146.8m
        "sequence_44_2020-02-07-17-11-56",  # H_atrium_circle_bright_2 - 139.2m
    ]
        
    if split == "val":
        return val_sequences
    elif split == "train":
        return [s for s in all_sequences if s not in val_sequences]
    else:
        raise ValueError("Invalid split. Choose 'train' or 'val'")


class OdomBeyondVision(BaseDataset):
    """
    Returns dataset class with images from database and queries for the odombeyondvision dataset
    """
    def __init__(self, args, db_modality, q_modality, datasets_folder, seq, augment, crop_images, 
                 vpr_test=False, vpr_train=False, dist_thresh=3, rescale_during_crop=False,
                 crop_during_vpr_test=False, val_positive_dist_threshold=-1, neg_ring_outer_radius=-1,
                 subsample=1):
        
        self.subsample = subsample
        # self.thr_res_after_crop = (212, 578)
        self.thr_res = (256, 640)
        
        self.crop_top = 0
        self.crop_bottom = 0
        self.crop_left = 0
        self.crop_right = 0
        
        self.val_positive_dist_threshold = val_positive_dist_threshold
        self.neg_ring_outer_radius = neg_ring_outer_radius
        self.location_type = 'gps'
        
        super().__init__(args=args, db_modality=db_modality, q_modality=q_modality, 
                        datasets_folder=datasets_folder, dist_thresh=dist_thresh, 
                        vpr_test=vpr_test, vpr_train=vpr_train, seq=seq, augment=augment, 
                        rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,
                        crop_images=crop_images)

    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }

    def seq_path(self, seq):
        """Return the full path to a sequence directory"""
        return os.path.join(self.datasets_folder, seq)

    def read_csv_data(self, seq):
        """Read and parse CSV file for a sequence"""
        seq_dir = self.seq_path(seq)
        csv_files = [f for f in os.listdir(seq_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV file found in sequence {seq}")
        
        # Assume first CSV file is the pose data
        csv_path = os.path.join(seq_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Apply subsampling
        if self.subsample > 1:
            df = df.iloc[::self.subsample].reset_index(drop=True)
        
        return df

    def generate_image_paths(self, db_abs_paths, q_abs_paths):
        """Generate absolute paths for database and query images"""
        for seq in self.seq:
            df = self.read_csv_data(seq)
            seq_dir = self.seq_path(seq)
            
            # Get RGB and thermal directories
            rgb_dir = os.path.join(seq_dir, 'rgb')
            thermal_dir = os.path.join(seq_dir, 'thermal')
            
            if not os.path.exists(rgb_dir):
                raise ValueError(f"RGB directory not found in sequence {seq}")
            if not os.path.exists(thermal_dir):
                raise ValueError(f"Thermal directory not found in sequence {seq}")
            
            # Generate paths based on modality
            for _, row in df.iterrows():
                rgb_filename = row['rgb_filename']
                thermal_filename = row['thermal_filename']
                
                rgb_path = os.path.join(rgb_dir, rgb_filename)
                thermal_path = os.path.join(thermal_dir, thermal_filename)
                
                # Add paths based on modality
                if self.db_modality == "rgb":
                    if os.path.exists(rgb_path):
                        db_abs_paths.append(rgb_path)
                elif self.db_modality == "thr":
                    if os.path.exists(thermal_path):
                        db_abs_paths.append(thermal_path)
                
                if self.q_modality == "rgb":
                    if os.path.exists(rgb_path):
                        q_abs_paths.append(rgb_path)
                elif self.q_modality == "thr":
                    if os.path.exists(thermal_path):
                        q_abs_paths.append(thermal_path)
        
        return db_abs_paths, q_abs_paths

    def form_db_qu_coords(self):
        """Form database and query coordinates from CSV files"""
        self.db_coords = []
        self.q_coords = []
        
        for seq in self.seq:
            df = self.read_csv_data(seq)
            
            coords = []
            for _, row in df.iterrows():
                coords.append((row['pos_x'], row['pos_y']))
            
            # Add coordinates for both database and query
            self.db_coords.extend(coords)
            self.q_coords.extend(coords)

    def read_rgb(self, path):
        """Reads rgb image from the path."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image at {path} could not be read. Please check the path.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        h, w = img.shape[:2]
        # img = img[self.crop_top:h - self.crop_bottom, self.crop_left:w - self.crop_right]
        # img = cv2.resize(img, (self.thr_res[1], self.thr_res[0]), interpolation=cv2.INTER_AREA)
        img = base_transform(img)
        return img

    def read_thermal(self, path):
        """Reads thermal image from the path."""
        img = load_as_float_img(path)
        img = process_one_image(img,type="hist_99")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h,w = img.shape[:2]
        # img = img[self.crop_top:h - self.crop_bottom, self.crop_left:w - self.crop_right].copy()
        img = base_transform(img)
        return img

    def check_seq_list(self, seq):
        """Checks if the sequence list is valid."""
        for s in seq:
            seq_dir = self.seq_path(s)
            if not os.path.exists(seq_dir):
                raise ValueError(f"Sequence {s} does not exist at {seq_dir}. Please check the sequence name.")
            
            # Check for required directories
            rgb_dir = os.path.join(seq_dir, 'rgb')
            thermal_dir = os.path.join(seq_dir, 'thermal')
            
            if not os.path.exists(rgb_dir):
                raise ValueError(f"RGB directory not found in sequence {s}")
            if not os.path.exists(thermal_dir):
                raise ValueError(f"Thermal directory not found in sequence {s}")
            
            # Check for CSV file
            csv_files = [f for f in os.listdir(seq_dir) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in sequence {s}")

    def semantic_classes_num_and_map_to_rgb(self):
        """Return semantic class information (not applicable for this dataset)"""
        return -1, {}