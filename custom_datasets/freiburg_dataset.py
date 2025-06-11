from .base_dataset import *

def return_freiburg_split(split):
    all_sequences = [
        'train_seq_00_day_00', 'train_seq_00_day_01', 'train_seq_00_day_02', 'train_seq_00_day_03', 'train_seq_00_day_04',
        'train_seq_00_night_00', 'train_seq_00_night_01', 'train_seq_00_night_02', 'train_seq_00_night_03', 'train_seq_00_night_04', 'train_seq_00_night_05',
        'train_seq_01_day_00', 'train_seq_01_day_01', 'train_seq_01_day_02', 'train_seq_01_day_03', 'train_seq_01_day_04',
        'train_seq_01_night_00', 'train_seq_01_night_01', 'train_seq_01_night_02', 'train_seq_01_night_03', 'train_seq_01_night_04', 'train_seq_01_night_05',
        'train_seq_02_day_00', 'train_seq_02_day_01', 'train_seq_02_day_02', 'train_seq_02_day_03', 'train_seq_02_day_04',
        'train_seq_02_night_00', 'train_seq_02_night_01', 'train_seq_02_night_02', 'train_seq_02_night_03', 'train_seq_02_night_04', 'train_seq_02_night_05', 'train_seq_02_night_06', 'train_seq_02_night_07',
        'train_seq_03_day_00', 'train_seq_03_day_01', 'train_seq_03_day_02', 'train_seq_03_day_03', 'train_seq_03_day_04', 'train_seq_03_day_05',
        'train_seq_04_day_00', 'train_seq_04_day_01', 'train_seq_04_day_02', 'train_seq_04_day_03', 'train_seq_04_day_04', 'train_seq_04_day_05', 'train_seq_04_day_06', 'train_seq_04_day_07',
    ]

    val_prefixes = ['train_seq_01_night', 'train_seq_02_day']

    if split == "val":
        val_seq_list = [s for s in all_sequences if any(s.startswith(prefix) for prefix in val_prefixes)]
        return val_seq_list

    elif split == "train":
        val_seq_set = set([s for s in all_sequences if any(s.startswith(prefix) for prefix in val_prefixes)])
        train_seq_list = [s for s in all_sequences if s not in val_seq_set]
        return train_seq_list

    else:
        raise ValueError("Please provide a valid split name. Options are 'train' or 'val'")



class Freiburg(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,vpr_test=False,vpr_train=False,dist_thresh = 25,use_clahe=True, rescale_during_crop=True,crop_during_vpr_test=False):


        assert crop_during_vpr_test == False, "Crop during VPR test is not supported for Freiburg dataset. Please set it to False."
        self.frame_list_dir = "/ocean/projects/cis220039p/mdt2/datasets/freiburg/frame_list"
        self.metadata = "/ocean/projects/cis220039p/mdt2/datasets/freiburg/crop_box_metadata.txt"
        self.read_metadata()
        #PARV_TODO: Change this if GPS data becomes available
        vpr_train = False
        vpr_test = False
        dist_thresh = -1
        
        self.use_clahe = use_clahe

        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop)
    
    def read_metadata(self):
        with open(self.metadata, 'r') as f:
            lines = f.readlines()
            self.crop_box_metadata = {}
            for line in lines:
                items = line.strip().split(" ")
                value = items[-1]
                key = items[0][:-1] # Remove the trailing colon
                self.crop_box_metadata[key] = value
        for k in ["row_start", "row_end", "col_start", "col_end"]:
            if k not in self.crop_box_metadata:
                raise ValueError(f"Metadata file is missing key: {k}. Please check the metadata file.")
        self.row_start = int(self.crop_box_metadata["row_start"])
        self.row_end = int(self.crop_box_metadata["row_end"])
        self.col_start = int(self.crop_box_metadata["col_start"])
        self.col_end = int(self.crop_box_metadata["col_end"])
        self.crop_box = (self.row_start, self.row_end, self.col_start, self.col_end)
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "seg_mask": self.read_seg_mask
        }

    
    def seq_path(self,seq):
        """
        Returns the path to the sequence.
        """
        items = seq.split("_")
        split = items[0]
        seq_name = "_".join(items[1:-1])
        subseq = items[-1].replace(".txt","")
        return os.path.join(self.datasets_folder,split,seq_name,subseq)
    
    def frame_list_from_seq(self,seq):
        frame_list_txt = os.path.join(self.frame_list_dir,f"{seq}.txt")
        if not os.path.exists(frame_list_txt):
            import pdb;pdb.set_trace()
            raise ValueError(f"Frame list for sequence {seq} does not exist. Please generate it first.")
        with open(frame_list_txt, 'r') as f:
            frame_list = f.readlines()
            frame_list = [x.strip() for x in frame_list]
        return frame_list
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            seq_path = self.seq_path(seq)
            frame_list = self.frame_list_from_seq(seq)
            if self.db_modality == "rgb":
                db_folder = "fl_rgb/fl_rgb_"
            elif self.db_modality == "thr":
                if self.use_clahe:
                    db_folder = "thermal8_clahe/fl_ir_aligned_"              
                else:
                    db_folder = "thermal8_aligned_RGB/fl_ir_aligned_"
            elif self.db_modality == "seg_mask":
                db_folder = "fl_rgb_labels/fl_rgb_labels_"

            if self.q_modality == "rgb":
                q_folder = "fl_rgb/fl_rgb_"
            elif self.q_modality == "thr":
                if self.use_clahe:
                    q_folder = "thermal8_clahe/fl_ir_aligned_"              
                else:
                    q_folder = "thermal8_aligned_RGB/fl_ir_aligned_"
            elif self.q_modality == "seg_mask":
                q_folder = "fl_rgb_labels/fl_rgb_labels_"


            for frame in frame_list:
                db_abs_paths.append(os.path.join(seq_path,db_folder+frame))
                q_abs_paths.append(os.path.join(seq_path,q_folder+frame))
            
        return db_abs_paths,q_abs_paths
    
    def semantic_classes_num_and_map_to_rgb(self):
        coding = {}
        coding[-1] = [100,100,100]
        coding[0] = [0, 0, 0]
        coding[1] = [200, 100, 50]
        coding[2] = [0, 0, 255]
        coding[3] = [255, 220, 200]
        coding[4] = [0, 255, 0]
        coding[5] = [0, 255, 255]
        coding[6] = [255, 255, 0]
        coding[7] = [20, 180, 150]
        coding[8] = [200, 50, 255]
        coding[9] = [80, 10, 100]
        coding[10] = [20, 150, 220]
        coding[11] = [230, 120, 10]
        coding[12] = [255, 0, 0]
        return 13,coding

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image at {path} could not be read. Please check the path.")
        #apply the crop box from the metadata
        img = img[self.row_start:self.row_end, self.col_start:self.col_end]
        # print("rgb pre base_transform shape",img.shape)

        img = base_transform(img)
        # print("rgb base_transform shape",img.shape)

        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to grayscale
        # print("thermal pre base_transform shape",img.shape)
        img = base_transform(img)
        # print("thermal base_transform shape",img.shape)
        return img
    
    def read_seg_mask(self, path):
        """
        Reads segmentation mask from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Segmentation mask at {path} could not be read. Please check the path.")
        #apply the crop box from the metadata
        img = img[self.row_start:self.row_end, self.col_start:self.col_end]
        img = base_transform(img)
        return img
    
    def form_gt_positives(self):
        # PARV_TODO : Implement this if GPS data is available
        pass
    
    def check_seq_list(self,seq):
        """
        Checks if the sequence list is valid.
        """
        for s in seq:
            if not os.path.exists(self.seq_path(s)):
                import pdb;pdb.set_trace()
                raise ValueError(f"Sequence {s} does not exist. Please check the sequence name.")