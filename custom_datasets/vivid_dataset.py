from .base_dataset import *

def return_vivid_split(split):
    """
    Returns the sequence list for the VIVID++ dataset.

    Args:
        split (str): One of ["train", "val"]

    Returns:
        List[str]: List of relative sequence paths like 'driving_full/city_day1'
    """

    all_sequences = [
        "driving_full/campus_day2",
        "driving_full/city_day2",
        "driving_full/city_night",
        "driving_full/campus_day1",
        "driving_full/campus_evening",
        "driving_full/campus_night",
        "driving_full/city_evening",
        "driving_full/city_day1",
        "driving_vision/urban_night",
        "driving_vision/campus_night_2",
        "driving_vision/campus_morning_2",
        "driving_vision/campus_day2",
        "driving_vision/campus_day2_2",
        "driving_vision/city_day2",
        "driving_vision/city_night",
        "driving_vision/urban_day",
        "driving_vision/campus_evening",
        "driving_vision/campus_morning_manual_small",
        "driving_vision/urban_evening_road",
        "driving_vision/city_morning_manual",
        "driving_vision/campus_night",
        "driving_vision/urban_evening",
        "driving_vision/campus_morning_manual",
        "driving_vision/campus_morning",
        "driving_vision/urban_morning_manual",
        "driving_vision/city_evening",
        "driving_vision/city_day1"
    ]

    if split == "val":
        return [s for s in all_sequences if s.startswith("driving_vision/urban")]
    elif split == "train":
        return [s for s in all_sequences if not s.startswith("driving_vision/urban")]
    else:
        raise ValueError("Invalid split. Choose 'train' or 'val'.")



class Vivid(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,vpr_test=False,vpr_train=False,dist_thresh = 25, rescale_during_crop=True, crop_during_vpr_test=False):

        self.frame_list_dir = "/ocean/projects/cis220039p/mdt2/datasets/VIVID++/frame_lists"
        self.thr_res = (512, 640)

        self.crop_top = 122
        self.crop_bottom = 122
        self.crop_left = 145
        self.crop_right = 108

        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal
        }

    def seq_path(self,seq):
        return os.path.join(self.datasets_folder,seq)

 
    def frame_list_dir_from_seq(self,seq):
        return os.path.join(self.frame_list_dir,seq)

    def read_frame_lists(self,frame_list_dir,frame_list):
        """
        Reads frame list from the frame_list_dir.
        """
        frame_list_path = os.path.join(frame_list_dir,frame_list)
        if not os.path.exists(frame_list_path):
            raise ValueError(f"Sequence {seq_path} does not exist. Please check the sequence name.")
        
        with open(frame_list_path, 'r') as f:
            lines = f.readlines()
            return [line.strip() for line in lines if line.strip()]
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            frame_list_dir = self.frame_list_dir_from_seq(seq)
            if self.db_modality == "rgb":
                db_file_name = "rgb_framelist.txt"
            elif self.db_modality == "thr":
                db_file_name = "thermal_framelist.txt"

            if self.q_modality == "rgb":
                q_file_name = "rgb_framelist.txt"
            elif self.q_modality == "thr":
                q_file_name = "thermal_framelist.txt"


            db_abs_paths.extend(self.read_frame_lists(frame_list_dir,db_file_name))
            q_abs_paths.extend(self.read_frame_lists(frame_list_dir,q_file_name))
            # import pdb;pdb.set_trace()
        return db_abs_paths,q_abs_paths
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        h,w = img.shape[:2]
        img = img[self.crop_top:h - self.crop_bottom, self.crop_left:w - self.crop_right]
        if img is None:
            raise ValueError(f"Image at {path} could not be read. Please check the path.")
        img = cv2.resize(img, (self.thr_res[1],self.thr_res[0]), interpolation=cv2.INTER_AREA)
        img = base_transform(img)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to grayscale
        img = base_transform(img)
        return img
    
    
    def form_gt_positives(self):
        # PARV_TODO : Implement this if GPS data is available
        self.db_coords = []
        self.q_coords = []
        # load files for the coordinates
        for seq in self.seq:
            frame_list_dir = self.frame_list_dir_from_seq(seq)

            if self.db_modality == "rgb":
                db_file_name = "rgb_gps_framelist.txt"
            elif self.db_modality == "thr":
                db_file_name = "thermal_gps_framelist.txt"
            
            if self.q_modality == "rgb":
                q_file_name = "rgb_gps_framelist.txt"
            elif self.q_modality == "thr":
                q_file_name = "thermal_gps_framelist.txt"

            db_gps_frame_list = os.path.join(frame_list_dir, db_file_name)
            q_gps_frame_list = os.path.join(frame_list_dir, q_file_name)

            if not os.path.exists(db_gps_frame_list):
                raise ValueError(f"GPS frame list {db_gps_frame_list} does not exist. Please check the sequence name.")
            if not os.path.exists(q_gps_frame_list):
                raise ValueError(f"GPS frame list {q_gps_frame_list} does not exist. Please check the sequence name.")

            with open(db_gps_frame_list) as f:
                lines = f.readlines()
                coord = [(float(x[0]), float(x[1])) for x in (line.strip().split() for line in lines)]

                self.db_coords.extend(coord)
            
            with open(q_gps_frame_list) as f:
                lines = f.readlines()
                coord = [(float(x[0]), float(x[1])) for x in (line.strip().split() for line in lines)]

                self.q_coords.extend(coord)
            
            # import pdb;pdb.set_trace()
        
        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        dist,soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= self.dist_thresh,
                                                            return_distance=True)
        return dist , soft_positives_per_query
    
    def check_seq_list(self,seq):
        """
        Checks if the sequence list is valid.
        """
        for s in seq:
            if not os.path.exists(self.seq_path(s)):
                print(f"Sequence {s} does not exist. Please check the sequence name.")
                import pdb;pdb.set_trace()
                raise ValueError(f"Sequence {s} does not exist. Please check the sequence name.")