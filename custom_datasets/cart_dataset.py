from .base_dataset import *

def return_cart_split_segmentation(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                "2023-03-21-09-59-39","2023-03-21-14-06-04","2023-03-21-18-20-21","2023-03-21-19-55-11",
                                # "2023-03-22-08-44-31",
                                "2023-03-22-14-31-06","2023-03-22-14-41-46",
                                '2022-12-20-11-40-28' #PARV_TODO - currently using 1 traj from the lake data - val split - figure this out
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = [ '2022-12-20-12-16-02', '2022-12-20-12-48-59','2022-12-20-13-37-37'
                             ]
        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")

def return_cart_split(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                '2022-07-26-10-39-11',"2022-07-26-10-50-52", '2022-07-26-11-00-21',"2022-07-26-11-05-36","2022-07-26-11-22-00","2022-10-06-12-23-29","2022-10-06-13-11-26",
                                "2023-03-21-09-59-39","2023-03-21-14-06-04",
                                # "2023-03-21-18-20-21","2023-03-21-19-55-11", nothing visible in RGB, PARV_TODO can use for themral only supervision 
                                # "2023-03-22-08-44-31", PARV_TODO will enable once data is extracted 
                                "2023-03-22-14-31-06","2023-03-22-14-41-46",
                                "2022-12-20-11-40-28" #PARV_TODO - currently using 1 traj from the lake data - val split - figure this out
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = ['2022-12-20-12-16-02', '2022-12-20-12-48-59','2022-12-20-13-37-37'
                             ]
        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")




class CART(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,vpr_test=False,dist_thresh = 25):
        self.root_frame_dir = root_frame_dir
        if vpr_test and len(seq) > 1:
            raise ValueError("Please provide a single sequence name since MS@ does not support combining odometry of multiple sequences. Input is a list")

        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,augment=augment,seq=seq)
        self.semantic_classes= 12

        if "seg_mask" in [self.db_modality, self.q_modality] and self.augment:
            raise ValueError("Segmentation mask cannot be used with augmentation. Please set augment to False")
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "thr_seg": self.read_thermal,
            "seg_mask" : self.read_segmentation_mask
        }
    def read_frame_list(self, frame_list_path):
        """
        Reads a list of frames from the given path.
        """
        with open(frame_list_path, 'r') as f:
            frame_list = f.readlines()
        frame_list = [x.strip() for x in frame_list]
        frame_list = [x for x in frame_list if x.endswith('.png')]
        return frame_list
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        """
        Generates image paths for the dataset. Return the updated db_abs_paths and q_abs_paths
        """
        for seq in self.seq:

            if "thr" in [self.db_modality, self.q_modality]:
                thermal_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_frame_list.txt")
                thermal_frames = self.read_frame_list(thermal_file)
            
            if "thr_seg" in [self.db_modality, self.q_modality]:
                thermal_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_frame_list_seg_pair.txt")
                thermal_frames = self.read_frame_list(thermal_file)

            if "rgb" in [self.db_modality, self.q_modality]:
                rgb_file = os.path.join(self.root_frame_dir,f"{seq}_rgb_frame_list.txt")
                rgb_frames = self.read_frame_list(rgb_file)
            
            if "seg_mask" in [self.db_modality, self.q_modality]:
                seg_mask_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_segmentation_frame_list.txt")
                seg_mask_frames = self.read_frame_list(seg_mask_file)


            if self.db_modality == "rgb":                
                db_abs_paths.extend(rgb_frames)
            elif self.db_modality == "thr" or self.db_modality == "thr_seg":
                db_abs_paths.extend(thermal_frames)
            elif self.db_modality == "seg_mask":
                db_abs_paths.extend(seg_mask_frames)
            else:
                raise ValueError("Please provide a valid db_modality. Currently only rgb and thr are supported")
            if self.q_modality == "rgb":
                q_abs_paths.extend(rgb_frames)
            elif self.q_modality == "thr" or self.q_modality == "thr_seg":
                q_abs_paths.extend(thermal_frames)
            elif self.q_modality == "seg_mask":
                q_abs_paths.extend(seg_mask_frames)
            else:
                raise ValueError("Please provide a valid q_modality. Currently only rgb and thr are supported")

        return db_abs_paths,q_abs_paths
    def check_seq_list(self,seq):
        for s in seq:
            if os.path.isdir(os.path.join(self.datasets_folder,s)) == False:
                import pdb; pdb.set_trace()  # Debugging line to inspect the sequence name
                raise ValueError(f"Please provide a valid sequence name. {s} does not exist")
    def extract_frame_number(self,filename):
        # Extract the frame number from the filename
        # Assuming the filename format is like "image_color-00001.jpg"
        # import pdb; pdb.set_trace()  # Debugging line to inspect the filename parts

        parts = filename.split('-')
        if len(parts) > 1:
            return int(parts[1].split('.')[0])  # Get the number before the file extension
        return -1  # Return -1 if no valid frame number is found


    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """
        self.db_coords = []
        self.q_coords = []
        # load files for the coordinates
        for seq in self.seq:
            img_idx_list = []
            for p in self.db_abs_paths:
                img_idx = int(self.extract_frame_number(p.split("/")[-1]))
                img_idx_list.append(img_idx)
            if os.path.exists(os.path.join(self.datasets_folder,seq,"csv/thermal_utm_coords.npy")) == False:
                raise ValueError(f"Please provide a valid sequence name. {seq} does not have thermal_utm_coords.npy file")
            all_gps_coords = np.load(os.path.join(self.datasets_folder,seq,"csv/thermal_utm_coords.npy"))
            
            for i in range(len(img_idx_list)):
                img_idx = img_idx_list[i]
                coord_x = all_gps_coords[img_idx][0]
                coord_y = all_gps_coords[img_idx][1]
                coord_z = all_gps_coords[img_idx][2]

                self.db_coords.append([coord_x,coord_y,coord_z])
                self.q_coords.append([coord_x,coord_y,coord_z])
            

    

        self.db_coords = np.array(self.db_coords)
        self.q_coords = np.array(self.q_coords)

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        dist,soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= self.dist_thresh,
                                                            return_distance=True)
        return dist , soft_positives_per_query         
    
    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        img = base_transform(img)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path)
        img = base_transform(img)
        return img
    
    def read_segmentation_mask(self, path):
        """
        Reads segmentation mask image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.expand_dims(img, axis=0)  # Add channel dimension and convert to float32
        return img