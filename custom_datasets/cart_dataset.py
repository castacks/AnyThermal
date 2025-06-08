from .base_dataset import *
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Tuple

def return_cart_split_segmentation(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train_easy":
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
    elif split == "train":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                "2023-03-21-09-59-39","2023-03-21-14-06-04","2023-03-21-18-20-21","2023-03-21-19-55-11",
                                # "2023-03-22-08-44-31",
                                "2023-03-22-14-31-06","2023-03-22-14-41-46",
                                # '2022-12-20-11-40-28' #PARV_TODO - currently using 1 traj from the lake data - val split - figure this out
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
    if split == "train_easy":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                '2022-07-26-10-39-11',"2022-07-26-10-50-52", '2022-07-26-11-00-21',"2022-07-26-11-05-36","2022-07-26-11-22-00","2022-10-06-12-23-29","2022-10-06-13-11-26",
                                "2023-03-21-09-59-39","2023-03-21-14-06-04",
                                # "2023-03-21-18-20-21","2023-03-21-19-55-11", nothing visible in RGB, PARV_TODO can use for themral only supervision 
                                "2023-03-22-08-44-31",
                                "2023-03-22-14-31-06","2023-03-22-14-41-46",
                                "2022-12-20-11-40-28" #PARV_TODO - currently using 1 traj from the lake data - val split - figure this out
                                ]
        return train_seq_list
    elif split == "train":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                '2022-07-26-10-39-11',"2022-07-26-10-50-52", '2022-07-26-11-00-21',"2022-07-26-11-05-36","2022-07-26-11-22-00","2022-10-06-12-23-29","2022-10-06-13-11-26",
                                "2023-03-21-09-59-39","2023-03-21-14-06-04",
                                # "2023-03-21-18-20-21","2023-03-21-19-55-11", nothing visible in RGB, PARV_TODO can use for themral only supervision 
                                "2023-03-22-08-44-31",
                                "2023-03-22-14-31-06","2023-03-22-14-41-46",
                                # "2022-12-20-11-40-28" #PARV_TODO - currently using 1 traj from the lake data - val split - figure this out
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = ['2022-12-20-12-16-02', '2022-12-20-12-48-59','2022-12-20-13-37-37'
                             ]
        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")

def seq_has_gps(seq, datasets_folder):
    """
    Checks if the sequence has GPS data.
    """
    gps_file = os.path.join(datasets_folder, seq, "csv/thermal_utm_coords.npy")
    return os.path.exists(gps_file)


resize_transform = T.Compose([
    T.Resize((300, 450), interpolation=T.InterpolationMode.NEAREST),  # Resize to 300x450
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # normalization is done in the model since it can be different for each modal
])


class CART(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,vpr_test=False,vpr_train=False,dist_thresh = 25):
        self.root_frame_dir = root_frame_dir
        if vpr_train or vpr_test:
            seq_filtered = [seq_single for seq_single in seq if seq_has_gps(seq_single, datasets_folder)]
            if len(seq_filtered) == 0:
                raise ValueError(f"No sequences with GPS data found. Please provide a valid sequence name., current seq is {seq}")
            if len(seq_filtered) < len(seq):
                # print(f"Filtered sequences with GPS data: {seq_filtered}. Original sequences: {seq}")
                for seq_single in seq:
                    if seq_single not in seq_filtered:
                        print(f"Sequence {seq_single} does not have GPS data and is filtered out.")
        else:
            seq_filtered = seq
        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,augment=augment,seq=seq_filtered)
    
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
        self.seq_wise_length=[]
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

            self.seq_wise_length.append(len(rgb_frames))  # Assuming rgb_frames and thermal_frames are of the same length

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
        global_counter = 0
        for seq_idx,seq in enumerate(self.seq):
            seq_len = self.seq_wise_length[seq_idx]
            img_idx_list = []
            for local_counter in range(global_counter, global_counter + seq_len):
                path = self.db_abs_paths[local_counter]
                if seq not in path:
                    import pdb; pdb.set_trace()
                    raise ValueError(f"Path {path} does not belong to sequence {seq}. Please check the paths.")
                img_idx = int(self.extract_frame_number(path.split("/")[-1]))
                img_idx_list.append(img_idx)
            if os.path.exists(os.path.join(self.datasets_folder,seq,"csv/thermal_utm_coords.npy")) == False:
                raise ValueError(f"Please provide a valid sequence name. {seq} does not have thermal_utm_coords.npy file")
            all_gps_coords = np.load(os.path.join(self.datasets_folder,seq,"csv/thermal_utm_coords.npy"))
            for i in range(len(img_idx_list)):
                img_idx = img_idx_list[i]
                if img_idx >= len(all_gps_coords):
                    import pdb; pdb.set_trace()  # Debugging line to inspect the index and coordinates
                coord_x = all_gps_coords[img_idx][0]
                coord_y = all_gps_coords[img_idx][1]
                coord_z = all_gps_coords[img_idx][2]

                self.db_coords.append([coord_x,coord_y,coord_z])
                self.q_coords.append([coord_x,coord_y,coord_z])
            global_counter += seq_len
    

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
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = torch.tensor(img).float()  # Convert to tensor
        # print(f"Segmentation mask shape: {img.shape}")  # Debugging line to check the shape
        return img
    
    def augment_function(self, modality1: str, modality2: str, img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if modality1 == "thr" and modality2 == "rgb":
            img2, img1 = self.rgb_thermal_augment(img2, img1)
        elif modality1 == "rgb" and modality2 == "thr":
            img1, img2 = self.rgb_thermal_augment(img1, img2)
        elif modality1 == "thr_seg" and modality2 == "seg_mask":
            img1, img2 = self.thermal_seg_augment(img1,img2)
        else:
            raise ValueError(f"Unsupported modality combination: {modality1}, {modality2}")
        return img1, img2  # No augmentation if modalities are not recognized
    
    def thermal_seg_augment(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augments thermal and segmentation mask images (tensor format only).
        Args:
            img1 (Tensor): Thermal image tensor (C, H, W) or (1, H, W)
            img2 (Tensor): Segmentation mask tensor (1, H, W) or (H, W)
        Returns:
            Tuple of augmented tensors (img1, img2)
        """

        # Ensure both images are 3D tensors (C, H, W)
        if img1.ndim == 2:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 2:
            img2 = img2.unsqueeze(0)

        # Random horizontal flip
        if random.random() > 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        # ----- Random Resized Crop -----
        _, H, W = img1.shape
        scale = (0.8, 1.0)
        ratio = (0.75, 1.33)

        crop_params = T.RandomResizedCrop.get_params(img1, scale=scale, ratio=ratio)
        i, j, h, w = crop_params

        img1 = F.resized_crop(img1, i, j, h, w, size=(300, 450), interpolation=F.InterpolationMode.BILINEAR)
        img2 = F.resized_crop(img2, i, j, h, w, size=(300, 450), interpolation=F.InterpolationMode.NEAREST) # Use nearest for segmentation mask

        # Brightness and contrast (thermal input only)
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)

        img1 = F.adjust_brightness(img1, brightness_factor)
        img1 = F.adjust_contrast(img1, contrast_factor)

        return img1, img2
    def semantic_classes_num_and_map_to_rgb(self):
        ID_TO_RGB = {
            0: (255, 36, 0),        # Unknown
            1: (0, 0, 0),           # Background
            2: (242, 216, 196),     # Bare ground
            3: (89, 70, 54),        # Rocky terrain
            4: (166, 166, 166),     # Developed structures
            5: (82, 89, 90),        # Road
            6: (155, 230, 0),       # Shrubs
            7: (0, 138, 53),        # Trees
            8: (0, 216, 245),       # Sky
            9: (13, 127, 252),      # Water
            10: (255, 249, 0),      # Vehicles
            11: (254, 0, 170),      # Person
        }
        return 12,ID_TO_RGB