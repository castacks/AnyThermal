from .base_dataset import *
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Tuple
    
def return_cart_split_segmentation_random(root_dir,split):
    """
    Returns the split for the ms2 dataset.
    """
    if split =="train":
        return [os.path.join(root_dir,"train.txt")]
    elif split == "val":
        return [os.path.join(root_dir,"val.txt")]
    elif split == "test":
        return [os.path.join(root_dir,"test.txt")]

def return_cart_split(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train_easy":
        raise ValueError("train_easy split is not supported for the CART dataset. Please use train or val split.")
    elif split == "train":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                '2022-05-15-06-00-09', '2022-05-15-06-14-42','2022-05-15-06-26-50', '2022-05-15-06-39-43', 
                                '2022-07-26-10-39-11',"2022-07-26-10-50-52", '2022-07-26-11-00-21',"2022-07-26-11-05-36","2022-07-26-11-22-00","2022-10-06-12-23-29","2022-10-06-13-11-26",
                                '2022-12-20-11-40-28','2022-12-20-12-16-02', '2022-12-20-12-48-59','2022-12-20-13-37-37',
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = ["2023-03-21-09-59-39","2023-03-21-14-06-04","2023-03-22-08-44-31","2023-03-22-14-31-06","2023-03-22-14-41-46"]

        # "2023-03-21-18-20-21","2023-03-21-19-55-11", nothing visible in RGB, PARV_TODO can use for themral only supervision 

        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")


def return_handheld_cart_split(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train_easy":
        raise ValueError("train_easy split is not supported for the CART dataset. Please use train or val split.")
    elif split == "train":
        raise ValueError("train split is not supported for the CART dataset. Please use val split.")
    elif split == "val":
        train_seq_list = ['2022-04-03-12-16-33', '2022-04-03-12-20-57', 
                                '2022-04-03-17-12-07', '2022-04-03-17-16-17', 
                                '2022-05-08-11-30-40','2022-05-08-11-34-00', '2022-05-08-11-37-09',
                                ]
        return train_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")


def return_cart_split_debug(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train_easy":
        raise ValueError("train_easy split is not supported for the CART dataset. Please use train or val split.")
    elif split == "train":
        train_seq_list = ['2022-12-20-11-40-28'
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = ['2022-12-20-11-40-28']

        # "2023-03-21-18-20-21","2023-03-21-19-55-11", nothing visible in RGB, PARV_TODO can use for themral only supervision 

        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or val")


def seq_has_gps(args,seq, datasets_folder):
    """
    Checks if the sequence has GPS data.
    """
    if args.not_filter_on_gps:
        return True
    gps_file = os.path.join(datasets_folder, seq, "csv/thermal_utm_coords.npy")
    return os.path.exists(gps_file)


class CART(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 15,rescale_during_crop=False,crop_during_vpr_test=False,seq_as_txt="",cart_split="",val_positive_dist_threshold=25,neg_ring_outer_radius = 40,not_filter_on_gps=False):
        
        self.root_frame_dir = root_frame_dir
        self.seq_as_txt = seq_as_txt
        self.val_positive_dist_threshold = val_positive_dist_threshold
        self.neg_ring_outer_radius = neg_ring_outer_radius
        self.cart_split = cart_split
        args.not_filter_on_gps = not_filter_on_gps
        self.dataset_shape = (512,640) # (w,h)
        self.location_type = 'gps'
        if cart_split  =="vpr":
            self.vpr_resize = [300,480]
        else:
            self.vpr_resize = None

        assert self.seq_as_txt in ['', 'thermal', 'rgbt'], "Please provide a valid seq_as_txt. Currently only thermal and rgbt are supported"

        if vpr_train or vpr_test:
            seq_filtered = [seq_single for seq_single in seq if seq_has_gps(args,seq_single, datasets_folder)]
            if len(seq_filtered) == 0:
                raise ValueError(f"No sequences with GPS data found. Please provide a valid sequence name., current seq is {seq}")
            if len(seq_filtered) < len(seq):
                # print(f"Filtered sequences with GPS data: {seq_filtered}. Original sequences: {seq}")
                for seq_single in seq:
                    if seq_single not in seq_filtered:
                        print(f"Sequence {seq_single} does not have GPS data and is filtered out.")
        else:
            seq_filtered = seq
        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,augment=augment,seq=seq_filtered,rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "thr_seg": self.read_thermal,
            "seg_mask" : self.read_segmentation_mask
        }
    def read_frame_list(self, frame_list_path, root_dir):
        """
        Reads a list of frames from the given path.
        """
        with open(frame_list_path, 'r') as f:
            frame_list = f.readlines()
        frame_list = [x.strip() for x in frame_list]
        frame_list = [os.path.join(root_dir,x) for x in frame_list if x.endswith('.png')]
        return frame_list

    def read_segmentation_frame_list(self, seq,mode):
        """
        Reads a list of frames from the given path.
        """
        with open(seq, 'r') as f:
            frame_list = f.readlines()
        frame_list = [x.strip() for x in frame_list]
        frame_list = [x.split(",") for x in frame_list]

        for frame in frame_list:
            for i in range(len(frame)):
                frame[i] = os.path.join(self.datasets_folder, frame[i])            

        if mode == "thermal":
            thermal_frames = [x[0] for x in frame_list]
            seg_frames = [x[1] for x in frame_list]
            return None,thermal_frames,seg_frames
        else:
            rgb_frames = [x[0] for x in frame_list]
            thermal_frames = [x[1] for x in frame_list]
            seg_frames = [x[2] for x in frame_list]
            return rgb_frames,thermal_frames,seg_frames
    def generate_seq_as_txt_paths(self,mode):
        """
        Generates image paths for the dataset in segmentation mode.
        """
        all_rgb_frames = []
        all_thermal_frames = []
        all_seg_frames = []

        for seq in self.seq:
            rgb_frames,thermal_frames,seg_frames = self.read_segmentation_frame_list(seq,mode)
            if len(thermal_frames) != len(seg_frames):
                raise ValueError(f"Please provide a valid sequence name. {seq} does not have matching thermal and segmentation frames")
            if not thermal_frames or not seg_frames:
                raise ValueError(f"Please provide a valid sequence name. {seq} does not have any frames in the segmentation mode")
            if rgb_frames is not None and len(rgb_frames) != len(thermal_frames):
                raise ValueError(f"Please provide a valid sequence name. {seq} does not have matching rgb and thermal frames")
            if rgb_frames is not None:
                all_rgb_frames.extend(rgb_frames)
            all_thermal_frames.extend(thermal_frames)
            all_seg_frames.extend(seg_frames)
        
        if len(all_thermal_frames) != len(all_seg_frames):
            raise ValueError("Please provide a valid sequence name. The number of thermal frames and segmentation frames do not match")
        
        return all_rgb_frames,all_thermal_frames, all_seg_frames
        

    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        """
        Generates image paths for the dataset. Return the updated db_abs_paths and q_abs_paths
        """

        if self.seq_as_txt != "":
            rgb_paths, thermal_paths, mask_paths = self.generate_seq_as_txt_paths(self.seq_as_txt)
            if self.db_modality == "rgb":
                db_abs_paths.extend(rgb_paths)
            elif self.db_modality == "thr" or self.db_modality == "thr_seg":
                db_abs_paths.extend(thermal_paths)
            elif self.db_modality == "seg_mask":
                db_abs_paths.extend(mask_paths)
            else:
                raise ValueError("Please provide a valid db_modality. Currently only rgb and thr are supported")
            
            if self.q_modality == "rgb":
                q_abs_paths.extend(rgb_paths)
            elif self.q_modality == "thr" or self.q_modality == "thr_seg":
                q_abs_paths.extend(thermal_paths)
            elif self.q_modality == "seg_mask":
                q_abs_paths.extend(mask_paths)
            else:
                raise ValueError("Please provide a valid q_modality. Currently only rgb and thr are supported")
            
            return db_abs_paths, q_abs_paths
        else:
            self.seq_wise_length=[]
            rgb_frames = None
            thermal_frames = None
            seg_mask_frames = None
            for seq in self.seq:

                if "thr" in [self.db_modality, self.q_modality]:
                    thermal_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_frame_list.txt")
                    thermal_frames = self.read_frame_list(thermal_file,os.path.join(self.datasets_folder,"bag_files"))
                
                if "thr_seg" in [self.db_modality, self.q_modality]:
                    thermal_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_frame_list_seg_pair.txt")
                    thermal_frames = self.read_frame_list(thermal_file,os.path.join(self.datasets_folder,"labeled_thermal_singles"))
                
                if "rgb" in [self.db_modality, self.q_modality]:
                    rgb_file = os.path.join(self.root_frame_dir,f"{seq}_rgb_frame_list.txt")
                    rgb_frames = self.read_frame_list(rgb_file,os.path.join(self.datasets_folder,"bag_files"))
                    len_frames = len(rgb_frames)
                if "seg_mask" in [self.db_modality, self.q_modality]:
                    seg_mask_file = os.path.join(self.root_frame_dir,f"{seq}_thermal_segmentation_frame_list.txt")
                    seg_mask_frames = self.read_frame_list(seg_mask_file , os.path.join(self.datasets_folder,"labeled_thermal_singles"))
                if thermal_frames is None and rgb_frames is None:
                    raise ValueError(f"Please provide a valid sequence name. {seq} does not have any frames in the segmentation mode")
                if rgb_frames is not None:
                    self.seq_wise_length.append(len(rgb_frames))  # Assuming rgb_frames and thermal_frames are of the same length
                elif thermal_frames is not None:
                    self.seq_wise_length.append(len(thermal_frames))
                else:
                    raise ValueError(f"Please provide a valid sequence name. {seq} does not have any frames in the segmentation mode")
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
            if not self.seq_as_txt:
                if os.path.isdir(os.path.join(self.datasets_folder,"bag_files",s)) == False:
                    import pdb; pdb.set_trace()  # Debugging line to inspect the sequence name
                    raise ValueError(f"Please provide a valid sequence name. {s} does not exist")
            else:
                if os.path.isfile(s) == False:
                    import pdb; pdb.set_trace()
                    raise ValueError(f"Please provide a valid sequence name. {s} does not exist or is not a file")
    def extract_frame_number(self,filename):
        parts = filename.split('-')
        if len(parts) > 1:
            return int(parts[1].split('.')[0])  # Get the number before the file extension
        return -1  # Return -1 if no valid frame number is found


    def form_db_qu_coords(self):
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

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = base_transform(img)
        if self.vpr_resize is not None:
            img = F.resize(img, self.vpr_resize, interpolation=T.InterpolationMode.BILINEAR, antialias=True)

        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path)
        img = base_transform(img)
        if self.vpr_resize is not None:
            img = F.resize(img, self.vpr_resize, interpolation=T.InterpolationMode.BILINEAR, antialias=True)

        return img
    
    def read_segmentation_mask(self, path):
        """
        Reads segmentation mask image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = torch.tensor(img).float()  # Convert to tensor
        img = img-2 # removing the background and unknown class
        return img
    
    def semantic_classes_num_and_map_to_rgb(self):
        ID_TO_RGB = {
            # 0: (255, 36, 0),        # Unknown
            # 1: (0, 0, 0),           # Background
            0: (242, 216, 196),     # Bare ground
            1: (89, 70, 54),        # Rocky terrain
            2: (166, 166, 166),     # Developed structures
            3: (82, 89, 90),        # Road
            4: (155, 230, 0),       # Shrubs
            5: (0, 138, 53),        # Trees
            6: (0, 216, 245),       # Sky
            7: (13, 127, 252),      # Water
            8: (255, 249, 0),      # Vehicles
            9: (254, 0, 170),      # Person
        }
        return 10,ID_TO_RGB
    
    def skip_classes_in_segmentation(self):
        """
        Returns the classes to skip in segmentation.
        """
        # Skip unknown and background classes
        return []


class HandheldCART(CART):
    def __init__(self,args,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 15,rescale_during_crop=False,crop_during_vpr_test=False,seq_as_txt="",cart_split="",val_positive_dist_threshold=-1):
        
        not_filter_on_gps = True
        self.positive_radius_index = 3
        self.val_extra_margin_positive_radius_index = 6
        self.neg_ring_outer_radius_index = 10
        dist_thresh = -1
        super().__init__(args,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test,vpr_train,dist_thresh,rescale_during_crop,crop_during_vpr_test,seq_as_txt,cart_split,val_positive_dist_threshold,not_filter_on_gps=not_filter_on_gps)
        self.location_type = 'time'
    def form_db_qu_coords(self):
        """
        Returns ground truth positives for the dataset.
        """
        self.db_coords = [None for _ in range(len(self.db_abs_paths))]
        self.q_coords = [None for _ in range(len(self.q_abs_paths))]