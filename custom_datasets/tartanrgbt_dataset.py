from base_dataset import *
import torchvision.transforms.functional as torch_F
import torchvision
from copy import copy

def return_tartanrgbt_split(split):
    if split == "val":
        return ['time_sync_check']
    elif split == "train":
        return ['time_sync_check']
    else:
        raise ValueError("Please provide a valid split name. Options are 'train' or 'val'")

class TartanRGBT(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 25, rescale_during_crop=False,crop_during_vpr_test=False, val_positive_dist_threshold=-1):
        self.subsample =1
        self.thr_res = [512,640] #PARV_TODO confirm
        self.thr_crop_bottom = 38
        self.thr_new_res = copy(self.thr_res)
        self.thr_new_res[0] -= self.thr_crop_bottom

        self.rgb_crop_top = 30
        self.rgb_crop_left = 106
        self.rgb_crop_right = 75
        assert crop_during_vpr_test == False, "Crop during VPR test is not supported for TartanRGBT dataset. Please set it to False."
        dist_thresh = -1
        self.positive_radius_index = 1
        self.val_extra_margin_positive_radius_index = 2
        self.neg_ring_outer_radius_index = 5
        self.location_type = 'time'
        
        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop,crop_images=crop_images)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal
        }
    
    def seq_path(self,seq):
        """
        Returns the path to the sequence.
        """
        return os.path.join(self.datasets_folder,seq)
    
    def extract_frames(self,seq_path):
        frame_list = natsorted(os.listdir(seq_path))[::self.subsample]
        return [os.path.join(seq_path,x) for x in frame_list if x.endswith('.png') or x.endswith('.jpg')]
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            seq_path = self.seq_path(seq)
            if self.db_modality == "rgb":
                db_folder = "zed_left_rect"
            elif self.db_modality == "thr":
                db_folder = "thermal_left_rect_8"
            else:
                raise ValueError("Please provide a valid db_modality. Options are 'rgb', 'thr'")

            if self.q_modality == "rgb":
                q_folder = "zed_left_rect"
            elif self.q_modality == "thr":
                q_folder = "thermal_left_rect_8"
            else:
                raise ValueError("Please provide a valid q_modality. Options are 'rgb', 'thr'")

            # for frame in frame_list:
            db_abs_paths.extend(self.extract_frames(os.path.join(seq_path,db_folder)))
            q_abs_paths.extend(self.extract_frames(os.path.join(seq_path,q_folder)))
            
        return db_abs_paths,q_abs_paths

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if img is None:
            raise ValueError(f"Image at {path} could not be read. Please check the path.")

        H, W = img.shape[:2]

        img = cv2.resize(img, (self.thr_res[1],self.thr_res[0]), interpolation=cv2.INTER_AREA)
        img = img[self.rgb_crop_top:, self.rgb_crop_left:-self.rgb_crop_right]
        # print("rgb image after cropping before resizing", img.shape)
        # crop
        
        #apply the crop box from the metadata
        img = cv2.resize(img, (self.thr_new_res[1],self.thr_new_res[0]), interpolation=cv2.INTER_AREA)
        # print("rgb image size after resizing", img.shape)
        # img = img[self.row_start:self.row_end, self.col_start:self.col_end]
        #apply center  of size 480x640 crop

        img = base_transform(img)

        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to grayscale
        # print("Original Thermal Image Size: ", img.shape)
        img = img[:-self.thr_crop_bottom]
        img = base_transform(img)
        # print("thermal new.shape", img.shape , "expected_res", self.thr_new_res)

        # img = torch_F.resize(img, (int(img.shape[-2]/2), int(img.shape[-1]/2)), antialias=True,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        # img = torch_F.resize(img, (320, img.shape[-1]), antialias=True,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        return img
    
    def form_db_qu_coords(self):
        # Form ground truth positives for the dataset. For a given index all images with indices wihting the self.positive_radius_index are considered positives.
        """
        Returns ground truth positives for the dataset.
        """
        self.db_coords = [None for _ in range(len(self.db_abs_paths))]
        self.q_coords = [None for _ in range(len(self.q_abs_paths))]
    
    def check_seq_list(self,seq):
        """
        Checks if the sequence list is valid.
        """
        for s in seq:
            if not os.path.exists(self.seq_path(s)):
                import pdb;pdb.set_trace()
                raise ValueError(f"Sequence {s} does not exist. Please check the sequence name.")
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}