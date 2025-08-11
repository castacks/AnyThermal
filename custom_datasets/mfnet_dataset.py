from base_dataset import *
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Tuple

def return_mfnet_split(split):
    """
    Returns the split for the ms2 dataset.
    """

    if split =="train":
        return ['train.txt']
    elif split == "val":
        return ['val.txt']
    elif split == "test":
        return ['test.txt']
    else:
        raise ValueError("Please provide a valid split name. Options are train, val or test")

class MFNet(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = -1,rescale_during_crop=False,crop_during_vpr_test=False,val_positive_dist_threshold=-1):
        
        self.root_frame_dir = "/ocean/projects/cis220039p/mdt2/datasets/MFNet"
        seq = [os.path.join(self.root_frame_dir, x) for x in seq]
        self.val_positive_dist_threshold = -1
        dist_thresh = -1
        self.dataset_shape = (480,640) # (w,h)
        assert vpr_train == False, "VPR train is not supported for MFNet dataset"
        assert vpr_test == False, "VPR test is not supported for MFNet dataset"
        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,augment=augment,seq=seq,rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "seg_mask" : self.read_segmentation_mask
        }
    def read_frame_list(self, frame_list_path):
        """
        Reads a list of frames from the given path.
        """
        with open(os.path.join(frame_list_path), 'r') as f:
            frame_list = f.readlines()
        frame_list = [x.strip() for x in frame_list]
        frame_list = [x+".png" for x in frame_list]

        rgb_frame_list = [os.path.join(self.root_frame_dir, "images",x) for x in frame_list]
        thr_frame_list = [os.path.join(self.root_frame_dir, "images",x) for x in frame_list]
        seg_mask_frame_list = [os.path.join(self.root_frame_dir, "labels",x) for x in frame_list]

        return rgb_frame_list, thr_frame_list, seg_mask_frame_list

    def generate_seq_as_txt_paths(self,db_abs_paths,q_abs_paths):
        """
        Generates image paths for the dataset in segmentation mode. Return the updated db_abs_paths and q_abs_paths
        """
        all_rgb_frames = []
        all_thermal_frames = []
        all_seg_frames = []

        for seq in self.seq:
            rgb_frames,thermal_frames,seg_frames = self.read_frame_list(seq)
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

        rgb_paths, thermal_paths, mask_paths = self.generate_seq_as_txt_paths(db_abs_paths, q_abs_paths)
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

    def check_seq_list(self,seq):
        for s in seq:
            if os.path.isfile(s) == False:
                import pdb; pdb.set_trace()
                raise ValueError(f"Please provide a valid sequence name. {s} does not exist or is not a file")


    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """
        raise NotImplementedError("Please implement the form_gt_positives method in the subclass")    
    
    def read_image(self, path):
        image = np.array(Image.open(path)) # (w,h,c)
        image.flags.writeable = True
        return image

    
    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = self.read_image(path)
        img = img[...,:3]
        img = base_transform(img)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = self.read_image(path)
        img = img[...,3]
        img = base_transform(img)
        return img
    
    def read_segmentation_mask(self, path):
        """
        Reads segmentation mask image from the path.
        """
        img = self.read_image(path).astype(np.int32)
        img = base_transform(img)
        # img = img-1 # removing the unlabelled class
        return img
    
    def semantic_classes_num_and_map_to_rgb(self):
        ID_TO_RGB = {
            0: (0, 0, 0),           # unlabelled
            1: (64,0,128),     # car
            2: (64,64,0),        # person
            3: (0,128,192),     # bike
            4: (0,0,192),        # curve
            5: (128,128,0),       # car_stop
            6: (64,64,128),        # guardrail
            7: (192,128,128),       # color_cone
            8: (192,64,0),      # bump
        }
        # unlabelled = [0,0,0]
        # car        = [64,0,128]
        # person     = [64,64,0]
        # bike       = [0,128,192]
        # curve      = [0,0,192]
        # car_stop   = [128,128,0]
        # guardrail  = [64,64,128]
        # color_cone = [192,128,128]
        # bump       = [192,64,0]
        return 9,ID_TO_RGB

    def skip_classes_in_segmentation(self):
        """
        Returns the classes to skip in segmentation.
        """
        return [0]