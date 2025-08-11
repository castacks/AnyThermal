from base_dataset import *
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Tuple

def return_pst900_split(split):
    """
    Returns the split for the ms2 dataset.
    """

    if split =="train":
        return ['train']
    elif split == "test":
        return ['test']
    else:
        raise ValueError("Please provide a valid split name. Options are train, val or test")

class PST900(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,root_frame_dir,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = -1,rescale_during_crop=False,crop_during_vpr_test=False,val_positive_dist_threshold=-1):
        
        self.root_frame_dir = "/ocean/projects/cis220039p/mdt2/datasets/PST900_RGBT_Dataset"
        seq = [os.path.join(self.root_frame_dir, x) for x in seq]
        self.val_positive_dist_threshold = -1
        dist_thresh = -1
        self.dataset_shape = (720,1280)
        assert vpr_train == False, "VPR train is not supported for MFNet dataset"
        assert vpr_test == False, "VPR test is not supported for MFNet dataset"
        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,augment=augment,seq=seq,rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "seg_mask" : self.read_segmentation_mask
        }
    def generate_frame_list(self, folder):
        """
        Reads a list of frames from the given path.
        """
        image_names = sorted(os.listdir(os.path.join(folder, "rgb")))

        rgb_frame_list = [os.path.join(folder, "rgb",x) for x in image_names]
        thr_frame_list = [os.path.join(folder, "thermal",x) for x in image_names]
        seg_mask_frame_list = [os.path.join(folder, "labels",x) for x in image_names]

        return rgb_frame_list, thr_frame_list, seg_mask_frame_list

    def generate_seq_as_txt_paths(self,db_abs_paths,q_abs_paths):
        """
        Generates image paths for the dataset in segmentation mode. Return the updated db_abs_paths and q_abs_paths
        """
        all_rgb_frames = []
        all_thermal_frames = []
        all_seg_frames = []

        for seq in self.seq:
            rgb_frames,thermal_frames,seg_frames = self.generate_frame_list(seq)
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
        elif self.db_modality == "thr":
            db_abs_paths.extend(thermal_paths)
        elif self.db_modality == "seg_mask":
            db_abs_paths.extend(mask_paths)
        else:
            raise ValueError("Please provide a valid db_modality. Currently only rgb and thr are supported")
        
        if self.q_modality == "rgb":
            q_abs_paths.extend(rgb_paths)
        elif self.q_modality == "thr":
            q_abs_paths.extend(thermal_paths)
        elif self.q_modality == "seg_mask":
            q_abs_paths.extend(mask_paths)
        else:
            raise ValueError("Please provide a valid q_modality. Currently only rgb and thr are supported")
        
        return db_abs_paths, q_abs_paths

    def check_seq_list(self,seq):
        for s in seq:
            if os.path.isdir(s) == False:
                import pdb; pdb.set_trace()
                raise ValueError(f"Please provide a valid sequence name. {s} does not exist or is not a file")


    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """
        raise NotImplementedError("Please implement the form_gt_positives method in the subclass")    
    
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
        img = cv2.imread(path,0)
        img = self.fill_thermal(img)  # Fill holes in the thermal image
        img = base_transform(img)
        return img
    
    def read_segmentation_mask(self, path):
        """
        Reads segmentation mask image from the path.
        """
        img = cv2.imread(path, -1).astype(np.int8)  # Read as grayscale
        img = base_transform(img)
        # img = img-1 # removing the unlabelled class
        return img
    
    def fill_thermal(self, thermal_image):
        """
        Example hole filling of thermal image
        """
        hole_mask = (thermal_image == 0).astype(np.uint8)
        filled_thermal = cv2.inpaint(
            thermal_image, 
            hole_mask, 
            10, 
            cv2.INPAINT_TELEA
        )
        return filled_thermal
    
    def semantic_classes_num_and_map_to_rgb(self):
        ID_TO_RGB = {
            0: (0, 0, 0),           
            1: (0,0,255),
            2: (0,255,0),
            3: (255,0,0),
            4: (255,255,255),        
        }
        return 5,ID_TO_RGB
    
    def skip_classes_in_segmentation(self):
        """
        Returns the classes to skip in segmentation.
        """
        return [0]