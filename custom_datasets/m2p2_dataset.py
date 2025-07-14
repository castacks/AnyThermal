from base_dataset import *
from pyproj import Transformer
from natsort import natsorted

def return_m2p2_split(split,rgb_only=False):
    """
    Returns the split for the sthereo dataset.
    """
    seq_list = os.listdir("/ocean/projects/cis220039p/mdt2/datasets/M2P2/extracted_data_new")
    train_seq_list = natsorted([x for x in seq_list if ("LBR" not in x) and ("MC" not in x)])
    val_seq_list = natsorted([x for x in seq_list if ("LBR" in x) or ("MC" in x)])
    if rgb_only:
        final_train_seq_list = []
        flag = False
        for seq in train_seq_list:
            if not flag:
                if seq == "BL_2024-09-04_19-36-36_chunk0002":
                    flag = True
                final_train_seq_list.append(seq)
            else:
                break
            
        train_seq_list = final_train_seq_list
        val_seq_list = final_train_seq_list #PARV_DEBUG what to use for val since rgb is very small and only in one area
    if split == "train":
        return train_seq_list
    elif split == "val":
        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or test")


class M2P2(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 25, rescale_during_crop=False,use_clahe=True, crop_during_vpr_test=False):
        self.subsample =10
        self.thr_res = (490, 650)
        self.crop_top = 302
        self.crop_bottom = 232
        self.crop_left = 353
        self.crop_right = 277

        if vpr_train or vpr_test:
            seq_filtered = [seq_single for seq_single in seq if "NOGPS" not in seq_single]
            if len(seq_filtered) == 0:
                raise ValueError(f"No sequences with GPS data found. Please provide a valid sequence name., current seq is {seq}")
            if len(seq_filtered) < len(seq):
                # print(f"Filtered sequences with GPS data: {seq_filtered}. Original sequences: {seq}")
                for seq_single in seq:
                    if seq_single not in seq_filtered:
                        print(f"Sequence {seq_single} does not have GPS data and is filtered out.")
        else:
            seq_filtered = seq

        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq_filtered,augment = augment, rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }
    
    def get_image_paths(self,folder):
        return [os.path.join(folder,x) for x in natsorted(os.listdir(folder))][::self.subsample]
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            if self.db_modality == "rgb":
                db_folder = "left_rgb"
            elif self.db_modality == "thr":                
                db_folder = "thermal"
            if self.q_modality == "rgb":
                q_folder = "left_rgb"
            elif self.q_modality == "thr":
                q_folder = "thermal"

            db_abs_paths.extend(self.get_image_paths(os.path.join(self.datasets_folder,seq,db_folder)))
            q_abs_paths.extend(self.get_image_paths(os.path.join(self.datasets_folder,seq,q_folder)))

        return db_abs_paths,q_abs_paths
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}

    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """
        self.db_coords = []
        self.q_coords = []
        # load files for the coordinates
        for seq in self.seq:

            gps_file = os.path.join(self.datasets_folder,seq,"gps.npy")
            self.db_coords.extend(list(np.load(gps_file))[::self.subsample])
            self.q_coords.extend(list(np.load(gps_file))[::self.subsample])

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        dist,soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= self.dist_thresh,
                                                            return_distance=True)
        return dist , soft_positives_per_query         
    def check_seq_list(self,seq):
        for s in seq:
            if os.path.isdir(os.path.join(self.datasets_folder,s)) == False:
                import pdb;pdb.set_trace()
                raise ValueError(f"Please provide a valid sequence name. {s} does not exist")

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        # print("Reading rgb image from path: ", path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.thr_res[1],self.thr_res[0]),interpolation=cv2.INTER_AREA)  # Resize to the desired resolution
        img = base_transform(img)
        # print("rgb image read shapwed: ", img.shape)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        # print("Reading thermal image from path: ", path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h,w = img.shape[:2]
        img = img[self.crop_top:h - self.crop_bottom, self.crop_left:w - self.crop_right]
        img = base_transform(img)
        # print("thermal image read shapwed: ", img.shape)

        return img
