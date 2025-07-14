from base_dataset import *
from ms2_utils import *
from pyproj import Transformer

def return_sthereo_split(split):
    """
    Returns the split for the sthereo dataset.
    """
    if split == "train":
        train_seq_list = ['snu_afternoon', 'snu_evening','snu_morning', 
                                'valley_afternoon', 'valley_evening','valley_morning'
                                ]
        return train_seq_list
    elif split == "val":
        val_seq_list = ['kaist_afternoon', 'kaist_evening', 'kaist_morning']

        return val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or test")




class STHEREO(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 15, rescale_during_crop=False,use_clahe=True, crop_during_vpr_test=False):
        self.subsample =10
        self.frame_list_dir = "/ocean/projects/cis220039p/mdt2/datasets/STHEREO/frame_lists"
        self.use_clahe = use_clahe
        self.thr_res = (284,558)
        self.crop_top = 121
        self.crop_bottom = 107
        self.crop_left = 52
        self.crop_right = 30
        self.val_positive_dist_threshold = 25


        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop, crop_during_vpr_test=crop_during_vpr_test,crop_images=crop_images)
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }

    def read_frame_lists(self,seq,idx,):
        """
        Reads frame list from the frame_list_dir.
        """
        frame_list_path = os.path.join(self.frame_list_dir,seq+"_frame_pairs.txt")
        if not os.path.exists(frame_list_path):
            raise ValueError(f"Sequence {seq} does not exist. Please check the sequence name.")
        
        with open(frame_list_path, 'r') as f:
            frame_list = f.readlines()
        
        frames = [x.strip().split()[idx] for x in frame_list]

        return frames
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            if self.db_modality == "rgb":
                db_idx = 0
                db_folder = "stereo_left"
            elif self.db_modality == "thr":
                db_idx = 1
                if self.use_clahe:
                    db_folder = "thermal8_left_clahe"
                else:
                    db_folder = "thermal8_left"
            if self.q_modality == "rgb":
                q_idx = 0
                q_folder = "stereo_left"
            elif self.q_modality == "thr":
                q_idx = 1
                if self.use_clahe:
                    q_folder = "thermal8_left_clahe"
                else:
                    q_folder = "thermal8_left"

            rel_db_paths = self.read_frame_lists(seq,db_idx)[::self.subsample]
            rel_q_paths = self.read_frame_lists(seq,q_idx)[::self.subsample]

            db_abs_paths.extend([os.path.join(self.datasets_folder,seq,"image",db_folder,x) for x in rel_db_paths])
            q_abs_paths.extend([os.path.join(self.datasets_folder,seq,"image",q_folder,x) for x in rel_q_paths])

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
            x_all = self.read_frame_lists(seq,2)[::self.subsample] # 2 is the index for coordinates in the frame list
            y_all = self.read_frame_lists(seq,3)[::self.subsample]
            z_all = self.read_frame_lists(seq,4)[::self.subsample]

            for x,y,z in zip(x_all,y_all,z_all):
                x = float(x.strip())
                y = float(y.strip())
                z = float(z.strip())
                coord = [x,y,z]
                if coord is not None:
                    self.db_coords.append(coord)
                    self.q_coords.append(coord)
                else:
                    import pdb;pdb.set_trace()
                    raise ValueError(f"Invalid coordinates: {lon}, {lat}, {alt} for sequence {seq}")
        # if 'kaist_afternoon' in self.seq:
        #     np.save(os.path.join("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc","sthereo_coords.npy"),np.array(self.db_coords))


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
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.thr_res[1],self.thr_res[0]),interpolation=cv2.INTER_AREA)  # Resize to the desired resolution
        img = base_transform(img)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h,w = img.shape[:2]
        img = img[self.crop_top:h - self.crop_bottom, self.crop_left:w - self.crop_right]
        img = base_transform(img)

        return img
