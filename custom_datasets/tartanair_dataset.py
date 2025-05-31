from .thermal_dataloader import *
import torch

class TartanAirDataset(CustomDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,seq,db_modality,q_modality,model_type="dinov2_vits14",datasets_folder='/ocean/projects/cis220039p/shared/datasets/tartanair_v2'):
        super().__init__()

        self.datasets_folder = datasets_folder
        self.db_modality = db_modality
        self.q_modality = q_modality
        self.seq = seq
        self.subsample_len = 200
        self.subsample = int(len(natsorted(os.listdir(os.path.join(self.datasets_folder,self.seq,"image_lcam_front"))))/self.subsample_len)
        self.model_type = model_type
        print("seq: ",self.seq)
        print("db_modality: ",self.db_modality)
        print("q_modality: ",self.q_modality)
        print("subsample: ",self.subsample)
        print("model_type: ",self.model_type)

        modality_to_folder_dict = {
            "rgb": "image_lcam_front",
            "depth": "depth_lcam_front"
        }

        self.db_folder = os.path.join(self.datasets_folder,self.seq,modality_to_folder_dict[self.db_modality])
        self.q_folder = os.path.join(self.datasets_folder,self.seq,modality_to_folder_dict[self.q_modality])
        if self.db_modality in modality_to_folder_dict:
            self.db_paths = natsorted(os.listdir(self.db_folder))[::self.subsample]
        else:
            raise ValueError("Invalid modality. Please choose 'rgb' or 'depth'.")
        
        if self.q_modality in modality_to_folder_dict:
            self.q_paths = natsorted(os.listdir(self.q_folder))[::self.subsample]
        else:
            raise ValueError("Invalid modality. Please choose 'rgb' or 'depth'.")
        

        self.db_abs_paths = []
        self.q_abs_paths = []

        for db_path in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.db_folder,db_path))
        for q_path in self.q_paths:
            self.q_abs_paths.append(os.path.join(self.q_folder,q_path))
        
        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)

        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.form_gt_positives()

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """

        # load files for the coordinates
        
        self.db_coords = []
        self.q_coords = []


        # for p in self.db_coord_paths:          
        with open(os.path.join(self.datasets_folder,self.seq,"pose_lcam_front.txt")) as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split()
                coord_x = float(elements[0])
                coord_y = float(elements[1])
                coord_z = float(elements[2])

                self.db_coords.append([coord_x,coord_y,coord_z])
                self.q_coords.append([coord_x,coord_y,coord_z])

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        self.dist,self.soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= 10,
                                                            return_distance=True)            

    
    def read_rgb(self, imgpath, scale = 1):
        '''
        copied from tartanairpy.tartanair.reader. This is used by TartanAirImageDatasetObject
        '''
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None or img.size==0:
            return None
        if scale != 1:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        h  = img.shape[0]
        w = img.shape[1]
        if self.model_type == "clip":
            img = cv2.resize(img, (224, 224))
        else:
            img = cv2.resize(img, ((w//14)*14, (h//14)*14))
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).to(torch.float32)
    def visdepth(self, depth, max_depth=500):
        depthvis = np.clip(max_depth/depth ,0 ,255)
        depthvis = depthvis.astype(np.uint8)
        depthvis = cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)
        depthvis = cv2.cvtColor(depthvis, cv2.COLOR_RGB2BGR)
        return depthvis
    
    def depth_rgba_float32(self, depth_rgba):
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def read_depth(self, depthpath):
        '''
        copied from tartanairpy.tartanair.reader. This is used by TartanAirImageDatasetObject
        '''
        if depthpath.endswith('npy'):
            depth = np.load(depthpath)
        else:
            depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
            if depth_rgba is None:
                return None
            depth = self.depth_rgba_float32(depth_rgba)
            depth = self.visdepth(depth)
            h  = depth.shape[0]
            w = depth.shape[1]
            if self.model_type == "clip":
                depth = cv2.resize(depth, (224, 224))
            else:
                depth = cv2.resize(depth, ((w//14)*14, (h//14)*14))
            depth = depth.transpose(2, 0, 1)
        return torch.from_numpy(depth).to(torch.float32)
    
    def __getitem__(self, index):

        if index>=self.database_num:
            if self.q_modality == "rgb":
                img = self.read_rgb(self.images_paths[index])
            elif self.q_modality == "depth":
                img = self.read_depth(self.images_paths[index])
            else:
                raise ValueError("Invalid modality. Please choose 'rgb' or 'depth'.")
            

        elif index<self.database_num:

            if self.db_modality == "rgb":
                img = self.read_rgb(self.images_paths[index])
            elif self.db_modality == "depth":
                img = self.read_depth(self.images_paths[index])
            else:
                raise ValueError("Invalid modality. Please choose 'rgb' or 'depth'.")

        return img, index
