from .base_dataset import *
from .ms2_utils import *
from pyproj import Transformer

def return_ms2_split(split):
    """
    Returns the split for the ms2 dataset.
    """
    if split == "train":
        ms2_train_seq_list = ['_2021-08-06-10-59-33', '_2021-08-06-11-23-45', 
                                '_2021-08-06-16-19-00', '_2021-08-06-16-59-13', '_2021-08-06-17-44-55',
                                '_2021-08-13-15-46-56', '_2021-08-13-16-08-46', '_2021-08-13-16-31-10', '_2021-08-13-17-06-04',
                                ]
        return ms2_train_seq_list
    elif split == "val":
        # "_2021-08-06-16-45-28 _2021-08-06-11-37-46 _2021-08-06-17-21-04 _2021-08-13-16-08-46 _2021-08-13-22-03-03 _2021-08-13-21-58-13"
        ms2_val_seq_list = ['_2021-08-06-11-37-46', '_2021-08-06-16-45-28', '_2021-08-06-17-21-04',
                            '_2021-08-13-16-08-46', '_2021-08-13-22-03-03', '_2021-08-13-21-58-13',
                            ]
        return ms2_val_seq_list
    else:
        raise ValueError("Please provide a valid split name. Options are train or test")




class MS2(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,db_modality,q_modality,datasets_folder,seq,augment,vpr_test=False,vpr_train=False,dist_thresh = 25, rescale_during_crop=False):
        self.subsample =10
        self.zone = 52
        self.utm_transformer = self.get_utm_transformer(self.zone)
        # lon = float("36.366181493439412975")
        # lat = float("127.365607591950677602")
        # easting, northing = self.utm_transformer.transform(lon, lat)
        # # print("Easting:", easting)
        # # print("Northing:", northing)
        if vpr_test and len(seq) > 1:
            raise ValueError("Please provide a single sequence name since MS2 does not support combining odometry of multiple sequences for a VPR test. Input is a list")

        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=seq,augment = augment, rescale_during_crop=rescale_during_crop)
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
            "lidar": self.read_lidar
        }
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            if self.db_modality == "rgb":
                rel_db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"rgb/img_left")))[::self.subsample]
                db_abs_paths.extend([os.path.join(self.datasets_folder,"sync_data",seq,"rgb/img_left",x) for x in rel_db_paths])
            elif self.db_modality == "thr":
                rel_db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"thr/img_left")))[::self.subsample]
                db_abs_paths.extend([os.path.join(self.datasets_folder,"sync_data",seq,"thr/img_left",x) for x in rel_db_paths])

            elif self.db_modality == "lidar":
                rel_db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"proj_depth/",seq,"rgb", "depth_filtered")))[::self.subsample]
                db_abs_paths.extend([os.path.join(self.datasets_folder,"proj_depth/",seq,"rgb", "depth_filtered",x) for x in rel_db_paths])

            if self.q_modality == "rgb":
                rel_q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"rgb/img_left")))[::self.subsample]
                q_abs_paths.extend([os.path.join(self.datasets_folder,"sync_data",seq,"rgb/img_left",x) for x in rel_q_paths])
            elif self.q_modality == "thr":
                rel_q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"thr/img_left")))[::self.subsample]
                q_abs_paths.extend([os.path.join(self.datasets_folder,"sync_data",seq,"thr/img_left",x) for x in rel_q_paths])

            elif self.q_modality == "lidar":
                rel_q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"proj_depth/",seq,"rgb", "depth_filtered")))[::self.subsample]
                q_abs_paths.extend([os.path.join(self.datasets_folder,"proj_depth/",seq,"rgb", "depth_filtered",x) for x in rel_q_paths])
        return db_abs_paths,q_abs_paths
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}
    
    def get_utm_transformer(self,zone_number):
        # UTM North zones are EPSG 326## (e.g., zone 33N → EPSG:32633)
        epsg_code = f"326{int(zone_number):02d}"
        return Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)

    def get_xyz_coords(self,lon,lat,alt):
        """
        Converts latitude, longitude and altitude to UTM coordinates.
        """
        assert abs(lat) <= 90 and abs(lon) <= 180, f"Invalid lat/lon: {lat}, {lon}"

        easting, northing = self.utm_transformer.transform(lon, lat)
        # import pdb;pdb.set_trace()
        return [easting, northing, alt]

    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """
        self.db_coords = []
        self.q_coords = []
        # load files for the coordinates
        for seq in self.seq:
            self.db_coord_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"gps_imu/data")))[::self.subsample]
            self.q_coord_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,"sync_data",seq,"gps_imu/data")))[::self.subsample]
        

            for p in self.db_coord_paths:          
                with open(os.path.join(self.datasets_folder,"sync_data",seq,"gps_imu/data",p)) as f:
                    lines = f.readlines()
                    # lat = float(lines[0].strip())
                    # lon = float(lines[1].strip())
                    lon = float(lines[0].strip()) #the MS2 descriptions seems wrong
                    lat = float(lines[1].strip()) 
                    alt = float(lines[2].strip())
                    coord = self.get_xyz_coords(lon,lat,alt)
                    # import pdb;pdb.set_trace()

                    self.db_coords.append(coord)
                    self.q_coords.append(coord)

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        dist,soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= self.dist_thresh,
                                                            return_distance=True)
        return dist , soft_positives_per_query         
    def check_seq_list(self,seq):
        for s in seq:
            if os.path.isdir(os.path.join(self.datasets_folder,"sync_data",s)) == False:
                raise ValueError(f"Please provide a valid sequence name. {s} does not exist")

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = base_transform(img)
        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = load_as_float_img(path)
        img = process_one_image(img,type="hist_99")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = base_transform(img)
        return img
    
    def read_lidar(self, path):
        """
        Reads lidar image from the path.
        """
        img = cv2.imread(path)
        img = sparse_to_dense(img)
        img = base_transform(img)
        return img