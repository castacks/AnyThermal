from .base_dataset import *
from .ms2_utils import *
from pyproj import CRS
import pandas as pd
from pyproj import Proj, Transformer
class CartDataloader(CustomDataset):
    def __init__(self,db_modality,q_modality,datasets_folder='/ocean/projects/cis220039p/mdt2/shared/CART/bag_files',dist_thresh = 25,test=False,seq=[]):
        self.rgb_subsample = 60
        self.thermal_subsample = 30
        self.odom_subsample = 5
        super().__init__(db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,test=test,seq=seq)

    
    def parse_gt_csv(self,gt_csv_path,subsample):

        df = pd.read_csv(gt_csv_path)


        # Use the first non-null lat/lon to determine UTM zone
        first_valid_index = df[['latitude', 'longitude']].dropna().index[0]
        first_lat = df.at[first_valid_index, 'latitude']
        first_lon = df.at[first_valid_index, 'longitude']

        # Determine UTM zone
        zone_number = int((first_lon + 180) / 6) + 1
        is_northern = first_lat >= 0
        epsg_code = 32600 + zone_number if is_northern else 32700 + zone_number

        # Create transformer with autodetected EPSG
        crs_name = CRS.from_epsg(epsg_code).name
        transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)

        # Re-extract lat/lon every 5th row
        skipped_df = df.iloc[::subsample]
        lon = skipped_df['longitude'].values
        lat = skipped_df['latitude'].values

        # Convert to UTM using autodetected zone
        utm_coords = [transformer.transform(lon_, lat_) for lon_, lat_ in zip(lon, lat)]
        utm_df = pd.DataFrame(utm_coords, columns=["UTM_Easting", "UTM_Northing"])
        utm_df["Time"] = skipped_df["Time"].values
        utm_df["UTM_Zone"] = f"{zone_number}{'N' if is_northern else 'S'}"
        utm_df["CRS"] = crs_name
        utm_array = utm_df[["UTM_Easting", "UTM_Northing"]].to_numpy()
        return utm_array



    
    def form_gt_positives(self):
        """
        Returns ground truth positives for the dataset.
        """

        self.db_coords = []
        self.q_coords = []

        for seq in self.seq:
            self.db_coords.append(self.parse_gt_csv(os.path.join(self.datasets_folder,seq,"csv/_gps_fix.csv"),self.odom_subsample))
            self.q_coords.append(self.parse_gt_csv(os.path.join(self.datasets_folder,seq,"csv/_gps_fix.csv"),self.odom_subsample))
        
        self.db_coords = np.concatenate(self.db_coords,axis=0)
        self.q_coords = np.concatenate(self.q_coords,axis=0)

        # do knn over the coordinates
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_coords)
        dist,soft_positives_per_query = knn.radius_neighbors(self.q_coords,
                                                            radius= self.dist_thresh,
                                                            return_distance=True)
        return dist , soft_positives_per_query         

    
    
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        #rgb at 30Hz, thermal at 60Hz, GPS at  5Hz
        for seq in self.seq:
            if self.db_modality == "rgb":
                rel_db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.seq,"images/eo/color/image_color/compressed")))[::self.rgb_subsample]
                db_abs_paths.extend([os.path.join(self.datasets_folder,self.seq,"images/eo/color/image_color/compressed",x) for x in rel_db_paths])
            elif self.db_modality == "thr":
                rel_db_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.seq,"images/boson/thermal/image_raw")))[::self.thermal_subsample]
                db_abs_paths.extend([os.path.join(self.datasets_folder,self.seq,"images/boson/thermal/image_raw",x) for x in rel_db_paths])
            else:
                raise ValueError("Please provide a valid db_modality. Currently only rgb and thr are supported")

            if self.db_modality == "rgb":
                rel_q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.seq,"images/eo/color/image_color/compressed")))[::self.rgb_subsample]
                q_abs_paths.extend([os.path.join(self.datasets_folder,self.seq,"images/eo/color/image_color/compressed",x) for x in rel_q_paths])
            elif self.db_modality == "thr":
                rel_q_paths = natsorted(os.listdir(os.path.join(self.datasets_folder,self.seq,"images/boson/thermal/image_raw")))[::self.thermal_subsample]
                q_abs_paths.extend([os.path.join(self.datasets_folder,self.seq,"images/boson/thermal/image_raw",x) for x in rel_q_paths])
        return db_abs_paths,q_abs_paths
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal,
        }
    
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
        #PARV_TODO this should be changed to read TIFF data
        return img

    #URGENT _ PARV_TODO: check why is there a flip in the ocean_duck dataset, and how to implement it
    
    def __getitem__(self,index):

        if index>=self.database_num:
            if self.q_modality == "thr":
                # img = load_as_float_img(self.images_paths[index])
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

            elif self.q_modality == "rgb":
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

        elif index < self.database_num:
            if self.db_modality == "thr":
                # img = load_as_float_img(self.images_paths[index])
                # import pdb;pdb.set_trace()
                img = cv2.imread(self.images_paths[index])                
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                img = base_transform(img)

            elif self.db_modality == "rgb":
                img = cv2.imread(self.images_paths[index])
                h = img.shape[0]
                w = img.shape[1]
                if self.model_type=="clip":
                    img = cv2.resize(img, (224, 224))
                else:
                    img = cv2.resize(img, ((w//14)*14, (h//14)*14))
                if self.seq=="ocean_duck":
                    #Flip vertically
                    img = cv2.flip(img,0)
                    # cv2.imwrite("test.png",img)
                    # import pdb;pdb.set_trace()
                img = base_transform(img)

        return img, index

if __name__ == "__main__":

    args = None
    dataset = CartDataloader(args,db_modality="rgb",q_modality="thr",seq="Idyll_wild")

    # cv2.imwrite("test.png",dataset[0+dataset.db_num][0])
