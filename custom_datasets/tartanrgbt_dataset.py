from .base_dataset import *
import torchvision.transforms.functional as torch_F
import torchvision
from copy import copy
import yaml

def return_tartanrgbt_split(split,debug=False):
    if not debug:
        #27
        if split == "val":
            return ['indoor_outdoor_mill19_building_interior_exterior','indoor_CFA_seq_2',
                    'urban_resedential_frick_park',
                    'park_frick_seq_4_return_tranquil_trail_start_to_falls_ravine','park_frick_seq_7_deer_creek_trail_and_nine_mile_start',
                    'offroad_turnpike_seq_1','offroad_turnpike_seq_4','offroad_turnpike_seq_3','offroad_turnpike_seq_2']
        elif split == "train":
            return ['indoor_NSH_third_floor','indoor_NSH_fourth_floor','indoor_NSH_first_floor','indoor_SQH_office',
                    'outdoor_campus_NSH_TO_CUT',
                    'outdoor_resedential_SQH_block',
                    'outdoor_urban_road_campus_to_marget_morrison',
                    'indoor_GATES_garage_1','indoor_GATES_garage_3','indoor_GATES_seq_1','indoor_GATES_seq_2',
                    'outdoor_urban_road_mill_19_seq_1','outdoor_urban_mill_19_circle_building_seq_1','outdoor_urban_road_mill_19_seq_2','outdoor_urban_mill_19_exterior',
                    'park_frick_seq_1_riverview_trail','park_frick_seq_3_openarea_to_tranquil_trail','park_frick_seq_5_return_falls_ravine_to_riverview','park_frick_seq_6_return_reverview','park_frick_seq_8_nine_mile_run',
                    'offroad_figure_eight_morning_1','offroad_figure_eight_morning_2','offroad_warehouse_loop_morning','offroad_warehouse_fence_morning','offroad_rough_rider_afternoon','offroad_figure_eight_morning_3','offroad_figure_eight_morning_4','offroad_figure_eight_rough_rider_start','offroad_warehouse_loop_evening','offroad_warehouse_to_garage','offroad_figure_eight_to_fence_to_garage']
            #offroad_return_turnpike_rough_rider_figure_8_to_fence cannot be used since it covers all areas - turnpike, figure8, rough rider
        elif split == "train_full":
            #first five trajectories were not used in the paper for training. For recreating paper, use train split. 
            return ['indoor_library', 
                    'urban_park_schenley', 
                    'urban_campus_outdoor_1',
                    "indoor_GATES_seq_3",
                    'outdoor_campus_GATES_to_cut_and_loop_around_cut',
                    'indoor_NSH_third_floor','indoor_NSH_fourth_floor','indoor_NSH_first_floor','indoor_SQH_office',
                    'outdoor_campus_NSH_TO_CUT',
                    'outdoor_resedential_SQH_block',
                    'outdoor_urban_road_campus_to_marget_morrison',
                    'indoor_GATES_garage_1','indoor_GATES_garage_3','indoor_GATES_seq_1','indoor_GATES_seq_2',"indoor_GATES_seq_3",
                    'outdoor_campus_GATES_to_cut_and_loop_around_cut',
                    'outdoor_urban_road_mill_19_seq_1','outdoor_urban_mill_19_circle_building_seq_1','outdoor_urban_road_mill_19_seq_2','outdoor_urban_mill_19_exterior',
                    'park_frick_seq_1_riverview_trail','park_frick_seq_3_openarea_to_tranquil_trail','park_frick_seq_5_return_falls_ravine_to_riverview','park_frick_seq_6_return_reverview','park_frick_seq_8_nine_mile_run',
                    'offroad_figure_eight_morning_1','offroad_figure_eight_morning_2','offroad_warehouse_loop_morning','offroad_warehouse_fence_morning','offroad_rough_rider_afternoon','offroad_figure_eight_morning_3','offroad_figure_eight_morning_4','offroad_figure_eight_rough_rider_start','offroad_warehouse_loop_evening','offroad_warehouse_to_garage','offroad_figure_eight_to_fence_to_garage']
            #offroad_return_turnpike_rough_rider_figure_8_to_fence cannot be used since it covers all areas - turnpike, figure8, rough rider
        else:
            raise ValueError("Please provide a valid split name. Options are 'train' or 'val'")
    elif debug:
        if split == "val":
            return ['indoor_SQH_office']
        elif split == "train":
            return ['indoor_SQH_office']
        else:
            raise ValueError("Please provide a valid split name. Options are 'train' or 'val'")
    else:
        raise ValueError("Please provide a valid split name. Options are 'train' or 'val'")



class TartanRGBT(BaseDataset):
    """
    Returns dataset class with images from database and queries for the vpair dataset. 
    """
    def __init__(self,args,db_modality,q_modality,datasets_folder,seq,augment,crop_images,vpr_test=False,vpr_train=False,dist_thresh = 25, rescale_during_crop=False,crop_during_vpr_test=False, val_positive_dist_threshold=-1):
        self.sequence_map_yaml = os.path.join(os.environ["ANYTHERMAL_PROJECT_ROOT"],"custom_datasets/splits/tartanRGBT/sequence.yaml")
        self.sequence_map = yaml.load(open(self.sequence_map_yaml,'r'), Loader=yaml.FullLoader)
        assert len(self.sequence_map.keys()) > 0, "Sequence map is empty. Please check the sequence_map_yaml path."
        actual_seq = []
        for s in seq:
            for key,value in self.sequence_map['traj_list'].items():
                if value == s:
                    actual_seq.append(key)
        self.subsample =10
        assert crop_during_vpr_test == False, "Crop during VPR test is not supported for TartanRGBT dataset. Please set it to False."
        dist_thresh = -1
        self.positive_radius_index = 1
        self.val_extra_margin_positive_radius_index = 2
        self.neg_ring_outer_radius_index = 5
        self.location_type = 'time'
        
        super().__init__(args=args,db_modality =db_modality,q_modality=q_modality,datasets_folder=datasets_folder,dist_thresh=dist_thresh,vpr_test=vpr_test,vpr_train=vpr_train,seq=actual_seq,augment = augment, rescale_during_crop=rescale_during_crop,crop_images=crop_images)
    
    def generate_read_fn(self):
        return {
            "rgb": self.read_rgb,
            "thr": self.read_thermal
        }
    
    def seq_path(self,seq,root):
        """
        Returns the path to the sequence.
        """
        return os.path.join(root,seq)
    
    def read_ffc(self,seq_path):
        file = os.path.join(os.path.join(seq_path,"thermal_left_ffc","data.txt"))
        with open(file,'r') as f:
            lines = f.readlines()
        ffc_list = np.array([int(line.strip()) for line in lines])
        return ffc_list[::self.subsample].astype(bool)

    
    def extract_frames(self,seq_path,ffc):
        frame_list = natsorted(os.listdir(seq_path))[::self.subsample]
        frame_list = [os.path.join(seq_path,x) for x in frame_list if x.endswith('.png') or x.endswith('.jpg')]
        assert len(ffc) == len(frame_list) , f"FFC length {len(ffc)} and frame length {len(frame_list)} do not match. Please check the dataset."
        return [frame_list[i] for i in range(len(frame_list)) if not ffc[i]]
    
    def extract_frames_from_common_folder_10_hz(self,seq_path,ffc):
        #no subsample
        frame_list = [x for x in natsorted(os.listdir(seq_path)) if "rgb_in_thermal" in x]
        frame_list = [os.path.join(seq_path,x) for x in frame_list if x.endswith('.png') or x.endswith('.jpg')]
        assert len(ffc) == len(frame_list) , f"FFC length {len(ffc)} and frame length {len(frame_list)} do not match. Please check the dataset."
        return [frame_list[i] for i in range(len(frame_list)) if not ffc[i]]
 
    def generate_image_paths(self,db_abs_paths,q_abs_paths):
        for seq in self.seq:
            ffc = self.read_ffc(self.seq_path(seq,self.datasets_folder))
            if self.db_modality == "rgb":
                db_folder = "rgb_in_thermal"
            elif self.db_modality == "thr":
                db_folder = "thermal_left_rect_8"
            else:
                raise ValueError("Please provide a valid db_modality. Options are 'rgb', 'thr'")

            if self.q_modality == "rgb":
                q_folder = "rgb_in_thermal"
            elif self.q_modality == "thr":
                q_folder = "thermal_left_rect_8"
            else:
                raise ValueError("Please provide a valid q_modality. Options are 'rgb', 'thr'")
            seq_path = self.seq_path(seq,self.datasets_folder)
            db_abs_paths.extend(self.extract_frames_from_common_folder_10_hz(os.path.join(seq_path,db_folder),ffc))
            q_abs_paths.extend(self.extract_frames(os.path.join(seq_path,q_folder),ffc))

            assert len(db_abs_paths) > 0, f"No images found in the database path: {os.path.join(seq_path,db_folder)}. Please check the dataset."
            assert len(q_abs_paths) > 0, f"No images found in the query path: {os.path.join(seq_path,q_folder)}. Please check the dataset."

        assert len(db_abs_paths) == len(q_abs_paths), "Database and Query image paths are not of same length. Please check the dataset."
            
        return db_abs_paths,q_abs_paths

    def read_rgb(self, path):
        """
        Reads rgb image from the path.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if img is None:
            raise ValueError(f"Image at {path} could not be read. Please check the path.")

        img = base_transform(img)

        return img
    
    def read_thermal(self, path):
        """
        Reads thermal image from the path.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to grayscale
        img = base_transform(img)
        return img

    def interpolate_odometry_strict(self,t_src, odo_src, t_tgt,target_timestamp_file):
        """
        Interpolate odometry poses at new timestamps with NO extrapolation or clamping.

        Args:
            t_src: (N,) float timestamps (seconds), unique but not necessarily sorted.
            odo_src: (N,7) = [x,y,z,qx,qy,qz,qw], quaternions should be unit (will be renormalized).
            t_tgt: (M,) float target timestamps.

        Returns:
            (M,7) interpolated poses.

        Raises:
            ValueError: if any t_tgt < min(t_src) or t_tgt > max(t_src), or if N < 2.
        """
        t_src = np.asarray(t_src, dtype=float)
        odo_src = np.asarray(odo_src, dtype=float)
        t_tgt = np.asarray(t_tgt, dtype=float)
        if odo_src.shape != (t_src.shape[0], 7):
            raise ValueError("odo_src must have shape (N,7).")
        if t_src.size < 2:
            raise ValueError("Need at least 2 source poses to interpolate.")

        # sort + dedupe
        order = np.argsort(t_src)
        t_src = t_src[order]
        odo_src = odo_src[order]
        keep = np.ones_like(t_src, dtype=bool)
        keep[1:] = t_src[1:] != t_src[:-1]
        t_src, odo_src = t_src[keep], odo_src[keep]
        N = t_src.size
        if N < 2:
            raise ValueError("After removing duplicate timestamps, need at least 2 poses.")

        # strict bounds check (no extrapolation)
        tmin, tmax = t_src[0], t_src[-1]
        tolerance = 1
        if np.any(t_tgt < tmin - tolerance) or np.any(t_tgt > tmax + tolerance):
            import pdb;pdb.set_trace()
            raise ValueError("Target timestamps outside source range are not allowed.")
        
        t_tgt = np.clip(t_tgt, tmin, tmax)
        # split into exact-last and the rest (searchsorted with 'right' overflows at last)
        exact_last = (t_tgt == tmax)
        need_interp = ~exact_last

        # prepare outputs
        out = np.empty((t_tgt.size, 7), dtype=float)

        # endpoints: exact last
        out[exact_last] = odo_src[-1]

        # interpolate others
        if np.any(need_interp):
            tt = t_tgt[need_interp]
            idx = np.searchsorted(t_src, tt, side="right")  # in [1, N-1]
            # segments [idx-1, idx]
            i0 = idx - 1
            i1 = idx

            t0 = t_src[i0]
            t1 = t_src[i1]
            dt = t1 - t0
            # dt must be > 0 due to dedupe & sort
            alpha = (tt - t0) / dt

            # positions (lerp)
            p0 = odo_src[i0, :3]
            p1 = odo_src[i1, :3]
            pos = p0 + alpha[:, None] * (p1 - p0)

            # quaternions (slerp)
            q0 = self._normalize_quat(odo_src[i0, 3:7])
            q1 = self._normalize_quat(odo_src[i1, 3:7])

            # shortest-arc per-pair
            dots = np.sum(q0 * q1, axis=1)
            neg = dots < 0.0
            q1 = q1.copy()
            q1[neg] = -q1[neg]
            dots = np.abs(dots)
            dots = np.clip(dots, -1.0, 1.0)

            omega = np.arccos(dots)
            sin_omega = np.sin(omega)
            eps = 1e-8
            use_nlerp = sin_omega < eps

            # slerp
            s0 = np.sin((1.0 - alpha) * omega) / (sin_omega + 1e-16)
            s1 = np.sin(alpha * omega) / (sin_omega + 1e-16)
            q_slerp = q0 * s0[:, None] + q1 * s1[:, None]

            # nlerp fallback for tiny angles
            q_nlerp = self._normalize_quat(q0 * (1.0 - alpha)[:, None] + q1 * alpha[:, None])

            q = np.empty_like(q0)
            q[~use_nlerp] = q_slerp[~use_nlerp]
            q[ use_nlerp] = q_nlerp[ use_nlerp]
            q = self._normalize_quat(q)

            out[need_interp, :3] = pos
            out[need_interp, 3:7] = q

        return out


    def _normalize_quat(self,q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return q / n

    def interpolate_odometry(self,odometry,target_timestamp_file):
        target_timestamps = np.loadtxt(target_timestamp_file)[::self.subsample]
        odom_timestamps = odometry[:,0] / 1e9  # Convert from nanoseconds to seconds
        interp_odom = self.interpolate_odometry_strict(odom_timestamps,odometry[:,1:],target_timestamps,target_timestamp_file)
        return interp_odom[:,:3]  # Return only x,y,z

    
    def form_db_qu_coords(self):
        # Form ground truth positives for the dataset. For a given index all images with indices wihting the self.positive_radius_index are considered positives.
        """
        Returns ground truth positives for the dataset.
        """
        offset = 0
        decrement = 5000
        self.db_coords = []
        self.q_coords = []
        for seq in self.seq:
            ffc = self.read_ffc(self.seq_path(seq,self.datasets_folder))
            offset -= decrement
            target_timestamp_file = os.path.join(self.seq_path(seq,self.datasets_folder),"target_timestamps.txt")
            odometry_file = np.load(os.path.join(self.seq_path(seq,self.datasets_folder),"odometry","poses.npy"))
            odometry = self.interpolate_odometry(odometry_file,target_timestamp_file)
            odometry[:,:2] += offset
            odometry = odometry[~ffc]
            self.db_coords.extend(odometry.tolist())
            self.q_coords.extend(odometry.tolist())            
    
    def check_seq_list(self,seq):
        """
        Checks if the sequence list is valid.
        """
        for s in seq:
            if not os.path.exists(self.seq_path(s,self.datasets_folder)) and not os.path.exists(self.seq_path(s,self.aligned_RGB_root)):
                import pdb;pdb.set_trace()
                raise ValueError(f"Sequence {s} does not exist. Please check the sequence name.")
    
    def semantic_classes_num_and_map_to_rgb(self):
        return -1,{}