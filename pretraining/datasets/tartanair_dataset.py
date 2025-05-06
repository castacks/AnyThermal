from .custom_dataset_loader import *
import sys
sys.path.append('/ocean/projects/cis220039p/pmaheshw/code/multi-modal/tartanairpy')
import tartanair as ta
tartanair_data_root = "/ocean/projects/cis220039p/shared/datasets/tartanair_v2"
ta.init(tartanair_data_root)

from tartanair.dataset import TartanAirImageDatasetObject

class Custom_TartanAirDataset(TartanAirImageDatasetObject):
    '''
        Using __init__ function of base class

        def __init__(self, tartanair_data_root, 
                        envs = [], 
                        difficulties = [], 
                        trajectory_ids = [], 
                        modalities = ['image'], 
                        camera_names = ['lcam_front'],
                        transform = None,
                        num_workers = 4):
        
        The TartanAirDatasetObject class implements a PyTorch Dataset object, which can be used to read data from the TartanAir dataset.

        Args:
        
        tartanair_data_root(str): The root directory of the TartanAir dataset.
        envs(list): A list of the environments to use. 
        difficulties(list): A list of the difficulties to use. The allowed names are: 'easy', 'hard'.
        trajectory_id(list): A list of the trajectory ids to use. If empty, then all the trajectories will be used.
        modalities(list): A list of the modalities to use. The allowed names are: 'image', 'depth', 'seg', 'imu', 'lidar'.
        camera_name(list): A list of the camera names to use. If the modality list does not include a form of an image (e.g. 'image', 'depth', 'seg'), then this parameter is ignored. 
    '''

    def __init__(self, tartanair_data_root, 
                        envs = [], 
                        difficulties = [], 
                        trajectory_ids = [], 
                        modalities = ['image'], 
                        camera_names = ['lcam_front'],
                        transform = None,
                        num_workers = 4,
                        frame_skip = 1):
        super().__init__(tartanair_data_root,envs,difficulties,trajectory_ids,modalities,camera_names,transform,num_workers)
        self.data = self.data[::frame_skip]
        self.num_data_entries = len(self.data)
        print('The dataset has {} entries after applying skip_frame = {}.'.format(self.num_data_entries,frame_skip))

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
        img = cv2.resize(img, ((w//14)*14, (h//14)*14))
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).to(torch.float32)
    
    def read_depth_overload(self, depthpath, max_depth=500):
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
            depth = self.visdepth(depth,max_depth)
            h  = depth.shape[0]
            w = depth.shape[1]
            depth = cv2.resize(depth, ((w//14)*14, (h//14)*14))
            depth = depth.transpose(2, 0, 1)
        return torch.from_numpy(depth).to(torch.float32)
    def depth_rgba_float32(self, depth_rgba):
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)
    def visdepth(self, depth, max_depth=500):
        depthvis = np.clip(max_depth/depth ,0 ,255)
        depthvis = depthvis.astype(np.uint8)
        depthvis = cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)
        depthvis = cv2.cvtColor(depthvis, cv2.COLOR_RGB2BGR)
        return depthvis

    def __getitem__(self, index):
        # Get the entry.
        entry = self.data[index]

        # Create the sample.
        sample = {}

        # Iterate over camera names.
        for camera_name in self.camera_names:
            # Create the camera sample.
            camera_sample = {}

            # Iterate over modalities.
            for modality in entry[camera_name]['data0'].keys():
                modality_new_name = modality
                # Get the data0 and data1 global paths. data0 is frame t and data1 is frame t+1
                data0_gp = entry[camera_name]['data0'][modality]
                # data1_gp = entry[camera_name]['data1'][modality] 

                # Read the data0 and data1.
                if 'image' in modality:
                    data0 = self.read_rgb(data0_gp)
                    # data1 = self.read_image(data1_gp)

                    # Transform the data0 and data1.
                    if self.transform is not None:
                        data0 = self.transform(data0)
                        # data1 = self.transform(data1)
                   
                    modality_new_name ='image'

                elif 'depth' in modality:  
                    data0 = self.read_depth_overload(data0_gp)
                    modality_new_name ='depth'
                    # camera_sample[modality_new_name+'_200'] = self.read_depth_overload(data0_gp,200)
                    # camera_sample[modality_new_name+'_1000'] = self.read_depth_overload(data0_gp,1000)


                    # data0 = np.tile(data0[...,np.newaxis], (1, 1,3))
                    # data1 = self.read_depth(data1_gp)

                elif 'dist' in modality:
                    data0 = self.read_dist(data0_gp)
                    # data1 = self.read_dist(data1_gp)

                elif 'seg' in modality:
                    data0 = self.read_seg(data0_gp)
                    # data1 = self.read_seg(data1_gp)

                # h = data0.shape[0]
                # w = data0.shape[1]
                
                # data0 = cv2.resize(data0, ((w//14)*14, (h//14)*14))

                # data0 = data0.transpose(2, 0, 1).astype(np.float32)
                # Add the data0 and data1 to the camera sample.
                camera_sample[modality_new_name] = data0
                # camera_sample[modality + '_1'] = data1

            # Add the camera sample to the sample.
            sample[camera_name] = camera_sample

            # Add the motion to the sample.
            sample[camera_name]['motion'] = entry[camera_name]['motion']
        if len(self.camera_names) > 1:
            raise ValueError("More than one camera name is not supported. Please use only one camera name.")
        # Return the sample.
        return sample[self.camera_names[0]]
