import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import faiss
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import logging

import sys 
from multi_dataset_loader import IntraDatasetBatchSampler
from .mmdistill_dinov2_model import MMDistillDinov2
from .base_model import *
import contextlib
import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
from salad.models.helper import get_aggregator
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
        
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.output_dim  = clusters_num * dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, model,modality):
        descriptors_num = 500000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_idx = np.random.choice(len(cluster_ds), images_num, replace=False)
        random_idx_to_dataset = cluster_ds.idx_to_dataset[random_idx.tolist()]
        sampler = IntraDatasetBatchSampler(random_idx_to_dataset, args.eval_batch_size)

        # random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.train_num_workers,
                                batch_sampler=sampler)
        with torch.no_grad():
            model.eval()
            print("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, self.dim), dtype=np.float32)
            batchix = 0
            for iteration, inputs in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs['item'][0][modality].to(model.device)
                preprocessed_inputs = model.preprocess(inputs)
                outputs = model.forward(preprocessed_inputs,use_head=False)
                print("Outputs shape: ", outputs.shape)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], self.dim, -1).permute(0, 2, 1)
                print("Image descriptors shape: ", image_descriptors.shape)
                image_descriptors = image_descriptors.cpu().numpy()
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    try:
                        descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
                    except Exception as e:
                        print(f"Error while assigning descriptors: {e}")
                        import pdb; pdb.set_trace()
                batchix += image_descriptors.shape[0]*descs_num_per_image
        del random_dl
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        print(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self.to(model.device)


class MMDistillDINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, modality,model_type,backbone_path,layer_to_hook='final',proj_head="",**kwargs):
        self.model_type = model_type
        self.backbone_path = backbone_path
        self.modality = modality
        self.layer_to_hook = layer_to_hook
        self.proj_head = proj_head
        super().__init__(**kwargs)

    def build_model(self):
        model = MMDistillDinov2(self.model_type,self.modality,un_frozen_layer_index=[],layers_to_hook=[self.layer_to_hook],backbone_path=self.backbone_path,proj_head=self.proj_head)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        """
        Preprocess the input images for the ImageBind model.image bind default preprocessing has keep_ratio as true (load_and_transform_vision_data in ImageBind/imagebind/data.py)
        But for consistenecy with other methods we use keep_ratio as false by default
        """
        return images

    def forward(self, inputs,mode="global"):
        assert self.proj_head =="", "Projection head is not supported in MMDistillDINOv2FeatureExtractor"
        output = self.model.forward_train(inputs, preprocess = True)[f"block_{self.layer_to_hook}_output"]
        if mode == "global":
            return output[1]
        elif mode == "patch":
            shape = output[0].shape
            return output[0].reshape(shape[0], shape[1], -1).permute(0,2,1)
        elif mode == "both":
            shape = output[0].shape
            return output[0].reshape(shape[0], shape[1], -1).permute(0,2,1), output[1]  # Return both patch and cls features
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please use 'global', 'patch', or 'both'.")
class MMDistillDinov2VLAD(MMDistillDINOv2FeatureExtractor):
    def __init__(self,num_clusters, modality,model_type, use_cls,backbone_path,layer_to_hook='final',proj_head="",**kwargs):
        if num_clusters>0:
            if use_cls:
                self.mode = "both"
            else:
                self.mode = "patch"
        else:
            self.mode = "global"
        print("!!!!!! Using mode: ", self.mode)
        super().__init__(modality=modality, model_type=model_type, backbone_path=backbone_path, layer_to_hook=layer_to_hook, proj_head=proj_head, **kwargs)
        self.num_clusters = num_clusters
        self.combined_feature_extraction = True if num_clusters > 0 else False

    def extract_all_features(self,args,db_model, db_dataset, qu_dataset ,batch_size):
        """
        Extracts features for the query and database datasets using the provided model.
        Args:
            db_model: The model used for feature extraction.
            db_dataset: The database dataset from which to extract features.
            qu_dataset: The query dataset from which to extract features.
            batch_size: The batch size for feature extraction.
        Returns:
            A tuple containing the query and database features.
        """
        db_patch_features,db_global_features = self.extract_batch(db_dataset, batch_size, db_model)
        qu_patch_features,qu_global_features = self.extract_batch(qu_dataset, batch_size)
        vlad_centroids = self.calculate_vlad_centroids(db_patch_features)
        
        db_features = self.calculate_vlad_feature(db_patch_features,vlad_centroids)
        qu_features = self.calculate_vlad_feature(qu_patch_features,vlad_centroids)

        if self.mode == "both":
            print("Combining patch and global features for both database and query datasets.")
            assert db_global_features is not None and qu_global_features is not None, "Global features should not be None when mode is 'both'."
            db_features = torch.cat((db_features, db_global_features), dim=1)
            qu_features = torch.cat((qu_features, qu_global_features), dim=1)
        
        print(f"Database features shape: {db_features.shape}, Query features shape: {qu_features.shape}")
        return db_features,qu_features
    def calculate_vlad_centroids(self, all_features):
        """
        Calculates the VLAD centroids for the database dataset.
        Args:
            db_model: The model used for feature extraction.
            db_dataset: The database dataset from which to extract features.
            batch_size: The batch size for feature extraction.
        Returns:

            The VLAD centroids.
        """
        torch.cuda.empty_cache()
        dim = all_features.shape[-1]
        print(f"Calculating VLAD centroids for the database dataset - dim : {dim} shape : {all_features.shape}")

        kmeans = faiss.Kmeans(d=dim, k=self.num_clusters, niter=100)
        kmeans.train(all_features.reshape(-1,dim).cpu().numpy())  # Train KMeans on all features
        vlad_centroids = torch.from_numpy(kmeans.centroids).to(all_features.device)
        vlad_centroids = F.normalize(
            vlad_centroids,p=2, dim=-1
        )  # L2-normalize centroids
        print("VLAD centroids shape: ", vlad_centroids.shape)
        return vlad_centroids
    
    def calculate_vlad_feature(self, features, vlad_centroids):
        """
        Computes VLAD encoding (batched) with intra-normalization and inter-normalization.

        Args:
            features: Patch features. Shape: [B, N, D]
            vlad_centroids: Precomputed VLAD centroids. Shape: [K, D]

        Returns:
            vlad_features: Final VLAD descriptors. Shape: [B, K * D]
        """
        torch.cuda.empty_cache()

        assert len(features.shape) == 3, "features should be of shape [B, N, D]"
        B, N, D = features.shape
        assert len(vlad_centroids.shape) == 2, "vlad_centroids should be of shape [K, D]"
        K = vlad_centroids.shape[0]  # num_clusters
        # Step 1: Compute assignments (nearest centroid for each patch)
        # distances: [B, N, K]
        distances = torch.cdist(features, vlad_centroids.unsqueeze(0), p=2)  # broadcast centroids
        assignments = torch.argmin(distances, dim=2)  # [B, N]

        # Step 2: Initialize VLAD matrix [B, K, D]
        vlad = torch.zeros((B, K, D), device=features.device)

        # Step 3: Accumulate residuals for each cluster
        for k in range(K):
            # mask: [B, N], select patches assigned to cluster k
            mask = (assignments == k).unsqueeze(-1).float()  # [B, N, 1]
            residuals = features - vlad_centroids[k]  # [B, N, D]
            # masked residuals: zero out patches not assigned to cluster k
            masked_residuals = residuals * mask  # [B, N, D]
            # sum residuals per image for cluster k
            vlad[:, k, :] = masked_residuals.sum(dim=1)  # [B, D]

        # Step 4: Intra-normalization (normalize each cluster residual vector per image)
        vlad = F.normalize(vlad, p=2, dim=-1)  # [B, K, D]

        # Step 5: Concatenate all cluster residuals
        vlad = vlad.view(B, -1)  # [B, K * D]

        # Step 6: Inter-normalization (normalize final VLAD vector per image)
        vlad = F.normalize(vlad, p=2, dim=-1)  # [B, K * D]

        return vlad  # [B, K * D]
    def extract_batch(self,dataset,batch_size,db_model= None):
        """
        Extracts features from the input images using the DINOv2 model.
        Args:
            inputs: The input images for feature extraction.
        Returns:
            The extracted features.
        """
        assert self.mode in ["patch", "both"], f"Unsupported mode: {self.mode}. Please use 'patch', or 'both'."
        all_features = []
        if self.mode =="both":
            all_global_features = []
        else:
            all_global_features = None
        data_loader = DataLoader(dataset, num_workers=4, shuffle=False,batch_size=batch_size)
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, ncols=100):
                if db_model is None:
                    features = self.forward(inputs.to(self.device), mode=self.mode)  # Use patch pooling for VLAD
                else:
                    features = db_model.forward(inputs.to(self.device),mode =self.mode)  # Use patch pooling for VLAD
                if self.mode =="both":
                    all_features.append(features[0].cpu())
                    all_global_features.append(features[1].cpu())
                elif self.mode == "patch":
                    all_features.append(features.cpu())
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}. Please use 'global', 'patch', or 'both'.")
        del data_loader
        all_features = torch.cat(all_features, dim=0)
        all_features = F.normalize(all_features, p=2, dim=-1)
        if self.mode == "both":
            all_global_features = torch.cat(all_global_features, dim=0)
            all_global_features = F.normalize(all_global_features, p=2, dim=-1)
        return all_features,all_global_features

class MMDistillVPRModel(BaseFeatureExtractor):
    def __init__(self, args,frozen_backbone,frozen_head,modality,un_frozen_layer_index=[],head_config=None,backbone_model_type="",backbone_path="",model_dict="", **kwargs):
        assert isinstance(head_config, dict) or head_config is None, "head_config should be a dictionary or None. If None, default aggregation configuration will be used."
        if head_config == {}:
            raise ValueError("head_config should not be an empty dictionary. Please provide a valid configuration or set it to None for default aggregation.")
        self.head_config = head_config

        
        self.backbone_path = backbone_path
        self.model_dict = model_dict
        self.args = args

        if not hasattr(self.args, 'conv_output_dim'):
            self.args.conv_output_dim = None
        if not hasattr(self.args, 'fc_output_dim'):
            self.args.fc_output_dim = None


        # if self.backbone_path != "" or self.model_dict != "":
        #     if backbone_model_type != "":
        #         raise ValueError("backbone_model_type should not be set if backbone_path or model_dict is set. Please set only one of them.")

        self.frozen_backbone = frozen_backbone
        self.un_frozen_layer_index = un_frozen_layer_index
        self.backbone_model_type = backbone_model_type
        self.frozen_head = frozen_head
        self.modality = modality
        super().__init__(**kwargs)
        self.backbone = self.backbone.to(self.device)
        if self.head is not None:
            self.head = self.head.to(self.device)
        if self.conv_layer is not None:
            self.conv_layer = self.conv_layer.to(self.device)

    def build_model(self):

        backbone_path = self.backbone_path
        load_head = False
        backbone_state_dict = None
        if self.backbone_path == "" and self.model_dict == "":
            print("Neither backbone_path nor model_path is set. Using default DINOv2 backbone.")
        elif self.backbone_path!= "" and self.model_dict == "":
            print(f"Loading backbone from {self.backbone_path}")
            self.backbone_model_type = torch.load(backbone_path, map_location=self.device,weights_only=True)["student_model_type"]
        elif self.model_dict != "" and self.backbone_path == "":
            head_state_dict = self.model_dict["vpr_head"]
            load_head = True
            backbone_path = self.model_dict["backbone_path"]
            self.backbone_model_type = self.model_dict["model_type"]
            assert self.head_config is None or self.head_config == self.model_dict["head_config"], "head_config in model_path does not match the provided head_config. Please check the model configuration."
            self.head_config = self.model_dict["head_config"]
            if "backbone" in self.model_dict.keys():
                backbone_state_dict = self.model_dict["backbone"]
        else:
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")


        self.conv_layer = None
        
        if self.args.conv_output_dim != None and self.args.conv_output_dim > 0:
            # Concatenate conv layer to the aggregation layer
            actual_conv_output_dim = int(self.args.conv_output_dim / self.head_config['agg_config']['num_clusters'])
            if self.args.add_bn:
                self.conv_layer = nn.Sequential(nn.Conv2d(768, actual_conv_output_dim, 1, bias=False),
                                                nn.BatchNorm2d(actual_conv_output_dim),)
            else:
                self.conv_layer = nn.Conv2d(768, actual_conv_output_dim, 1)
            self.args.features_dim = actual_conv_output_dim
        else:
            self.args.features_dim = 768  # Default feature dimension for DINOv2

        if self.head_config == None:
            head = None
        elif self.head_config["agg_arch"] == "NetVLAD":
            head = NetVLAD(clusters_num=self.head_config['agg_config']['num_clusters'],dim=self.args.features_dim).to(self.device)
        else:
            head = get_aggregator(**self.head_config).to(self.device)
        
        self.backbone = MMDistillDinov2(self.backbone_model_type, self.modality, un_frozen_layer_index = self.un_frozen_layer_index, backbone_path=backbone_path, layers_to_hook=['final'])
        self.head = head
        self.backbone_path = backbone_path
        if backbone_state_dict is not None:
            print(f"Loading backbone state dict from VPR model")
            self.backbone.load_model(backbone_state_dict)

        if self.head is not None:
            if self.args.fc_output_dim != None and self.args.fc_output_dim > 0:
                # Concatenate fully connected layer to the aggregation layer
                self.head = nn.Sequential(self.head,
                                                nn.Linear(self.head.output_dim, self.args.fc_output_dim),
                                                L2Norm())
                self.args.features_dim = self.args.fc_output_dim
            else:
                self.head = nn.Sequential(self.head, L2Norm())
                self.args.features_dim = self.head[0].output_dim

            if self.conv_layer is not None:
                self.head = nn.Sequential(self.conv_layer, self.head)
        
        else:
            self.head = self.conv_layer

        assert self.head is not None , "Either head or conv_layer should be defined. Please check the model configuration."
        
        if load_head:
            self.head.load_state_dict(head_state_dict)
            
        
        if self.frozen_backbone:
            self.backbone.eval()
        if self.frozen_head:
            self.head.eval()

        return None
    
    def preprocess(self, images, keep_ratio=False, resize=True):
        return images
    
    def forward(self, x, use_head=True):
        with torch.no_grad() if self.frozen_backbone else contextlib.nullcontext():
            tokens = self.backbone_forward(x)

        if self.conv_layer is not None:
            if use_head:
                return self.head(tokens)  # Apply conv layer to the patch tokens
            else:
                return self.conv_layer(tokens)
        else:
            if use_head:
                return self.head(tokens)
            else:
                return tokens


    def backbone_forward(self, x):
        output = self.backbone.forward_train(x, preprocess=True)['block_final_output']
        if self.head_config["agg_arch"] == "NetVLAD":
            return output[0]  # B, C, H, W
        elif self.head_config["agg_arch"] == "SALAD":
            return output #return patch,cls
        else:
            raise ValueError(f"Unsupported aggregation architecture: {self.head_config['agg_arch']}. Please use 'NetVLAD' or 'SALAD'.")


    def extract_feature(self, images,keep_ratio=False,resize=True, test=True):
        # import pdb; pdb.set_trace()
        images = self.preprocess(images,keep_ratio=keep_ratio,resize=resize).to(self.device)
        with torch.no_grad() if test else contextlib.nullcontext():
            feature = self.forward(images)
            assert feature is not None, "Feature extraction failed. Check the model and input."
            assert len(feature.shape) == 2, "Feature extraction should return a 2D tensor."
            assert feature.shape[0] == images.shape[0], "Feature shape mismatch with input batch size."
        return feature / feature.norm(p=2, dim=1, keepdim=True)  # Normalize the feature vector

    def eval(self):
        self.backbone.eval()
        if self.head is not None:
            self.head.eval()
    
    def train(self):
        self.backbone.train()
        if self.head is not None:
            self.head.train()
    
    def state_dict(self):
        """
        Returns the state dict of the model.
        """
        output_dict = {
            "backbone_path" : self.backbone_path,
            "model_type": self.backbone_model_type,
            "head_config": self.head_config,
        }
        if self.backbone.unfrozen_parameters() != []:
            output_dict["backbone"] = self.backbone.return_model_dict_for_saving()
        if self.head is not None:
            output_dict["vpr_head"] = self.head.state_dict()
        return output_dict

    def trainable_params(self):
        """
        Returns the trainable parameters of the model.
        """
        output = []
        for param in self.backbone.unfrozen_parameters():
            output.append(param)
        if self.head is not None:
            for param in self.head.parameters():
                if param.requires_grad:
                    output.append(param)
        return output