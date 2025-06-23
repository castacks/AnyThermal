from .base_model import *
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from .dinov2_model import DINOv2FeatureExtractor
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import tqdm
from .dinov2_segmentation_model import seg_head_str_to_dict, BaseDinov2SegmentationModel
from .dinov2_vpr_model import NetVLADHead
import contextlib
from abc import ABC, abstractmethod
import inspect
import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad")
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition")
from salad.models.helper import get_aggregator

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MMDistillDinov2():
    def __init__(self, model_type, modality, un_frozen_layer_index,backbone_path=""):
        self.model_type = model_type
        self.modality = modality
        self.un_frozen_layer_index = un_frozen_layer_index
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_type).to(global_device)
        if backbone_path != "":
            print(f"Loading backbone from {backbone_path}")
            state_dict = torch.load(backbone_path, map_location=global_device)["student_model_state_dict"]
            self.model.load_state_dict(state_dict)
        # freeze the layer by setting requires_grad to False
        print("Unfreezing dinov2 layers:", self.un_frozen_layer_index)
        for name, param in self.model.named_parameters():
            if "blocks" in name:
                if int(name.split('.')[1]) not in self.un_frozen_layer_index:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif "patch_embed" in name:
                if "patch_embed" not in self.un_frozen_layer_index:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif "norm" in name:
                if "norm" not in self.un_frozen_layer_index:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
            # print(f"Setting requires_grad for {name} to {param.requires_grad}")

        print(f"Using model {self.model_type} with un_frozen_layer_index {self.un_frozen_layer_index} for modality {self.modality}")
    def forward(self,x,preprocess=False,debug=False):
        #implement a forward method that uses un_frozen_layer_index and self.model
        # if self.modality =='thr':
        #     debug =True
        if debug:
            print("x.requires_grad:", x.requires_grad)

        if preprocess:
            x = self.preprocess(x)
            if debug:
                print("After preprocess, x.requires_grad:", x.requires_grad)

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        if debug:
            print("After prepare_tokens_with_masks, x.requires_grad:", x.requires_grad)
        for idx,blk in enumerate(self.model.blocks):
            # with torch.no_grad() if idx not in self.un_frozen_layer_index else contextlib.nullcontext():
            if debug:
                if idx in self.un_frozen_layer_index:
                    print(f"UnFreezing layer {idx} for modality {self.modality}")
                print("x.requires_grad:", x.requires_grad)
            x = blk(x)
        
        x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.model.embed_dim)).permute(0, 3, 1, 2)
        return f, t

    
    def forward_train(self, x, preprocess=False,return_local_features=False,end_to_end=False):
        batch = x.shape[0]
        # import pdb; pdb.set_trace()
        if end_to_end:
            return self.model(x)
        f,t = self.forward(x, preprocess=preprocess)
        # temp = self.model(x.clone())
        # import pdb; pdb.set_trace()
        if return_local_features:
            return f,t
        else:
            return t
        
    def eval(self):
        self.model.eval()
    def train(self):
        self.model.train()
    
    def unfrozen_parameters(self):
        output = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                output.append(param)
        return output
            

    def preprocess(self, images):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'vit')
        elif self.modality == 'thr':
            return preprocess_dinov2(images, normalise_model = 'vit',normalise=True) #normalising thermal also
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thr'.")



class BaseDinov2VPRModel(nn.Module):
    def __init__(self, backbone: MMDistillDinov2, head):
        super().__init__()
        self.backbone = backbone  # ViT-based model returning tokens
        self.head = head  # VPR head
        
        assert isinstance(self.backbone, MMDistillDinov2), "Backbone must be an instance of MMDistillDinov2"
        

    def forward(self, x, frozen_backbone=True):
        if frozen_backbone:
            with torch.no_grad():
                tokens = self.backbone_forward(x)
        else:
            tokens = self.backbone_forward(x)
        # import pdb;pdb.set_trace()
        assert isinstance(tokens, tuple), "Backbone must return a tuple"
        return self.head(tokens)

    def backbone_forward(self, x):
        return self.backbone.forward(x,preprocess=False,debug=False)





class FixedThermalDistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    """
        The difference between this and the VariableThermalDistillDINOv2FeatureExtractor is that this uses a fixed size input - 518*518a as compared to a variable one - new_H = (H // patch_size) * patch_size , new_W = (W // patch_size) * patch_size
    """
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio,size=518,resize=resize,normalise=True)

class VariableThermalDistillDINOv2FeatureExtractor(FixedThermalDistillDINOv2FeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        return preprocess_dinov2(images, normalise_model = 'vit', normalise=True) 


class FixedRGBDistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio,size=518,resize=resize)

class VariableRGBDistillDINOv2FeatureExtractor(FixedRGBDistillDINOv2FeatureExtractor):
    def preprocess(self, images,keep_ratio=False,resize=True):
        return preprocess_dinov2(images, normalise_model = 'vit')


class DistillDINOv2FeatureExtractor(DINOv2FeatureExtractor):
    """
        The difference between this and the VariableThermalDistillDINOv2FeatureExtractor is that this uses a fixed size input - 518*518a as compared to a variable one - new_H = (H // patch_size) * patch_size , new_W = (W // patch_size) * patch_size
    """
    def build_model(self):
        model = torch.hub.load("facebookresearch/dinov2", self.model_type)
        state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, images,keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'vit',keep_ratio=keep_ratio,size=518,resize=resize)



class MMDistillSegmentationModel(BaseSegmentationModel):
    def __init__(self,head_model, un_frozen_layer_index,frozen_head,modality,device,num_classes,upscale_method,backbone_path="",model_path="",   **kwargs):
        self.head_model = head_model
        self.un_frozen_layer_index = un_frozen_layer_index
        
        self.backbone_path = backbone_path
        self.frozen_head = frozen_head
        self.model_path = model_path
        self.modality = modality
        self.upscale_method = upscale_method
        super().__init__(device, num_classes)

    def build_model(self):
        assert self.head_model in seg_head_str_to_dict, f"Unsupported head model: {self.head_model}. Supported models are: {list(seg_head_str_to_dict.keys())}"
        head_type  = seg_head_str_to_dict[self.head_model]
        head = head_type(in_channels=768, num_classes=self.num_classes).to(self.device)
        backbone_path = self.backbone_path
        backbone_model_type = "dinov2_vitb14"
        if self.backbone_path!= "" and self.model_path != "":
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")
        if self.backbone_path!= "":
            print(f"Loading backbone from {self.backbone_path}")
            backbone_model_type = torch.load(backbone_path, map_location=self.device)["student_model_type"]

            # state_dict = torch.load(self.backbone_path, map_location=self.device)["student_model_state_dict"]
            # backbone.load_state_dict(state_dict)
        elif self.model_path != "":
            print(f"Loading backbone and head model from {self.model_path}")
            head_state_dict = torch.load(self.model_path, map_location=self.device)["seg_head"]
            backbone_path = torch.load(self.model_path, map_location=self.device)["backbone_path"]
            if backbone_path != "":
                backbone_model_type = torch.load(backbone_path, map_location=self.device)["student_model_type"]
            # backbone.load_state_dict(backbone_state_dict)
            head.load_state_dict(head_state_dict)

        
        backbone = MMDistillDinov2(backbone_model_type, self.modality, backbone_path=backbone_path,un_frozen_layer_index=self.un_frozen_layer_index)

        model = BaseDinov2SegmentationModel(backbone, head,self.upscale_method).to(self.device)
        if self.frozen_head:
            model.head.eval()
        return model
    
    def preprocess(self, images):
        return None
    
    def forward(self, x):
        return self.model(x)
    
    def unfrozen_parameters(self):
        output = self.model.backbone.unfrozen_parameters()
        if not self.frozen_head:
            output.extend(list(self.model.head.parameters()))
        return output


default_agg_dict = {
    "agg_arch":'SALAD',
    "agg_config":{
        'num_channels': 768,
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256,
    }
}


class MMDistillVPRModel(BaseFeatureExtractor):
    def __init__(self, frozen_backbone,frozen_head,modality,un_frozen_layer_index=[],head_config={},backbone_model_type="",backbone_path="",model_path="", model=None, **kwargs):
        assert isinstance(head_config, dict), "head_config should be a dictionary containing the head configuration."
        if head_config == {}:
            head_config = default_agg_dict
            print("Using default head configuration:", head_config)
        
        self.backbone_path = backbone_path
        self.model_path = model_path


        if self.backbone_path != "" or self.model_path != "":
            if backbone_model_type != "":
                raise ValueError("backbone_model_type should not be set if backbone_path or model_path is set. Please set only one of them.")

        self.frozen_backbone = frozen_backbone
        self.un_frozen_layer_index = un_frozen_layer_index
        if self.frozen_backbone:
            self.un_frozen_layer_index =[]
        else:
            raise ValueError("frozen_backbone should be True for MMDistillVPRModel")
        self.backbone_model_type = backbone_model_type
        self.frozen_head = frozen_head
        self.modality = modality
        self.model = model
        self.head_config = head_config
        super().__init__(**kwargs)


    def build_model(self):
        if self.model is not None:
            print("Using provided model")
            return self.model.to(self.device)

        # import pdb; pdb.set_trace()
        if self.head_config["agg_arch"] == "NetVLAD":
            head = NetVLADHead(mode="local_only").to(self.device) #PARV_TODO make the mode and choosing a head configurable
        else:
            head = get_aggregator(**self.head_config).to(self.device)
        backbone_path = self.backbone_path
        if self.backbone_path == "" and self.model_path == "":
            backbone_model_type = self.backbone_model_type
        elif self.backbone_path!= "" and self.model_path == "":
            print(f"Loading backbone from {self.backbone_path}")
            backbone_model_type = torch.load(backbone_path, map_location=self.device)["student_model_type"]

        elif self.model_path != "" and self.backbone_path == "":
            print(f"Loading backbone and head model from {self.model_path}")
            head_state_dict = torch.load(self.model_path, map_location=self.device)["thermal_vpr_head"]
            head.load_state_dict(head_state_dict)
            backbone_path = torch.load(self.model_path, map_location=self.device)["backbone_path"]
            backbone_model_type = torch.load(backbone_path, map_location=self.device)["student_model_type"]

        else:
            raise ValueError("Both backbone_path and model_path cannot be set at the same time. Please set only one of them.")
        
        backbone = MMDistillDinov2(backbone_model_type, self.modality, un_frozen_layer_index = self.un_frozen_layer_index, backbone_path=backbone_path)
        model = BaseDinov2VPRModel(backbone, head)

        if self.frozen_backbone:
            model.backbone.eval()
        if self.frozen_head:
            model.head.eval()
        return model
    
    def preprocess(self, images, keep_ratio=False, resize=True):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'vit')
        elif self.modality == 'thermal':
            return preprocess_dinov2(images, normalise_model = 'vit',normalise=True) #normalising thermal also
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thermal'.")
    
    def forward(self, x):
        return self.model(x, frozen_backbone=self.frozen_backbone)

    def extract_feature(self, images,keep_ratio=False,resize=True, test=True):
        # import pdb; pdb.set_trace()
        images = self.preprocess(images,keep_ratio=keep_ratio,resize=resize).to(self.device)
        with torch.no_grad() if test else contextlib.nullcontext():
            feature = self.forward(images)
            assert feature is not None, "Feature extraction failed. Check the model and input."
            assert len(feature.shape) == 2, "Feature extraction should return a 2D tensor."
            assert feature.shape[0] == images.shape[0], "Feature shape mismatch with input batch size."
        return feature / feature.norm(p=2, dim=1, keepdim=True)  # Normalize the feature vector