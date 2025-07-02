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
    def __init__(self, model_type, modality, un_frozen_layer_index,layers_to_hook,backbone_path="",proj_head=""):
        self.model_type = model_type
        self.modality = modality
        self.un_frozen_layer_index = un_frozen_layer_index
        self.proj_head = proj_head
        if self.proj_head =="linear":
            self.modality_specific_proj_head = nn.Linear(768, 384).to(global_device)
            self.modality_shared_proj_head = nn.Linear(768, 384).to(global_device)
        elif self.proj_head == "":
            self.modality_specific_proj_head = None
            self.modality_shared_proj_head = None
        else:
            raise ValueError(f"Unsupported proj_head: {self.proj_head}. Supported proj_heads are 'linear' and ''.")

        self.model = torch.hub.load("facebookresearch/dinov2", self.model_type).to(global_device)
        self.patch_size = 14    
        if backbone_path != "":
            print(f"Loading backbone from {backbone_path}")
            state_dict = torch.load(backbone_path, map_location=global_device)["student_model_state_dict"]
            self.load_model(state_dict)
        # freeze the layer by setting requires_grad to False
        print("Unfreezing dinov2 layers:", self.un_frozen_layer_index)
        self.layers_to_hook = layers_to_hook
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

        if self.proj_head!= "":
            if 'proj_head' in self.un_frozen_layer_index:
                for param in self.modality_specific_proj_head.parameters():
                    param.requires_grad = True
                for param in self.modality_shared_proj_head.parameters():
                    param.requires_grad = True
            else:
                if self.un_frozen_layer_index != []:
                    raise ValueError("un_frozen_layer_index should contain 'proj_head' if proj_head is set. Please set it to ['proj_head'] or include 'proj_head' in the un_frozen_layer_index list.")
                for param in self.modality_specific_proj_head.parameters():
                    param.requires_grad = False
                for param in self.modality_shared_proj_head.parameters():
                    param.requires_grad = False

        print(f"Using model {self.model_type} with un_frozen_layer_index {self.un_frozen_layer_index} for modality {self.modality}")
    def forward_train(self, x, preprocess=False,return_local_features=False):
        if preprocess:
            x = self.preprocess(x)
        output_tokens_dict = self.extract_vit_tokens(x)

        if return_local_features:
            return output_tokens_dict
        else:
            return t
        
    def load_model(self, model_dict):
        """
        Loads the model state dict from a dictionary.
        """
        self.model.load_state_dict(model_dict["backbone_model_state_dict"])
        if self.proj_head != "":
            self.modality_specific_proj_head.load_state_dict(model_dict["modality_specific_proj_head_state_dict"])
            self.modality_shared_proj_head.load_state_dict(model_dict["modality_shared_proj_head_state_dict"])
        else:
            if "modality_specific_proj_head_state_dict" in model_dict:
                raise ValueError("modality_specific_proj_head_state_dict not found in model_dict. Please check the model_dict.")
        print("Model loaded successfully")

    
    def return_model_dict_for_saving(self):
        output = {}
        output["backbone_model_state_dict"] = self.model.state_dict()
        if self.modality_specific_proj_head is not None:
            output["modality_specific_proj_head_state_dict"] = self.modality_specific_proj_head.state_dict()
            output["modality_shared_proj_head_state_dict"] = self.modality_shared_proj_head.state_dict()
        return output

    def eval(self, set_grad=False):
        self.model.eval()
        if self.modality_specific_proj_head is not None:
            self.modality_specific_proj_head.eval()
        if self.modality_shared_proj_head is not None:
            self.modality_shared_proj_head.eval()

        if set_grad:
            for param in self.model.parameters():
                param.requires_grad = False

            if self.proj_head != "":
                for param in self.modality_specific_proj_head.parameters():
                    param.requires_grad = True
                for param in self.modality_shared_proj_head.parameters():
                    param.requires_grad = True

        
    def train(self):
        self.model.train()
        if self.modality_specific_proj_head is not None:
            self.modality_specific_proj_head.train()
        if self.modality_shared_proj_head is not None:
            self.modality_shared_proj_head.train()
    
    def shared(self,x):
        """
        Returns the shared projection head output
        """
        if len(x.shape) ==2:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}), but got {x.shape}"
            return self.modality_shared_proj_head(x)
        elif len(x.shape) == 4:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}, H, W), but got {x.shape}"
            return self.modality_shared_proj_head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Permute to (B, C', H, W)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected shape to be (B, C) or (B, C, H, W).")
    
    def specific(self,x):
        """
        Returns the modality specific projection head output
        """

        if len(x.shape) ==2:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}), but got {x.shape}"
            return self.modality_specific_proj_head(x)
        elif len(x.shape) == 4:
            assert x.shape[1] == self.model.embed_dim, f"Expected input shape to be (B, {self.model.embed_dim}, H, W), but got {x.shape}"
            return self.modality_specific_proj_head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected shape to be (B, C) or (B, C, H, W).")
            
    
    def unfrozen_parameters(self):
        output = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                output.append(param)
        if self.proj_head != "":
            for param in self.modality_specific_proj_head.parameters():
                if param.requires_grad:
                    output.append(param)
            for param in self.modality_shared_proj_head.parameters():
                if param.requires_grad:
                    output.append(param)
        return output
    
    def unfrozen_named_parameters(self):
        output = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                output.append((name, param))

        if self.proj_head != "":
            for name, param in self.modality_specific_proj_head.named_parameters():
                if param.requires_grad:
                    output.append((f"specific_proj_head.{name}", param))
            for name, param in self.modality_shared_proj_head.named_parameters():
                if param.requires_grad:
                    output.append((f"shared_proj_head.{name}", param))

        return output
            

    def preprocess(self, images):
        if self.modality == 'rgb':
            return preprocess_dinov2(images, normalise_model = 'vit')
        elif self.modality == 'thr':
            return preprocess_dinov2(images, normalise_model = 'vit',normalise=True) #normalising thermal also
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are 'rgb' and 'thr'.")

    

    def extract_vit_tokens(self,x):
        """
        Extracts CLS and patch tokens at specified intermediate transformer layers
        and final output after the norm — all from a single forward pass.

        Returns:
            tokens_dict with keys:
                block_{i}_cls, block_{i}_patch,
                final_cls, final_patch
        """
        B,H,W = x.shape[0],x.shape[-2],x.shape[-1]
        tokens_dict = {}

        def make_hook(i):
            if i == 'final':
                def hook_fn_final(module, input, output):
                    # output: (B, N+1, C)
                    tokens_dict["final_input"] = output
                return hook_fn_final
            else:
                def hook_fn(module, input, output):
                    # output: (B, N+1, C)
                    tokens_dict[f"block_{i}_output"] = (output[:, 1:].reshape((B, H // 14, W // 14, self.model.embed_dim)).permute(0, 3, 1, 2),output[:, 0])  # Patch tokens
                    tokens_dict[f"block_{i}_shared_output"] = self.shared(tokens_dict[f"block_{i}_output"][0]),self.shared(tokens_dict[f"block_{i}_output"][1])  # Shared projection head output
                    tokens_dict[f"block_{i}_specific_output"] = self.specific(tokens_dict[f"block_{i}_output"][0]),self.specific(tokens_dict[f"block_{i}_output"][1])  # Modality specific projection head output
                return hook_fn
        
        

        # Register hooks
        hook_handles = []
        for i in self.layers_to_hook:
            if i == 'final':
                h = self.model.blocks[-1].register_forward_hook(make_hook(i))
            else:
                h = self.model.blocks[i].register_forward_hook(make_hook(i))
            hook_handles.append(h)

        # Single forward pass
        _ = self.model(x)

        # Apply final norm manually to last hooked output
        if "final_input" not in tokens_dict:
            raise ValueError("Last layer not hooked; cannot compute final tokens.")
        final_output = self.model.norm(tokens_dict["final_input"])  # (B, N+1, C)
        tokens_dict.pop("final_input")  # Remove the raw output of the last layer
        tokens_dict["block_final_output"] = (final_output[:, 1:].reshape((B, H // 14, W // 14, self.model.embed_dim)).permute(0, 3, 1, 2) , final_output[:, 0])
        tokens_dict["block_final_shared_output"] = self.shared(tokens_dict["block_final_output"][0]),self.shared(tokens_dict["block_final_output"][1])  # Shared projection head output
        tokens_dict["block_final_specific_output"] = self.specific(tokens_dict["block_final_output"][0]),self.specific(tokens_dict["block_final_output"][1])  # Modality specific projection head output

        # Cleanup
        for h in hook_handles:
            h.remove()

        return tokens_dict

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

        
        backbone = MMDistillDinov2(backbone_model_type, self.modality, backbone_path=backbone_path,un_frozen_layer_index=self.un_frozen_layer_index,layers_to_hook=['final'], proj_head='linear')

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