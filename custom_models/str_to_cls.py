from .alexnet_model import AlexNetFeatureExtractor
from .vgg16_model import VGG16FeatureExtractor
from .resnet_model import ResNetFeatureExtractor
from .squeezenet_model import SqueezeNetFeatureExtractor
import torch 
import os
def get_model_from_string(args,name: str,task,**kwargs):
    """
    Returns the corresponding feature extractor class based on the model name.
    Supported:
    - 'alexnet'
    - 'vgg16'
    - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    - 'squeezenet1_0', 'squeezenet1_1'
    - 'dinov2_vits14' 
    - 'imagebind'
    """
    name = name.lower()

    if task =='vpr':
    
        if name == 'alexnet':
            return AlexNetFeatureExtractor()
        
        elif name=='sgm':
            from .sgm_model import SGMFeatureExtractor
            return SGMFeatureExtractor()

        elif name == 'vgg16':
            return VGG16FeatureExtractor()

        elif name.startswith('resnet'):
            try:
                depth = int(name.replace('resnet', ''))
                return ResNetFeatureExtractor(resnet_depth=depth)
            except ValueError:
                raise ValueError(f"Invalid ResNet name: {name}")

        elif name.startswith('squeezenet'):
            if name not in ['squeezenet1_0', 'squeezenet1_1']:
                raise ValueError(f"Unsupported SqueezeNet version: {name}")
            version = name.split('squeezenet')[-1]
            return SqueezeNetFeatureExtractor(version=version)

        elif name.startswith('imagebind'):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thr']:
                raise ValueError(f"Unsupported ImageBind modality: {modality}")
            if modality == 'rgb':
                from .imagebind_model import ImageBindRGBFeatureExtractor
                return ImageBindRGBFeatureExtractor()
            elif modality == 'thr':
                from .imagebind_model import ImageBindThermalFeatureExtractor
                return ImageBindThermalFeatureExtractor()
            else:
                raise ValueError(f"Unsupported ImageBind modality: {modality}")

        elif name == "salad":
            from .dinov2salad_model import DinoV2SALADFeatureExtractor
            return DinoV2SALADFeatureExtractor()
        elif name.startswith("mmdistill"):
            modality = name.split('_')[-1]
    
            vlad = True if 'vlad' in name.split('_')[-3] else False
            use_cls = True
            if vlad:
                use_cls = "global" in name.split('_')[-3]
                num_clusters = int(name.split('_')[-2])
                assert num_clusters > 0, f"num_clusters should be greater than 0, got {num_clusters}"
            else:
                num_clusters = 0
                
            model_name = "_".join(name.split('_')[1:-3]) if vlad else "_".join(name.split('_')[1:-1])

            def is_integer(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            layer_to_hook = model_name.split('_')[-1]
            if layer_to_hook =="final" or is_integer(layer_to_hook):
                print(f"Using features from {layer_to_hook} layer")
                if is_integer(layer_to_hook):
                    layer_to_hook = int(layer_to_hook)
                model_name = "_".join(model_name.split('_')[:-1])

            else:
                layer_to_hook = "final"
            if model_name == "frozen_dinov2":
                thermal_backbone = ""
            elif model_name == "anythermal":
                thermal_backbone = "pretrained_checkpoints/backbone/AnyThermal_full/model20.pth"
            else:
                raise ValueError(f"Unsupported new mmdistill model name: {model_name}")
            
            if args.same_backbone:
                rgb_backbone = thermal_backbone
            else:
                rgb_backbone = ""

            if modality not in ['rgb', 'thr']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")

            from .dinov2_vpr_model import MMDistillDinov2VLAD
            model_type = getattr(args, "backbone_model_type", "dinov2_vitb14")
            return MMDistillDinov2VLAD(num_clusters = num_clusters,
                                       model_type=model_type,
                                       modality=modality,
                                       backbone_path=thermal_backbone if modality =='thr' else rgb_backbone,use_cls=use_cls,layer_to_hook=layer_to_hook)

        elif name.startswith("vpr_mmdistill_salad"):
            modality = name.split('_')[-1]
            model_name = "_".join(name.split('_')[3:-1])
            if model_name == "anythermal_full":
                model_path = "pretrained_checkpoints/vpr/model_35.pth"
            else :  
                raise ValueError(f"Unsupported new mmdistill model name: {model_name}")
            model_dict = torch.load(model_path, map_location='cpu')
            thermal_model_dict = model_dict["thermal_state_dict"]
            if "rgb_state_dict" in model_dict:
                rgb_model_dict = model_dict["rgb_state_dict"]
            else:
                print("No RGB state dict found, using thermal state dict for RGB")
                rgb_model_dict = thermal_model_dict
            
            
            if args.same_backbone:
                rgb_model_dict = thermal_model_dict

            if modality not in ['rgb', 'thr']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")

            from .dinov2_vpr_model import MMDistillVPRModel
            return MMDistillVPRModel(args=args,frozen_backbone=True,frozen_head=True,modality=modality,model_dict = thermal_model_dict if modality =='thr' else rgb_model_dict)
        else:
            raise ValueError(f"Model name '{name}' not recognized.")
    elif task == 'segmentation':
        if name.startswith("mmdistill_cart"):

            model_name = "_".join(name.split('_')[2:])

            backbone_model_type = ""
            from .dinov2_segmentation_model import MMDistillSegmentationModel

            if model_name == "anythermal":
                model_path = "pretrained_checkpoints/segmentation/cart/model110.pth"
                head_model = "non_linear_64"
            else:
                raise ValueError(f"Unsupported mmdistill cart model name: {model_name}")
            return MMDistillSegmentationModel(args=args,backbone_model_type=backbone_model_type,head_model=head_model,un_frozen_layer_index=[],frozen_head=True,device='cuda',modality="thr",model_path=model_path, **kwargs)
        elif name.startswith("mmdistill_mfnet"):
            model_name = "_".join(name.split('_')[2:])
            backbone_model_type = ""
            if model_name == "anythermal":
                model_path = "pretrained_checkpoints/segmentation/mfnet/model25.pth"
                head_model = "non_linear_64"
            else:
                raise ValueError(f"Unsupported mmdistill mfnet model name: {model_name}")
            
            from .dinov2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(args=args,backbone_model_type=backbone_model_type,head_model=head_model,un_frozen_layer_index=[],frozen_head=True,device='cuda',modality="thr",model_path=model_path, **kwargs)
        elif name.startswith("rtfnet"):
            num_resnet_layers = int(name.split('_')[1])
            if num_resnet_layers not in [18, 34, 50, 101, 152]:
                raise ValueError(f"Unsupported ResNet type: {num_resnet_layers}")
            from .rtfnet_model import RTFNetModel
            return RTFNetModel(device='cuda', num_resnet_layers=num_resnet_layers, **kwargs)
        elif name == "mcnet":
            from .mcnet_model import MCNetModel
            return MCNetModel(device = 'cuda',**kwargs)