from .alexnet_model import AlexNetFeatureExtractor
from .vgg16_model import VGG16FeatureExtractor
from .resnet_model import ResNetFeatureExtractor
from .squeezenet_model import SqueezeNetFeatureExtractor

def get_model_from_string(name: str,task,**kwargs):
    """
    Returns the corresponding feature extractor class based on the model name.
    Supported:
    - 'alexnet'
    - 'vgg16'
    - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    - 'squeezenet1_0', 'squeezenet1_1'
    - 'netvlad'
    - 'dinov2_vits14' 
    - 'imagebind'
    - 'mixvpr'
    """
    name = name.lower()

    if task =='vpr':
    
        if name == 'alexnet':
            return AlexNetFeatureExtractor()
        
        elif name == 'r2former':
            from .r2former_model import R2FormerFeatureExtractor
            return R2FormerFeatureExtractor()
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
        
        elif name == 'netvlad':
            from .netvlad_model import NetVLADFeatureExtractor
            return NetVLADFeatureExtractor()
        
        elif name=='mixvpr':
            from .mixvpr_model import MixVPRFeatureExtractor
            return MixVPRFeatureExtractor()

        elif name.startswith('imagebind'):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported ImageBind modality: {modality}")
            if modality == 'rgb':
                from .imagebind_model import ImageBindRGBFeatureExtractor
                return ImageBindRGBFeatureExtractor()
            elif modality == 'thermal':
                from .imagebind_model import ImageBindThermalFeatureExtractor
                return ImageBindThermalFeatureExtractor()
        
        elif name.startswith('dinov2'):
            elements = name.split('_')
            if len(elements) == 2:
                from .dinov2_model import DINOv2FeatureExtractor
                return DINOv2FeatureExtractor(model_type=name,use_intermediate_layers=False)
            elif elements[-1] == 'variable':
                from .dinov2_model import DINOv2FeatureExtractor_Variable
                model_name = '_'.join(elements[:-1])
                return DINOv2FeatureExtractor_Variable(model_type=model_name,use_intermediate_layers=False)
            else:
                raise ValueError(f"Unsupported DINOv2 model name: {name}")
        
        elif name.startswith('mmdistill_dinov2'):
            modality = name.split('_')[-1]
            model_type = '_'.join(name.split('_')[3:5])
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                image_input_mode = name.split('_')[2]
                if image_input_mode not in ['fixed', 'variable']:
                    raise ValueError(f"Unsupported mmdistill_dinov2 image input mode: {image_input_mode}")
                
                if image_input_mode == 'fixed':
                    from .mmdistill_dinov2_model import FixedRGBDistillDINOv2FeatureExtractor
                    return FixedRGBDistillDINOv2FeatureExtractor(model_type=model_type,use_intermediate_layers=False)
                elif image_input_mode == 'variable':
                    # This is a placeholder for the variable input mode
                    # You can implement the corresponding class as needed
                    from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                    return VariableRGBDistillDINOv2FeatureExtractor(model_type=model_type,use_intermediate_layers=False)
            elif modality == 'thermal':
                image_input_mode = name.split('_')[2]
                if image_input_mode not in ['fixed', 'variable']:
                    raise ValueError(f"Unsupported mmdistill_dinov2 image input mode: {image_input_mode}")
                
                if image_input_mode == 'fixed':
                    from .mmdistill_dinov2_model import FixedThermalDistillDINOv2FeatureExtractor
                    return FixedThermalDistillDINOv2FeatureExtractor(model_type=model_type,use_intermediate_layers=False)
                elif image_input_mode == 'variable':
                    # This is a placeholder for the variable input mode
                    # You can implement the corresponding class as needed
                    from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                    return VariableThermalDistillDINOv2FeatureExtractor(model_type=model_type,use_intermediate_layers=False)

        elif name == "salad":
            from .dinov2salad_model import DinoV2SALADFeatureExtractor
            return DinoV2SALADFeatureExtractor()
        elif name.startswith("salad_mmdistill_dinov2"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                print(f"Using RGBMMDistillDinoV2SALADFeatureExtractor model for {name}")
                from .dinov2salad_model import RGBMMDistillDinoV2SALADFeatureExtractor
                return RGBMMDistillDinoV2SALADFeatureExtractor()
            elif modality == 'thermal':
                print(f"Using ThermalMMDistillDinoV2SALADFeatureExtractor model for {name}")
                from .dinov2salad_model import ThermalMMDistillDinoV2SALADFeatureExtractor
                return ThermalMMDistillDinoV2SALADFeatureExtractor()
            else:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
        elif name.startswith("cart_train_normal"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart/rgb_thr/2025-06-09_19-39-04/model8.pth')
        elif name.startswith("cart_train_easy"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart/rgb_thr/2025-06-02_03-44-59/model9.pth')
        elif name.startswith("ms2_mmdistill"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-06-09_19-43-06/model19.pth')
        elif name.startswith("combined_mmdistill"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart_ms2_freiburg_sthereo_vivid/rgb_thr/2025-06-10_14-56-00/model20.pth')
        elif name.startswith("combined_5_nocosine"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_sthereo_vivid_boson_freiburg/rgb_thr/2025-06-18_01-16-56_no_cosine/model15.pth')
        elif name.startswith("combined_5_with_cosine"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor(model_type="dinov2_vitb14",use_intermediate_layers=False,backbone_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_sthereo_vivid_boson_freiburg/rgb_thr/2025-06-18_01-19-58_cosine_also/model15.pth')
        

        elif name.startswith("netvlad_mmdistill_dinov2_cart"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import MMDistillVPRModel
                return MMDistillVPRModel(frozen_backbone=True, frozen_head=True,modality="rgb",model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/model_15.pth') 
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import MMDistillVPRModel
                return MMDistillVPRModel(frozen_backbone=True, frozen_head=True,modality="thermal",model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/model_15.pth')
        elif name.startswith("netvlad_mmdistill_dinov2_ms2"):
            modality = name.split('_')[-1]
            head_config={"agg_arch":"NetVLAD"}
            if modality not in ['rgb', 'thermal']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
            if modality == 'rgb':
                from .mmdistill_dinov2_model import MMDistillVPRModel
                return MMDistillVPRModel(frozen_backbone=True, frozen_head=True,modality="rgb",model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2/2025-06-14_22-36-30/model_0.pth', head_config=head_config)
            elif modality == 'thermal':
                from .mmdistill_dinov2_model import MMDistillVPRModel
                return MMDistillVPRModel(frozen_backbone=True, frozen_head=True,modality="thermal",model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2/2025-06-14_22-36-30/model_0.pth', head_config=head_config)
        else:
            raise ValueError(f"Model name '{name}' not recognized.")
    elif task == 'segmentation':
        if name =="mmdistill_dinov2_cart":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250613-230300/model44.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132308/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_only_patch_nce":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132431/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_allign_1":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132359/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_allign_1000":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132324/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive_weighted_ce":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-221832_decart_weighted_ce_base_head_padding/model20.pth', **kwargs)
        elif name =="mmdistill_dinov2_combined_model_weighted_ce":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-222151_combined_model_wo_cart_base_weighted_ce/model20.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive_weighted_ce_dpt":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='dpt',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-223200_cart_weighted_ce_dpt/model18.pth', **kwargs)
        elif name =="frozen_rgb_deep_seg_head":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='base',modality="rgb",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-180417_cart_rgb_frozen/model39.pth', **kwargs)
        elif name =="frozen_thr_deep_seg_head":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-174137_cart_thr_final_norm_trained_bilinear/model39.pth', **kwargs)
        elif name =="frozen_thr_non_linear_128_head_dice":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_128',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-204548_cart_thr_non_linear_128_final_norm_trained_bilinear_dice_also/model37.pth', **kwargs)
        elif name =="frozen_thr_non_linear_64_head_dice":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_64',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-211800_cart_thr_non_linear_64_final_norm_trained_bilinear_dice_also/model36.pth', **kwargs)
        elif name =="frozen_rgb_non_linear_128_head_dice":
            from .mmdistill_dinov2_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_128',modality="rgb",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-220006_cart_rgb_non_linear_128_final_norm_trained_bilinear_dice_also/model37.pth', **kwargs)