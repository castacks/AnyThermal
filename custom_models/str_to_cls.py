from .alexnet_model import AlexNetFeatureExtractor
from .vgg16_model import VGG16FeatureExtractor
from .resnet_model import ResNetFeatureExtractor
from .squeezenet_model import SqueezeNetFeatureExtractor

def get_model_from_string(args,name: str,task,**kwargs):
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
            if modality not in ['rgb', 'thr']:
                raise ValueError(f"Unsupported ImageBind modality: {modality}")
            if modality == 'rgb':
                from .imagebind_model import ImageBindRGBFeatureExtractor
                return ImageBindRGBFeatureExtractor()
            elif modality == 'thermal':
                from .imagebind_model import ImageBindThermalFeatureExtractor
                return ImageBindThermalFeatureExtractor()

        elif name == "salad":
            from .dinov2salad_model import DinoV2SALADFeatureExtractor
            return DinoV2SALADFeatureExtractor()
        elif name.startswith("salad_mmdistill_dinov2"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thr']:
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
        elif name.startswith("mmdistill"):
            modality = name.split('_')[-1]
    
            vlad = True if name.split('_')[-3] == 'vlad' else False
            if vlad:
                num_clusters = int(name.split('_')[-2])
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
            assert layer_to_hook =="final" or is_integer(layer_to_hook), f"layer_to_hook should be 'final' or an integer, got {layer_to_hook}"
            if is_integer(layer_to_hook):
                layer_to_hook = int(layer_to_hook)
            model_name = "_".join(model_name.split('_')[:-1])
            if model_name == "combine_contrastive_all":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-08_17-16-47_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_combined_all_layers_loss_global_contrastive_final/model4.pth"
            elif model_name == "combine_both_cosine_all":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-08_17-15-24_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_combined_all_layers_loss_both_cosine_final/model2.pth"
            elif model_name == "cart_only_contrastive_all":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart/rgb_thr/2025-07-11_12-38-19_dinov2_vitb14_cart_thr_distill_all_layers_loss_global_contrastive_final/model100.pth"
            elif model_name == "ms2_only_contrastive_all":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-07-11_12-38-19_dinov2_vitb14_ms2_thr_distill_all_layers_loss_global_contrastive_final/model98.pth"
            elif model_name == "frozen_dinov2":
                thermal_backbone =""
            elif model_name == "combine_both_contrastive_all":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-12_14-45-55_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_all_layers_loss_both_contrastive_final/model13.pth"
            elif model_name == "combine_contrastive_intradataset":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-13_13-43-37_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_all_layers_loss_global_contrastive_final/model6.pth"
            elif model_name == "combine_contrastive_intradataset_rescale":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-13_12-30-55_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_all_layers_loss_global_contrastive_final_rescale/model6.pth"
            elif model_name == "combine_contrastive_intradataset_rescale_no_intradataset":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid_sthereo_cart_boson_freiburg/rgb_thr/2025-07-13_12-29-30_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_all_layers_loss_global_contrastive_final_rescale_no_intradataset/model6.pth"
            else:
                raise ValueError(f"Unsupported new mmdistill model name: {model_name}")
            
            if args.same_backbone:
                rgb_backbone = thermal_backbone
            else:
                rgb_backbone = ""

            if modality not in ['rgb', 'thr']:
                raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")

            from .dinov2_vpr_model import MMDistillDinov2VLAD
            return MMDistillDinov2VLAD(num_clusters = num_clusters, 
                                       model_type="dinov2_vitb14", 
                                       modality=modality, 
                                       backbone_path=thermal_backbone if modality =='thr' else rgb_backbone,use_cls=True,layer_to_hook=layer_to_hook)

        elif name.startswith("netvlad_mmdistill_dinov2_cart"):
            modality = name.split('_')[-1]
            if modality not in ['rgb', 'thr']:
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
            if modality not in ['rgb', 'thr']:
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
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250613-230300/model44.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132308/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_only_patch_nce":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132431/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_allign_1":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132359/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_ms2_allign_1000":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,model_type='dinov2_vitb14',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/dinov2_vitb14/20250618-132324/model39.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive_weighted_ce":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-221832_decart_weighted_ce_base_head_padding/model20.pth', **kwargs)
        elif name =="mmdistill_dinov2_combined_model_weighted_ce":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-222151_combined_model_wo_cart_base_weighted_ce/model20.pth', **kwargs)
        elif name =="mmdistill_dinov2_cart_only_contrastive_weighted_ce_dpt":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(frozen_backbone=True,head_model='dpt',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250620-223200_cart_weighted_ce_dpt/model18.pth', **kwargs)
        elif name =="frozen_rgb_deep_seg_head":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='base',modality="rgb",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-180417_cart_rgb_frozen/model39.pth', **kwargs)
        elif name =="frozen_thr_deep_seg_head":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='base',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-174137_cart_thr_final_norm_trained_bilinear/model39.pth', **kwargs)
        elif name =="frozen_thr_non_linear_128_head_dice":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_128',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-204548_cart_thr_non_linear_128_final_norm_trained_bilinear_dice_also/model37.pth', **kwargs)
        elif name =="frozen_thr_non_linear_64_head_dice":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_64',modality="thr",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-211800_cart_thr_non_linear_64_final_norm_trained_bilinear_dice_also/model36.pth', **kwargs)
        elif name =="frozen_rgb_non_linear_128_head_dice":
            from .dinv2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(upscale_method='bilinear',un_frozen_layer_index=[],head_model='non_linear_128',modality="rgb",device='cuda', model_path='/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250621-220006_cart_rgb_non_linear_128_final_norm_trained_bilinear_dice_also/model37.pth', **kwargs)