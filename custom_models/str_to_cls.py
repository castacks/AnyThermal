from .alexnet_model import AlexNetFeatureExtractor
from .vgg16_model import VGG16FeatureExtractor
from .resnet_model import ResNetFeatureExtractor
from .squeezenet_model import SqueezeNetFeatureExtractor
import torch 
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
            assert layer_to_hook =="final" or is_integer(layer_to_hook), f"layer_to_hook should be 'final' or an integer, got {layer_to_hook}"
            if is_integer(layer_to_hook):
                layer_to_hook = int(layer_to_hook)
            model_name = "_".join(model_name.split('_')[:-1])
            if model_name == "frozen_dinov2":
                thermal_backbone =""
            elif model_name == "frozen_salad":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/place_recognition/salad/pretrained_models/salad_backbone.ckpt"
            elif model_name == "thermal_dinov2_all_with_tartan_rgbt":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_sthereo_tartanrgbt_vivid/rgb_thr/2025-09-05_22-53-44_dinov2_vitb14_boson_freiburg_sthereo_tartanrgbt_vivid_thr_distill_no_holes_correct_rectification_ffc_considered_equal_samples/model20.pth"
                
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
                                       backbone_path=thermal_backbone if modality =='thr' else rgb_backbone,use_cls=use_cls,layer_to_hook=layer_to_hook)

        elif name.startswith("vpr_mmdistill_salad"):
            modality = name.split('_')[-1]
            model_name = "_".join(name.split('_')[3:-1])
            if model_name == "all_with_tartan_rgbt_frac_1":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-08_20-34-05thermal_dino_with_correct_tartanrgbt_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_40.pth"
            elif model_name == "all_with_tartan_rgbt_frac_0.5":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-08_20-34-05thermal_dino_with_correct_tartanrgbt_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.5/model_40.pth"
            elif model_name == "all_with_tartan_rgbt_frac_0.75":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-08_20-34-05thermal_dino_with_correct_tartanrgbt_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "anythermal_salad_icra_recrete":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-18_18-18-14anythermal_salad_icra_recreate_mach2_scaling_full_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "frozen_with_all_for_vpr_head":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-09_18-44-09frozen_dino_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"


            # equal samples in VPR training 

            elif model_name == "all_with_tartan_rgbt_frac_0.5_equal_samples":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-13_01-30-09thermal_dino_with_correct_tartanrgbt_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_equal_samples_hard_frac_0.5/model_15.pth"
            elif model_name == "frozen_with_all_for_vpr_head_0.5_equal_samples":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-13_01-30-29frozen_dino_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_equal_samples_hard_frac_0.5/model_15.pth"
            #scaling - vpr and bn use same datasets
            elif model_name == "bn_boson_vpr_same":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson/2025-09-09_18-44-09thermal_dino_scaling_bn_boson_vpr_same_boson_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_vpr_same":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_vivid/2025-09-09_20-16-42thermal_dino_scaling_bn_boson_vivid_vpr_same_boson_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_freiburg_vpr_same":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_vivid/2025-09-09_20-25-01thermal_dino_scaling_bn_boson_vivid_freiburg_vpr_same_boson_freiburg_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_freiburg_sthereo_vpr_same":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_vivid/2025-09-09_20-41-07thermal_dino_scaling_bn_boson_vivid_freiburg_sthereo_vpr_same_boson_freiburg_sthereo_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"

            #scaling - bn usees incremental datasets, vpr uses all

            elif model_name == "bn_boson_vpr_all":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-09_18-44-10thermal_dino_scaling_bn_boson_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_vpr_all":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-09_18-44-09thermal_dino_scaling_bn_boson_vivid_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_freiburg_vpr_all":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-09_18-44-09thermal_dino_scaling_bn_boson_vivid_freiburg_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            elif model_name == "bn_boson_vivid_freiburg_sthereo_vpr_all":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-09-09_18-44-09thermal_dino_scaling_bn_boson_vivid_freiburg_sthereo_vpr_all_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.75/model_35.pth"
            
            
            elif model_name == "anythermal_m0.1_hard_frac_0.6":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-19_21-58-31anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.6/model_35.pth"
            elif model_name == "anythermal_m0.1_hard_frac_0.7":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-19_21-58-31anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.7/model_35.pth"
            elif model_name == "anythermal_m0.1_hard_frac_0.85":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-19_21-58-31anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.1_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.85/model_35.pth"
            elif model_name == "anythermal_m0.05_hard_frac_0.6":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-19_22-35-18anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.05_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.6/model_35.pth"
            elif model_name == "anythermal_m0.05_hard_frac_0.7":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-20_00-35-24anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.05_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.7/model_35.pth"
            elif model_name == "anythermal_m0.05_hard_frac_0.85":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-20_12-20-08anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.05_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.85/model_35.pth"
            elif model_name == "anythermal_m0.15_hard_frac_0.6":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-20_12-47-37anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.15_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.6/model_35.pth"
            elif model_name == "anythermal_m0.15_hard_frac_0.7":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-20_14-28-04anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.15_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.7/model_35.pth"
            elif model_name == "anythermal_m0.15_hard_frac_0.85":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_sthereo_tartanrgbt_vivid/2025-10-20_14-28-04anythermal_salad_icra_recreate_mach3_boson_freiburg_sthereo_tartanrgbt_vivid_salad_margin_0.15_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_hard_frac_0.85/model_35.pth"
            
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
        if name.startswith("mmdistill_cart"):

            model_name = "_".join(name.split('_')[2:])

            backbone_model_type = ""
            from .dinov2_segmentation_model import MMDistillSegmentationModel

            if model_name == "thermal_dinov2_with_tartan_rgbt":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250907-141327_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_with_tarratan_rgbt_without_holes_correct_rectification_ffc_considered_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-035440_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-083730_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_freiburg_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_vivid":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-083734_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg_vivid":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-083730_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_freiburg_vivid_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_vivid_sthereo":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-085920_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_stehreo_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg_vivid_sthereo":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250908-083902_cart_random_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_stehreo_freiburg_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "frozen_rgb_dinov2":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart_random/20250907-141327_cart_random_thr_non_linear_64_weighted_ce_bilinear_rgb_frozen_dinov2_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            
            return MMDistillSegmentationModel(args=args,backbone_model_type=backbone_model_type,head_model=head_model,un_frozen_layer_index=[],frozen_head=True,device='cuda',modality="thr",model_path=model_path, **kwargs)
        elif name.startswith("mmdistill_mfnet"):
            model_name = "_".join(name.split('_')[2:])
            backbone_model_type = ""
            if model_name == "thermal_dinov2_with_tartan_rgbt":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250906-132022_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_with_tarratan_rgbt_without_holes_correct_rectification_ffc_considered_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-014737_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_vivid":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-063856_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-063857_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_freiburg_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg_vivid":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-063857_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_freiburg_vivid_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_freiburg_vivid_sthereo":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-063857_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_stehreo_freiburg_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"
            elif model_name == "thermal_dinov2_scaling_boson_vivid_sthereo":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250908-070205_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2_scaling_boson_vivid_stehreo_augmentedbrightness_contrast_gamma_dropout0.1/model100.pth"
                head_model = "non_linear_64"

            elif model_name == "frozen_rgb_dinov2":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250817-034126_mfnet_thr_non_linear_64_dice_bilinear_frozen_dinov2_augmentedcrop_with_random_ratio_gamma/model100.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            
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