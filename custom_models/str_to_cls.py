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
            # elif model_name == "combine_global_contrastive_all_equal_10":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_freiburg_ms2_sthereo_vivid/rgb_thr/2025-07-17_20-13-17_dinov2_vitb14_ms2_vivid_sthereo_cart_boson_freiburg_thr_distill_all_layers_loss_gloabal_contrastive_final_cart_boson_rescaled_equal_samples/model10.pth"
            # elif model_name == "combine_global_contrastive_no_boson":
            #     thermal_backbone ="/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_freiburg_vivid_sthereo_cart/rgb_thr/2025-07-28_04-11-19_dinov2_vitb14_ms2_freiburg_vivid_sthereo_cart_thr_distill_all_layers_loss_global_contrastive_final/model14.pth"
            # elif model_name == "combine_global_contrastive_no_cart":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_ms2_sthereo_vivid/rgb_thr/2025-07-29_01-08-19_dinov2_vitb14_boson_freiburg_ms2_sthereo_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model47.pth"
            # elif model_name == "combine_global_contrastive_no_ms2":
            #     thermal_backbone ="/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_freiburg_sthereo_vivid/rgb_thr/2025-07-29_01-09-19_dinov2_vitb14_boson_cart_freiburg_sthereo_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model100.pth"
            # elif model_name == "combine_global_contrastive_salad_backbone":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_freiburg_ms2_sthereo_vivid/rgb_thr/2025-07-30_18-55-07_dinov2_vitb14_boson_cart_freiburg_ms2_sthereo_vivid_thr_distill_all_layers_loss_gloabal_contrastive_final_equal_samples_salad_backbone_all_equal_samples/model21.pth"
            # elif model_name == "salad_init_combine_global_contrastive_no_boson":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart_freiburg_ms2_sthereo_vivid/rgb_thr/2025-07-30_18-55-07_dinov2_vitb14_cart_freiburg_ms2_sthereo_vivid_thr_distill_all_layers_loss_gloabal_contrastive_final_equal_samples_salad_backbone_no_boson_equal_samples/model23.pth"
            

            # elif model_name == "ms2":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-07-28_04-09-11_dinov2_vitb14_ms2_thr_distill_all_layers_loss_global_contrastive_final/model90.pth"
            # elif model_name == "ms2_vivid":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_vivid/rgb_thr/2025-07-30_20-39-20_dinov2_vitb14_ms2_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model20.pth"
            # elif model_name == "ms2_freiburg":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_freiburg/rgb_thr/2025-07-29_00-19-56_dinov2_vitb14_ms2_freiburg_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model58.pth"
            # elif model_name == "ms2_vivid_freiburg":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_freiburg_vivid/rgb_thr/2025-07-29_00-20-14_dinov2_vitb14_ms2_freiburg_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model37.pth"
            # elif model_name == "ms2_vivid_freiburg_sthereo":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2_freiburg_vivid_sthereo/rgb_thr/2025-07-29_00-20-14_dinov2_vitb14_ms2_freiburg_vivid_sthereo_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model57.pth"
            # elif model_name == "ms2_vivid_freiburg_sthereo_boson":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_ms2_sthereo_vivid/rgb_thr/2025-07-29_01-08-19_dinov2_vitb14_boson_freiburg_ms2_sthereo_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model47.pth"


            # #No MS2 test

            # elif model_name == "vivid":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vivid/rgb_thr/2025-08-03_21-32-13_dinov2_vitb14_vivid_thr_distill_no_ms2_scale_test_equal_samples/model100.pth"
            # elif model_name == "vivid_freiburg":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/freiburg_vivid/rgb_thr/2025-08-03_21-32-13_dinov2_vitb14_freiburg_vivid_thr_distill_no_ms2_scale_test_equal_samples/model79.pth"
            # elif model_name == "vivid_freiburg_sthereo":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/freiburg_sthereo_vivid/rgb_thr/2025-08-03_21-32-13_dinov2_vitb14_freiburg_sthereo_vivid_thr_distill_no_ms2_scale_test_equal_samples/model100.pth"
            # elif model_name == "vivid_freiburg_sthereo_boson":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_sthereo_vivid/rgb_thr/2025-08-03_21-32-13_dinov2_vitb14_boson_freiburg_sthereo_vivid_thr_distill_no_ms2_scale_test_equal_samples/model100.pth"
            # # elif model_name == "combine_global_contrastive_no_ms2": USe this also
            


            # elif model_name == "cart":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/cart/rgb_thr/2025-07-28_04-09-30_dinov2_vitb14_cart_thr_distill_all_layers_loss_global_contrastive_final/model100.pth"
            # elif model_name == "cart_boson":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart/rgb_thr/2025-07-30_20-43-09_dinov2_vitb14_boson_cart_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model42.pth"
            # elif model_name == "cart_boson_ms2":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_ms2/rgb_thr/2025-07-30_20-43-08_dinov2_vitb14_boson_cart_ms2_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model26.pth"
            # elif model_name == "cart_boson_ms2_freiburg":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_freiburg_ms2/rgb_thr/2025-07-30_20-43-26_dinov2_vitb14_boson_cart_freiburg_ms2_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model30.pth"
            # elif model_name == "cart_boson_ms2_freiburg_vivid":
            #     thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_cart_freiburg_ms2_vivid/rgb_thr/2025-07-30_20-43-26_dinov2_vitb14_boson_cart_freiburg_ms2_vivid_thr_distill_all_layers_loss_global_contrastive_final_equal_samples/model30.pth"
            

            elif model_name == "ms2":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/ms2/rgb_thr/2025-08-05_21-09-34_dinov2_vitb14_ms2_thr_distill_equal_samples/model10.pth"
            elif model_name == "ms2_freiburg":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/freiburg_ms2/rgb_thr/2025-08-05_21-09-33_dinov2_vitb14_freiburg_ms2_thr_distill_equal_samples/model10.pth"
            elif model_name == "ms2_freiburg_sthereo":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/freiburg_ms2_sthereo/rgb_thr/2025-08-05_21-09-33_dinov2_vitb14_freiburg_ms2_sthereo_thr_distill_equal_samples/model10.pth"
            elif model_name == "ms2_freiburg_sthereo_boson":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_ms2_sthereo/rgb_thr/2025-08-05_21-09-33_dinov2_vitb14_boson_freiburg_ms2_sthereo_thr_distill_equal_samples/model10.pth"
            elif model_name == "ms2_freiburg_sthereo_boson_vivid":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_freiburg_ms2_sthereo_vivid/rgb_thr/2025-08-05_21-09-33_dinov2_vitb14_boson_freiburg_ms2_sthereo_vivid_thr_distill_equal_samples/model10.pth"
            elif model_name == "boson":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson/rgb_thr/2025-08-05_21-09-49_dinov2_vitb14_boson_thr_distill_equal_samples/model10.pth"
            elif model_name == "boson_ms2":
                thermal_backbone = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/boson_ms2/rgb_thr/2025-08-05_21-09-49_dinov2_vitb14_boson_ms2_thr_distill_equal_samples/model10.pth"
            
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
            # if model_name == "salad_frozen_normal_backbone":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-02-26no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_normal_salad_backbone":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-02-26mmdistill_init_salad_backbone_no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_frozen_normal_backbone_32":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-04-40no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_32_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_normal_salad_backbone_32":
            #     model_path="/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-04-40mmdistill_init_salad_backbone_no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_32_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_frozen_normal_backbone_different":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-02-26no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_margin_0.3_same_backboneFalse_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_normal_salad_backbone_different":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-02-26mmdistill_init_salad_backbone_no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_margin_0.3_same_backboneFalse_frozen_backbone_True_un_frozen_layer_index_/model_10.pth"
            # elif model_name == "salad_frozen_normal_backbone_32_different":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-04-40no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_32_margin_0.3_same_backboneFalse_frozen_backbone_True_un_frozen_layer_index_/model_9.pth"
            # elif model_name == "salad_normal_salad_backbone_32_different":
            #     model_path="/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_ms2_vivid_sthereo_boson/2025-07-20_20-04-40mmdistill_init_salad_backbone_no_allign_loss_cart_ms2_vivid_sthereo_boson_salad_32_margin_0.3_same_backboneFalse_frozen_backbone_True_un_frozen_layer_index_/model_9.pth"
            

            # #test for effect of just using one dataset vs all for thraining, eval on the same (one) dataset
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_all":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_ms2_sthereo_vivid/bn_all_dataset_vpr_salad_all_ms2_freiburg_vivid_sthereo_boson_cart/model_11.pth"
            # elif model_name == "salad_backbone_salad_init_combined_backbone_hard_triplet_vpr_all":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_ms2_sthereo_vivid/2025-07-31_00-41-15salad_init_boson_cart_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_19.pth"
            # elif model_name == "salad_normal_ms2_hard_triplet_vpr_ms2":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2/bn_ms2_vpr_salad_ms2/model_19.pth"
            # elif model_name == "salad_normal_boson_hard_triplet_vpr_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson/bn_boson_vpr_salad_boson/model_200.pth"
            # elif model_name == "salad_normal_cart_hard_triplet_vpr_cart":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart/bn_cart_vpr_salad_cart/model_25.pth"
            

            # # test for effect of scaling data in both backbone and vpr head 
            # elif model_name == "salad_normal_ms2_vivid_hard_triplet_vpr_ms2_vivid":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2_vivid/2025-07-31_00-21-38mmdistill_ms2_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_6.pth"
            # # elif model_name == "salad_normal_ms2_freiburg_hard_triplet_vpr_ms2_freiburg":
            # #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2/2025-07-29_10-24-23mmdistill_freiburg_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            # elif model_name == "salad_normal_ms2_freiburg_vivid_hard_triplet_vpr_ms2_freiburg_vivid":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2_vivid/2025-07-29_10-24-23mmdistill_freiburg_ms2_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            # elif model_name == "salad_normal_ms2_freiburg_vivid_sthereo_hard_triplet_vpr_ms2_freiburg_vivid_sthereo":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2_sthereo_vivid/2025-07-29_10-24-23mmdistill_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_24.pth"
            # elif model_name == "salad_normal_ms2_freiburg_vivid_sthereo_boson_hard_triplet_vpr_ms2_freiburg_vivid_sthereo_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo_vivid/2025-07-29_10-24-23mmdistill_boson_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_21.pth"
            

            # elif model_name == "salad_normal_boson_cart_hard_triplet_vpr_boson_cart":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart/2025-07-31_11-42-12mmdistill_boson_cart_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_47.pth"
            # elif model_name == "salad_normal_boson_cart_ms2_hard_triplet_vpr_boson_cart_ms2":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_ms2/2025-07-31_11-42-12mmdistill_boson_cart_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_29.pth"
            # elif model_name == "salad_normal_boson_cart_ms2_freiburg_hard_triplet_vpr_boson_cart_ms2_freiburg":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_ms2/2025-07-31_11-11-33mmdistill_boson_cart_freiburg_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_17.pth"
            # elif model_name == "salad_normal_boson_cart_ms2_freiburg_vivid_hard_triplet_vpr_boson_cart_ms2_freiburg_vivid":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_ms2_vivid/2025-07-31_11-11-33mmdistill_boson_cart_freiburg_ms2_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_11.pth"
            
            # # for the test if you have a general backbone then what is the trend of icnreasing data for just the VPR head
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_ms2":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2/bn_all_dataset_vpr_salad_ms2/model_6.pth"
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_ms2_freiburg":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2/bn_all_dataset_vpr_salad_ms2_freiburg/model_10.pth"
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_ms2_freiburg_vivid":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2_vivid/bn_all_dataset_vpr_salad_ms2_freiburg_vivid/model_15.pth"
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_ms2_freiburg_vivid_sthereo":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2_sthereo_vivid/bn_all_dataset_vpr_salad_ms2_freiburg_sthereo/model_24.pth"
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_ms2_freiburg_vivid_sthereo_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo_vivid/bn_all_dataset_vpr_salad_ms2_freiburg_vivid_sthereo_boson/model_9.pth"
            # elif model_name == "salad_normal_combined_backbone_hard_triplet_vpr_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson/bn_all_dataset_vpr_salad_boson/model_171.pth"
            
            
            
            # elif model_name == "salad_normal_no_cart_hard_triplet_vpr_no_cart": #same as salad_normal_ms2_freiburg_vivid_sthereo_boson_hard_triplet_vpr_ms2_freiburg_vivid_sthereo_boson
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo_vivid/2025-07-29_10-24-23mmdistill_boson_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_21.pth"
            # elif model_name == "salad_normal_no_ms2_hard_triplet_vpr_no_ms2":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_sthereo_vivid/2025-07-30_18-13-06mmdistill_boson_cart_freiburg_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_22.pth"
            # elif model_name == "salad_normal_no_boson_hard_triplet_vpr_no_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_freiburg_ms2_sthereo_vivid/2025-07-29_10-24-23mmdistill_cart_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_22.pth"
            # elif model_name == "salad_backbone_salad_init_no_boson_hard_triplet_vpr_no_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_freiburg_ms2_sthereo_vivid/2025-07-31_00-41-15salad_init_cart_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_22.pth"
            

            # #frozen RGB dino v2 backbone

            # elif model_name == "frozen_rgb_dinov2_no_ms2":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_sthereo_vivid/2025-08-04_01-07-15frozen_backbone_no_ms2_boson_cart_freiburg_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_22.pth"
            # elif model_name == "frozen_rgb_dinov2_no_boson":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/cart_freiburg_ms2_sthereo_vivid/2025-08-04_01-07-15frozen_backbone_no_boson_cart_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_22.pth"
            # elif model_name == "frozen_rgb_dinov2_no_cart":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo_vivid/2025-08-04_01-07-15frozen_backbone_no_cart_boson_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_21.pth"
            # elif model_name == "frozen_rgb_dinov2_all":
            #     model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_cart_freiburg_ms2_sthereo_vivid/2025-08-05_07-17-59frozen_backbone_all_boson_cart_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_13.pth"
            

            if model_name == "ms2":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/ms2/2025-08-06_00-17-48mmdistill_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            elif model_name == "ms2_freiburg":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2/2025-08-06_00-17-48mmdistill_freiburg_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            elif model_name == "ms2_freiburg_sthereo":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/freiburg_ms2_sthereo/2025-08-06_00-17-48mmdistill_freiburg_ms2_sthereo_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            elif model_name == "ms2_freiburg_sthereo_boson":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo/2025-08-06_08-45-03mmdistill_boson_freiburg_ms2_sthereo_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            elif model_name == "ms2_freiburg_sthereo_boson_vivid":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo_vivid/2025-08-06_08-45-03mmdistill_boson_freiburg_ms2_sthereo_vivid_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_21.pth"
            
            elif model_name == "ms2_freiburg_sthereo_boson_margin_0.2":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo/2025-08-09_02-43-37mmdistill_boson_freiburg_ms2_sthereo_salad_margin_0.2_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"

            elif model_name == "boson":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson/2025-08-06_00-18-09mmdistill_boson_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            elif model_name == "boson_ms2":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_ms2/2025-08-06_00-17-48mmdistill_boson_ms2_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet_equal_samples/model_25.pth"
            
            elif model_name == "frozen_ms2_freiburg_sthereo_boson":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/vpr/boson_freiburg_ms2_sthereo/2025-08-06_08-45-03frozen_backbone_no_cart_no_vivid_boson_freiburg_ms2_sthereo_salad_margin_0.3_same_backboneTrue_frozen_backbone_True_un_frozen_layer_index_hard_triplet/model_25.pth"
            
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
            if model_name == "non_linear_64":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250724-135945_cart_thr_non_linear_64_weighted_ce_bilinear_combined_global_contrastive_dropout0.1/model9.pth"
                head_model = "non_linear_64"
            elif model_name == "non_linear_64_no_dropout":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250721-224011_cart_thr_non_linear_64_weighted_ce_bilinear_combined_global_contrastive/model5.pth"
                head_model = "non_linear_64"
            elif model_name == "non_linear_64_dice":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250724-122232_cart_thr_non_linear_64_dice_bilinear_combined_global_contrastive_dropout0.1/model15.pth"
                head_model = "non_linear_64"
            elif model_name == "frozen_non_linear_64":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250724-134730_cart_thr_non_linear_64_weighted_ce_bilinear_frozen_rgb_dinov2_dropout0.1/model9.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            elif model_name == "frozen_non_linear_64_no_dropout":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250721-223926_cart_thr_non_linear_64_weighted_ce_bilinear_frozen_rgb_dinov2/model5.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            elif model_name == "frozen_non_linear_64_dice":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250724-131212_cart_thr_non_linear_64_dice_bilinear_frozen_rgb_dinov2_dropout0.1/model15.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            elif model_name == "non_linear_64_salad_init":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250721-233300_cart_thr_non_linear_64_weighted_ce_bilinear_salad_initilisation_global_contrastive_dropout0.2/model25.pth"
                head_model = "non_linear_64"
            elif model_name == "non_linear_64_salad_init_dice":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250724-122315_cart_thr_non_linear_64_dice_bilinear_salad_initilisation_global_contrastive_dropout0.1/model25.pth"
                head_model = "non_linear_64"
            elif model_name == "linear":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250723-210753_cart_thr_linear_weighted_ce_bilinear_combined_global_contrastive_dropout0.2/model6.pth"
                head_model = "linear"
            elif model_name == "frozen_linear":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250723-210753_cart_thr_linear_weighted_ce_bilinear_frozen_rgb_dinov2_dropout0.2/model7.pth"
                head_model = "linear"
                backbone_model_type = "dinov2_vitb14"
            elif model_name == "linear_salad_init":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/cart/20250723-211549_cart_thr_linear_weighted_ce_bilinear_salad_initilisation_global_contrastive_dropout0.2/model20.pth"
                head_model = "linear"
            from .dinov2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(args=args,backbone_model_type=backbone_model_type,head_model=head_model,un_frozen_layer_index=[],frozen_head=True,device='cuda',modality="thr",model_path=model_path, **kwargs)
        elif name.startswith("mmdistill_mfnet"):
            model_name = "_".join(name.split('_')[2:])
            backbone_model_type = ""
            if model_name == "non_linear_64":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250809-052432_mfnet_thr_non_linear_64_dice_bilinear_thermal_dinov2/model40.pth"
                head_model = "non_linear_64"
            elif model_name == "frozen_non_linear_64":
                model_path = "/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints/segmentation/mfnet/20250809-034404_mfnet_thr_non_linear_64_dice_bilinear_frozen_rgb_dinov2/model40.pth"
                head_model = "non_linear_64"
                backbone_model_type = "dinov2_vitb14"
            else:
                raise ValueError(f"Unsupported mmdistill mfnet model name: {model_name}")
            
            from .dinov2_segmentation_model import MMDistillSegmentationModel
            return MMDistillSegmentationModel(args=args,backbone_model_type=backbone_model_type,head_model=head_model,un_frozen_layer_index=[],frozen_head=True,device='cuda',modality="thr",model_path=model_path, **kwargs)
