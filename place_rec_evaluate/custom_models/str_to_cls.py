from .alexnet_model import AlexNetFeatureExtractor
from .vgg16_model import VGG16FeatureExtractor
from .resnet_model import ResNetFeatureExtractor
from .squeezenet_model import SqueezeNetFeatureExtractor

def get_model_from_string(name: str):
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
            return DINOv2FeatureExtractor(model_type=name)
        elif elements[-1] == 'variable':
            from .dinov2_model import DINOv2FeatureExtractor_Variable
            model_name = '_'.join(elements[:-1])
            return DINOv2FeatureExtractor_Variable(model_type=model_name)
        else:
            raise ValueError(f"Unsupported DINOv2 model name: {name}")
    
    elif name.startswith('mmdistill_dinov2'):
        modality = name.split('_')[-1]
        if modality not in ['rgb', 'thermal']:
            raise ValueError(f"Unsupported mmdistill_dinov2 modality: {modality}")
        if modality == 'rgb':
            image_input_mode = name.split('_')[2]
            if image_input_mode not in ['fixed', 'variable']:
                raise ValueError(f"Unsupported mmdistill_dinov2 image input mode: {image_input_mode}")
            
            if image_input_mode == 'fixed':
                from .mmdistill_dinov2_model import FixedRGBDistillDINOv2FeatureExtractor
                return FixedRGBDistillDINOv2FeatureExtractor()
            elif image_input_mode == 'variable':
                # This is a placeholder for the variable input mode
                # You can implement the corresponding class as needed
                from .mmdistill_dinov2_model import VariableRGBDistillDINOv2FeatureExtractor
                return VariableRGBDistillDINOv2FeatureExtractor()
        elif modality == 'thermal':
            image_input_mode = name.split('_')[2]
            if image_input_mode not in ['fixed', 'variable']:
                raise ValueError(f"Unsupported mmdistill_dinov2 image input mode: {image_input_mode}")
            
            if image_input_mode == 'fixed':
                from .mmdistill_dinov2_model import FixedThermalDistillDINOv2FeatureExtractor
                return FixedThermalDistillDINOv2FeatureExtractor()
            elif image_input_mode == 'variable':
                # This is a placeholder for the variable input mode
                # You can implement the corresponding class as needed
                from .mmdistill_dinov2_model import VariableThermalDistillDINOv2FeatureExtractor
                return VariableThermalDistillDINOv2FeatureExtractor()

    elif name == "salad":
        from .dinov2salad_model import DinoV2SALADFeatureExtractor
        return DinoV2SALADFeatureExtractor()
    
    else:
        raise ValueError(f"Model name '{name}' not recognized.")
