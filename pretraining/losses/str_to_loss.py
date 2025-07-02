from .global_contrastive import GlobalContrastiveLoss
from .global_cosine import GlobalCosineLoss
from .attention_alignment import AttentionAlignmentLoss
from .patch_nce import PatchNCELoss,PatchNCELossSinglePositive,GlobalAndPatchCosine, PatchCosine
from .vic_reg import InvarianceLoss, VarianceLoss, CovarianceLoss, CosineRepulsionLoss
from .shared_specific import ModalitySharedandSpecificPositiveLoss,ModalitySharedandSpecificNegativeLoss

str_to_loss_dict = {"global_contrastive": GlobalContrastiveLoss,
                    "global_cosine": GlobalCosineLoss,
                    "attention_alignment": AttentionAlignmentLoss,
                    "patch_nce": PatchNCELoss,
                    "patch_nce_single_positive": PatchNCELossSinglePositive,
                    "global_and_patch_cosine": GlobalAndPatchCosine,
                    "patch_cosine": PatchCosine,
                    "modality_shared_and_specific_pos": ModalitySharedandSpecificPositiveLoss,
                    "modality_shared_and_specific_neg": ModalitySharedandSpecificNegativeLoss,
                    "invariance_loss": InvarianceLoss,
                    "variance_loss": VarianceLoss,
                    "covariance_loss": CovarianceLoss,
                    "cosine_repulsion_loss": CosineRepulsionLoss
                    }