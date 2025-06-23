from .global_contrastive import GlobalContrastiveLoss
from .global_cosine import GlobalCosineLoss
from .attention_alignment import AttentionAlignmentLoss
from .patch_nce import PatchNCELoss,PatchNCELossSinglePositive


str_to_loss_dict = {"global_contrastive": GlobalContrastiveLoss,
                    "global_cosine": GlobalCosineLoss,
                    "attention_alignment": AttentionAlignmentLoss,
                    "patch_nce": PatchNCELoss,
                    "patch_nce_single_positive": PatchNCELossSinglePositive}