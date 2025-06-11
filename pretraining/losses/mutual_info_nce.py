
from .base import *

# ========== 1. Mutual Information (InfoNCE-style) ==========
class MutualInformationInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_feats, positive_feats):
        """
        Args:
            anchor_feats: (B, D) - thermal (student) global features
            positive_feats: (B, D) - RGB (teacher) global features
        Returns:
            InfoNCE-based MI loss
        """
        anchor_feats = F.normalize(anchor_feats, dim=1)
        positive_feats = F.normalize(positive_feats, dim=1)

        logits = torch.matmul(anchor_feats, positive_feats.T) / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)


