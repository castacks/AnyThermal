from .base import *
# ========== 3. KL Divergence Loss ==========
class KLFeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: (B, D)
            teacher_feats: (B, D)
        Returns:
            KL divergence between softmaxed student and teacher features
        """
        student_probs = F.log_softmax(student_feats, dim=1)
        teacher_probs = F.softmax(teacher_feats, dim=1)
        return self.kl(student_probs, teacher_probs)

