from .base import *

class MMDLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: (B, D)
            teacher_feats: (B, D)
        Returns:
            MMD loss between student and teacher distributions
        """
        K_ss = self.rbf_kernel(student_feats, student_feats, self.gamma)
        K_tt = self.rbf_kernel(teacher_feats, teacher_feats, self.gamma)
        K_st = self.rbf_kernel(student_feats, teacher_feats, self.gamma)

        mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        return mmd
    
        # ========== 2. Maximum Mean Discrepancy (MMD) ==========
    def rbf_kernel(self,x, y, gamma=1.0):
        """
        Compute RBF kernel between two sets of vectors.
        """
        x = x.unsqueeze(1)  # (B, 1, D)
        y = y.unsqueeze(0)  # (1, B, D)
        return torch.exp(-gamma * ((x - y) ** 2).sum(dim=2))  # (B, B)


