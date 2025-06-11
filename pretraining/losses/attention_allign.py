from .base import *

class AttentionAlignmentLoss(nn.Module):
    def __init__(self, mode='l1'):
        """
        Args:
            mode (str): 'l1' or 'kl' – type of divergence used.
        """
        super().__init__()
        assert mode in ['l1', 'kl'], "Mode must be either 'l1' or 'kl'"
        self.mode = mode
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_tokens, teacher_tokens, mask=None):
        """
        Args:
            student_tokens: (B, N, D) – ViT tokens from thermal encoder
            teacher_tokens: (B, N, D) – ViT tokens from RGB encoder
            mask: optional (B, N) – binary mask (1 = valid, 0 = ignore)
        Returns:
            Scalar loss between normalized attention maps
        """
        # Compute spatial attention as L2 norm over D
        attn_s = torch.norm(student_tokens, dim=-1)  # (B, N)
        attn_t = torch.norm(teacher_tokens, dim=-1)  # (B, N)

        if mask is not None:
            attn_s = attn_s * mask
            attn_t = attn_t * mask

            # Normalize each attention map over valid positions
            sum_s = (attn_s * mask).sum(dim=1, keepdim=True) + 1e-6
            sum_t = (attn_t * mask).sum(dim=1, keepdim=True) + 1e-6
        else:
            sum_s = attn_s.sum(dim=1, keepdim=True) + 1e-6
            sum_t = attn_t.sum(dim=1, keepdim=True) + 1e-6

        attn_s = attn_s / sum_s
        attn_t = attn_t / sum_t

        if self.mode == 'l1':
            if mask is not None:
                return F.l1_loss(attn_s * mask, attn_t * mask)
            return F.l1_loss(attn_s, attn_t)
        else:
            # Apply log only where mask is valid
            attn_s_log = torch.log(attn_s + 1e-6)
            if mask is not None:
                attn_s_log = attn_s_log * mask
                attn_t = attn_t * mask
            return self.kl(attn_s_log, attn_t)
