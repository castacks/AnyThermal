from .base import *

class AttentionAlignmentLoss(nn.Module):
    def __init__(self, mode='l1'):
        super().__init__()
        assert mode in ['l1', 'kl']
        self.mode = mode
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output, mask=None):
        student_tokens = student_output[0]  # (B, D, H, W)
        teacher_tokens = teacher_output[0]  # (B, D, H, W)

        B, D, H, W = student_tokens.shape
        student_tokens = student_tokens.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        teacher_tokens = teacher_tokens.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)

        attn_s = torch.norm(student_tokens, dim=-1)
        attn_t = torch.norm(teacher_tokens, dim=-1)

        if mask is not None:
            attn_s = attn_s * mask
            attn_t = attn_t * mask
            sum_s = attn_s.sum(dim=1, keepdim=True) + 1e-6
            sum_t = attn_t.sum(dim=1, keepdim=True) + 1e-6
        else:
            sum_s = attn_s.sum(dim=1, keepdim=True) + 1e-6
            sum_t = attn_t.sum(dim=1, keepdim=True) + 1e-6

        attn_s = attn_s / sum_s
        attn_t = attn_t / sum_t

        if self.mode == 'l1':
            return F.l1_loss(attn_s, attn_t)
        else:
            return self.kl(torch.log(attn_s + 1e-6), attn_t)
