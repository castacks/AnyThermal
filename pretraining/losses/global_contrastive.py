from .base import *


class GlobalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        teacher_embed = teacher_output[1]  # (B, D)
        student_embed = student_output[1]  # (B, D)
        teacher_embed = F.normalize(teacher_embed.detach(), dim=-1)
        student_embed = F.normalize(student_embed, dim=-1)

        logits = torch.matmul(teacher_embed, student_embed.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_mask = torch.eye(logits.size(0), dtype=torch.float, device=logits.device)
        return -(log_probs * pos_mask).sum() / (pos_mask.sum() + 1e-8)
