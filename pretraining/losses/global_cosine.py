from .base import *


class GlobalCosineLoss(nn.Module):
    def __init__(self, return_mean=True):
        super().__init__()
        self.return_mean = return_mean

    def forward(self, student_output, teacher_output):
        teacher_embed = teacher_output[1]  # (B, D)
        student_embed = student_output[1]  # (B, D)
        teacher_embed = F.normalize(teacher_embed.detach(), dim=-1)
        student_embed = F.normalize(student_embed, dim=-1)

        cos_sim = F.cosine_similarity(student_embed, teacher_embed, dim=-1)
        loss = 1 - cos_sim
        return loss.mean() if self.return_mean else loss
