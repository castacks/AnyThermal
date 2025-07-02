from .base import *


class GlobalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def calculate_contrastive_loss(self, student_embed, teacher_embed):
        teacher_embed = F.normalize(teacher_embed, dim=-1)
        student_embed = F.normalize(student_embed, dim=-1)

        logits = torch.matmul(teacher_embed, student_embed.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_mask = torch.eye(logits.size(0), dtype=torch.float, device=logits.device)
        return -(log_probs * pos_mask).sum() / (pos_mask.sum() + 1e-8)


    def forward(self, student_output, teacher_output):
        #teacher_output[0][1] is the cls token for the complete 768 dimensional embedding
        teacher_embed = teacher_output[0][1].detach()  # (B, D)
        student_embed = student_output[0][1]  # (B, D)
        
        return self.calculate_contrastive_loss(student_embed, teacher_embed)

