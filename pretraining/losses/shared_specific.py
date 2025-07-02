from .base import *

class ModalitySharedandSpecificPositiveLoss(nn.Module):
    def __init__(self,mode='global', margin=0.2):
        super().__init__()
        self.margin = margin
        self.clamp_eps = 1e-5
        self.mode = mode

    def cosine_sim(self, a, b):
        sim = torch.sum(a * b, dim=-1)
        return torch.clamp(sim, -1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
    
    def extract_embeddings(self, output):
        if self.mode == 'global':
            shared_embed = output[1][1]     # (B, D)
            specific_embed = output[2][1]
        elif self.mode == 'patch':
            shared_embed = output[1][0]
            specific_embed = output[2][0]
        elif self.mode == "global_and_patch":
            B,D,H,W = output[1][0].shape
            shared_embed_patch = output[1][0].reshape((B, D, -1)).permute(0, 2, 1).reshape((-1,D))
            specific_embed_patch = output[2][0].reshape((B, D, -1)).permute(0, 2, 1).reshape((-1,D))
            shared_embed = torch.cat((shared_embed_patch, output[1][1]), dim=0)
            specific_embed = torch.cat((specific_embed_patch, output[2][1]), dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return shared_embed, specific_embed

    def forward(self, student_output, teacher_output):
        teacher_shared_embed, teacher_specific_embed = self.extract_embeddings(teacher_output)
        student_shared_embed, student_specific_embed = self.extract_embeddings(student_output) 

        pos_loss = F.mse_loss(student_shared_embed, teacher_shared_embed)
        return pos_loss

class ModalitySharedandSpecificNegativeLoss(ModalitySharedandSpecificPositiveLoss):
    def forward(self, student_output, teacher_output):
        teacher_shared_embed, teacher_specific_embed = self.extract_embeddings(teacher_output)
        student_shared_embed, student_specific_embed = self.extract_embeddings(student_output)
        # Normalize all embeddings
        student_shared = F.normalize(student_shared_embed, dim=-1)
        student_specific = F.normalize(student_specific_embed, dim=-1)
        teacher_shared = F.normalize(teacher_shared_embed, dim=-1)
        teacher_specific = F.normalize(teacher_specific_embed, dim=-1)

        # Cosine similarities with clamping
        neg1 = self.cosine_sim(student_specific, teacher_specific)
        neg2 = self.cosine_sim(student_shared, student_specific)
        neg3 = self.cosine_sim(teacher_shared, teacher_specific)

        # Loss terms
        neg_loss1 = F.relu(torch.abs(neg1) - self.margin)
        neg_loss2 = F.relu(torch.abs(neg2) - self.margin)
        neg_loss3 = F.relu(torch.abs(neg3) - self.margin)

        total_loss = neg_loss1 + neg_loss2 + neg_loss3
        return total_loss.mean()

class ModalitySharedandSpecificVarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_output, teacher_output):
        # Extract embeddings
        teacher_shared_embed = teacher_output[1][1]     # (B, D)
        teacher_specific_embed = teacher_output[2][1]   # (B, D)
        student_shared_embed = student_output[1][1]              # (B, D)
        student_specific_embed = student_output[2][1]            # (B, D)
        teacher_global_embed = teacher_output[0][1]     # (B, D)
        student_global_embed = student_output[0][1]              # (B, D)

        # Compute variance loss
        shared_loss = self.variance_loss(student_shared_embed)
        specific_loss = self.variance_loss(student_specific_embed)
        # teacher_shared_loss = self.variance_loss(teacher_shared_embed)
        # teacher_specific_loss = self.variance_loss(teacher_specific_embed)
        global_loss = self.variance_loss(student_global_embed) #+ self.variance_loss(teacher_global_embed)

        # Combine losses

        total_loss = (shared_loss + specific_loss + global_loss)/3

        return total_loss.mean()

    def variance_loss(self,x, eps=1e-4):
        std = torch.std(x, dim=0)  # across batch
        return torch.mean(F.relu(1.0 - std + eps))



