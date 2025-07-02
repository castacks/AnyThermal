from .base import *

class SSL(nn.Module):
    def __init__(self, use_shared_specific="shared",mode="global_and_patch"):
        super().__init__()
        self.second_view = True  # This loss requires a second view
        self.use_shared_specific = use_shared_specific
        self.mode = mode  # Default mode, can be overridden

    
    def extract_embeddings(self, output):
        if self.mode == 'global':
            shared_embed = output[1][1]     # (B, D)
            specific_embed = output[2][1]
        elif self.mode == 'patch':
            B,D,H,W = output[1][0].shape
            shared_embed = output[1][0].reshape((B, D, -1))
            specific_embed = output[2][0].reshape((B, D, -1))
        elif self.mode == "global_and_patch":
            B,D,H,W = output[1][0].shape
            shared_embed_patch = output[1][0].reshape((B, D, -1)).permute(0, 2, 1)
            specific_embed_patch = output[2][0].reshape((B, D, -1)).permute(0, 2, 1)
            shared_embed = torch.cat((shared_embed_patch, output[1][1].reshape((B,1,D))), dim=1)
            specific_embed = torch.cat((specific_embed_patch, output[2][1].reshape((B,1,D))), dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return shared_embed, specific_embed

    
    def get_embedding(self, student_output, student_dual_output):
        student_dual_shared_embed, student_dual_specific_embed = self.extract_embeddings(student_dual_output)
        student_shared_embed, student_specific_embed = self.extract_embeddings(student_output) 

        if self.use_shared_specific == "shared":
            z1,z2 = student_shared_embed, student_dual_shared_embed
        elif self.use_shared_specific == "specific":
            z1,z2 = student_specific_embed, student_dual_specific_embed
        else:
            raise ValueError(f"Unknown use_shared_specific: {self.use_shared_specific}")
        
        return z1, z2

class InvarianceLoss(SSL):
    def forward(self, student_output, student_dual_output):
        z1, z2 = self.get_embedding(student_output, student_dual_output)
        return F.mse_loss(z1, z2)

class VarianceLoss(SSL):
    def forward(self, student_output, student_dual_output):
        z1, z2 = self.get_embedding(student_output, student_dual_output)
        z1, z2 = z1.reshape(-1, z1.shape[-1]), z2.reshape(-1, z2.shape[-1])
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

class CovarianceLoss(SSL):
    def forward(self, student_output, student_dual_output):
        z1, z2 = self.get_embedding(student_output, student_dual_output)
        z1, z2 = z1.reshape(-1, z1.shape[-1]), z2.reshape(-1, z2.shape[-1])

        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (z1.shape[0] - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (z2.shape[0] - 1)

        return off_diagonal(cov_z1).pow(2).mean() + off_diagonal(cov_z2).pow(2).mean()

class CosineRepulsionLoss(SSL):
    def __init__(self, cosine_repulsion_margin=0.4, use_shared_specific="shared", mode="global_and_patch"):
        super().__init__(use_shared_specific=use_shared_specific, mode=mode)
        self.cosine_repulsion_margin = cosine_repulsion_margin

    def forward(self, student_output, student_dual_output):
        z1, z2 = self.get_embedding(student_output, student_dual_output)
        return self.cosine_repulsion(z1) + self.cosine_repulsion(z2)

    def cosine_repulsion(self, z):
        """
        Penalizes both positive and negative high cosine similarities across samples.
        Supports input shapes (B, D) and (B, N, D).
        Encourages angular diversity across batch (for each patch if N > 1).
        """
        if z.dim() == 3:  # (B, N, D)
            B, N, D = z.shape
            z = z.transpose(0, 1).reshape(N, B, D)  # (N, B, D)
            loss = 0.0
            for n in range(N):
                loss += self._cosine_batchwise(z[n])
            return loss / N
        elif z.dim() == 2:  # (B, D)
            return self._cosine_batchwise(z)
        else:
            raise ValueError(f"Expected shape (B,D) or (B,N,D), got {z.shape}")

    def _cosine_batchwise(self, z):
        z_norm = F.normalize(z, dim=-1)
        sim_matrix = z_norm @ z_norm.T  # (B, B)
        B = z.size(0)
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

        sim_values = sim_matrix[off_diag_mask].abs()
        repulsion_loss = F.relu(sim_values - self.cosine_repulsion_margin).mean()
        return repulsion_loss
