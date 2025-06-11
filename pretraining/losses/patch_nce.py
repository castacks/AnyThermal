from .base import *

class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07, patch_radius=1):
        """
        Args:
            temperature (float): Temperature for softmax scaling.
            patch_radius (int): Radius for spatial soft-positive patch sampling.
        """
        super().__init__()
        self.temperature = temperature
        self.patch_radius = patch_radius

    def get_neighbor_indices(self, idx, H, W):
        """Return linear indices of patches in a neighborhood around given patch index."""
        y, x = divmod(idx, W)
        neighbors = []
        for dy in range(-self.patch_radius, self.patch_radius + 1):
            for dx in range(-self.patch_radius, self.patch_radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbors.append(ny * W + nx)
        return neighbors

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: (B, C, H, W) - thermal features
            teacher_feats: (B, C, H, W) - RGB features
        Returns:
            Scalar PatchNCE loss
        """
        B, C, H, W = student_feats.shape
        N = H * W
        loss = 0.0
        total = 0

        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        # Flatten spatial dims
        student_flat = student_feats.permute(0, 2, 3, 1).reshape(B, N, C)
        teacher_flat = teacher_feats.permute(0, 2, 3, 1).reshape(B, N, C)

        for b in range(B):
            for anchor_idx in range(N):
                anchor = student_flat[b, anchor_idx]  # (C,)
                neighbors = self.get_neighbor_indices(anchor_idx, H, W)
                pos_feats = teacher_flat[b, neighbors]  # (P, C)

                neg_mask = torch.ones(N, dtype=torch.bool, device=teacher_feats.device)
                neg_mask[neighbors] = 0
                neg_feats = teacher_flat[b, neg_mask]  # (N-P, C)

                sim_pos = torch.matmul(pos_feats, anchor) / self.temperature  # (P,)
                sim_neg = torch.matmul(neg_feats, anchor) / self.temperature  # (N-P,)

                # Use hardest or average positive
                log_pos = sim_pos.max()
                log_denominator = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=0), dim=0)
                loss += -(log_pos - log_denominator)
                total += 1

        return loss / total
