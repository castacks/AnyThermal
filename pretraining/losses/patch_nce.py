from .base import *

class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07, patch_radius=1, max_patches=256):
        super().__init__()
        self.temperature = temperature
        self.patch_radius = patch_radius
        self.max_patches = max_patches

    def get_pos_mask(self, H, W, device):
        N = H * W
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        coords = coords.view(-1, 2).to(device)
        dists = torch.cdist(coords.float(), coords.float(), p=2)
        return (dists <= self.patch_radius).float()

    def forward(self, student_output, teacher_output):
        student_feats = student_output[0]  # (B, D, H, W)
        teacher_feats = teacher_output[0]  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)

        pos_mask = self.get_pos_mask(H, W, student_feats.device)

        loss = 0.0
        for b in range(B):
            anchor_indices = torch.randperm(H * W)[:min(H * W, self.max_patches)]
            anchors = student_flat[b, anchor_indices]  # (M, D)
            sims = torch.matmul(teacher_flat[b], anchors.T) / self.temperature
            pos_mask_b = pos_mask[:, anchor_indices].to(student_feats.device)
            pos_logits = (sims * pos_mask_b).max(dim=0).values
            denom = torch.logsumexp(sims, dim=0)
            loss += -(pos_logits - denom).mean()
        return loss / B

class PatchNCELossSinglePositive(nn.Module):
    def __init__(self, temperature=0.07, patch_radius=1, exclude_anchor_from_denominator=True):
        super().__init__()
        self.temperature = temperature
        self.patch_radius = patch_radius
        self.exclude_anchor_from_denominator = exclude_anchor_from_denominator

    def get_ignore_mask(self, H, W, device):
        N = H * W
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        coords = coords.view(-1, 2).to(device)
        dists = torch.cdist(coords.float(), coords.float(), p=2)
        return (dists <= self.patch_radius).float()  # (N, N)

    def forward(self, student_output, teacher_output):
        student_feats = student_output[0]  # (B, D, H, W)
        teacher_feats = teacher_output[0]  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        N = H * W
        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)

        ignore_mask = self.get_ignore_mask(H, W, student_feats.device)  # (N, N)

        loss = 0.0
        for b in range(B):
            anchor_indices = torch.arange(N, device=student_feats.device)
            anchors = student_flat[b, anchor_indices]  # (M, D)
            positives = teacher_flat[b, anchor_indices]  # (M, D)

            sims = torch.matmul(teacher_flat[b], anchors.T) / self.temperature  # (N, M)

            ignore_mask_b = ignore_mask[:, anchor_indices]

            if not self.exclude_anchor_from_denominator:
                ignore_mask_b = ignore_mask_b.clone()
                for i, anchor_idx in enumerate(anchor_indices):
                    ignore_mask_b[anchor_idx, i] = 0.0  # explicitly mask the anchor itself  # (N, M)
            ignore_mask_b = ignore_mask_b.to(student_feats.device)

            # mask out neighbors including self (positive + local ignored)
            sims = sims.masked_fill(ignore_mask_b.bool(), float('-inf'))

            pos_logits = torch.sum(positives * anchors, dim=-1) / self.temperature  # (M,)
            denom = torch.logsumexp(sims, dim=0)  # (M,)

            loss += -(pos_logits - denom).mean()

        return loss / B

class GlobalAndPatchCosine(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

    def forward(self, student_output, teacher_output):
        student_feats = student_output[0]  # (B, D, H, W)
        teacher_feats = teacher_output[0]  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        N = H * W
        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)

        student_global = student_output[1]  # (B, D)
        teacher_global = teacher_output[1]

        student_global = F.normalize(student_global, dim=1)  # (B, D)
        teacher_global = F.normalize(teacher_global, dim=1)

        student_flat_final = torch.cat([student_global, student_flat], dim=0)
        teacher_flat_final = torch.cat([teacher_global, teacher_flat], dim=0)
        
        cos_sim = F.cosine_similarity(student_flat_final, teacher_flat_final, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()

class PatchCosine(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

    def forward(self, student_output, teacher_output):
        student_feats = student_output[0]  # (B, D, H, W)
        teacher_feats = teacher_output[0]  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        N = H * W
        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)
        
        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()