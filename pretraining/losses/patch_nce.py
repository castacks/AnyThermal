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
    def __init__(self, temperature=0.07,mode='global_and_patch'):
        super().__init__()
        self.mode = mode

    def forward(self, student_output, teacher_output):
        student_feats = student_output[0][0]  # (B, D, H, W)
        teacher_feats = teacher_output[0][0].detach()  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        N = H * W
        student_feats = F.normalize(student_feats, dim=1)
        teacher_feats = F.normalize(teacher_feats, dim=1)

        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(0, 2, 1).view(-1, D)  # (B*N, D)

        student_global = student_output[0][1]  # (B, D)
        teacher_global = teacher_output[0][1].detach()  # (B, D)

        student_global = F.normalize(student_global, dim=1)  # (B, D)
        teacher_global = F.normalize(teacher_global, dim=1)
        
        if self.mode == 'global':
            student_flat_final = student_global
            teacher_flat_final = teacher_global
        elif self.mode == 'patch':
            student_flat_final = student_flat
            teacher_flat_final = teacher_flat
        elif self.mode == 'global_and_patch':
            student_flat_final = torch.cat([student_global, student_flat], dim=0)
            teacher_flat_final = torch.cat([teacher_global, teacher_flat], dim=0)
        
        cos_sim = F.cosine_similarity(student_flat_final, teacher_flat_final, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()

class PatchContrastive(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (B, D, H, W) - thermal features (student)
            teacher_output: (B, D, H, W) - RGB features (teacher)
        Returns:
            Scalar loss
        """
        student_feats = student_output[0][0]  # (B, D, H, W)
        teacher_feats = teacher_output[0][0]  # (B, D, H, W)

        B, D, H, W = student_feats.shape
        N = H * W

        # Normalize features
        student_feats = F.normalize(student_feats, dim=1)  # (B, D, H, W)
        teacher_feats = F.normalize(teacher_feats, dim=1)  # (B, D, H, W)

        # Flatten patches
        student_flat = student_feats.view(B, D, -1).permute(0, 2, 1).unsqueeze(2)  # (B, N,1, D)
        teacher_flat = teacher_feats.view(B, D, -1).permute(2, 0, 1).unsqueeze(0)  # (1,N, B, D)

        # Dot product along D
        logits = []
        for i in range(B):
            logits.append(torch.sum(student_flat[[i]] * teacher_flat, dim=-1))
        logits = torch.cat(logits, dim=0)

        logits = logits / self.temperature

        # For each patch in student (thermal), positive is teacher patch at same (B, N)
        # negatives are patches at same (N) position in other images in batch
        labels = torch.arange(B).to(student_feats.device)  # (B,)
        labels = labels.repeat_interleave(N)  # (B*N,)

        # Reshape logits: (B*N, B)
        logits = logits.view(-1, B)

        # Cross-entropy loss
        # import pdb; pdb.set_trace()
        loss = F.cross_entropy(logits, labels)

        return loss