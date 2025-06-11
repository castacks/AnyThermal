import torch
import torch.nn as nn
import torch.nn.functional as F
class NetVLADHead(nn.Module):
    def __init__(self, num_clusters=64, dim=768, output_dim=None, mode='local_only'):
        """
        Args:
            num_clusters: Number of clusters for NetVLAD
            dim: Dimension of each token (usually 768 for DINOv2)
            output_dim: If set, adds a projection layer to reduce final feature dimension
            mode: 'local_only', 'append_global', 'fuse_attention', or 'all_tokens'
        """
        super().__init__()
        assert mode in ['local_only', 'append_global', 'fuse_attention', 'all_tokens']
        self.mode = mode
        self.num_clusters = num_clusters
        self.dim = dim
        self.output_dim = output_dim

        self.clusters = nn.Parameter(torch.rand(num_clusters, dim))
        self.cluster_weights = nn.Linear(dim, num_clusters, bias=False)

        if self.mode == 'fuse_attention':
            self.attention_fuser = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
            self.vlad_proj = nn.Linear(num_clusters * dim, dim)

        input_dim = {
            'local_only': num_clusters * dim,
            'all_tokens': num_clusters * dim,
            'append_global': num_clusters * dim + dim,
            'fuse_attention': dim
        }[mode]

        if output_dim is not None:
            self.project = nn.Linear(input_dim, output_dim)
        else:
            self.project = nn.Identity()

    def _netvlad_aggregate(self, x):
        # x: (B, N, D)
        soft_assign = self.cluster_weights(x)  # (B, N, K)
        soft_assign = F.softmax(soft_assign, dim=-1)

        residual = x.unsqueeze(2) - self.clusters.unsqueeze(0).unsqueeze(0)  # (B, N, K, D)
        residual *= soft_assign.unsqueeze(-1)  # (B, N, K, D)

        vlad = residual.sum(dim=1)  # (B, K, D)
        vlad = F.normalize(vlad, p=2, dim=-1)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # (B, K*D)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalization
        return vlad

    def forward(self, token_dict):
        """
        token_dict: dict from dinov2.forward_features(x)
        Expected keys: 'x_norm_clstoken', 'x_norm_patchtokens'
        Returns: final embedding of shape (B, output_dim) if set, else varies by mode
        """
        batch = token_dict[0].shape[0]  # Get batch size
        patch_tokens = token_dict[0].reshape((batch,-1,self.dim))     # (B, N, D)
        cls_token = token_dict[1].unsqueeze(1)  # (B, 1, D)

        assert len(patch_tokens.shape) == 3, "Expected patch_tokens to be of shape (B, N, D)"
        assert len(cls_token.shape) == 3 and cls_token.shape[1] == 1, "Expected cls_token to be of shape (B, 1, D)"

        if self.mode == 'local_only':
            vlad = self._netvlad_aggregate(patch_tokens)
            return self.project(vlad)

        elif self.mode == 'append_global':
            vlad = self._netvlad_aggregate(patch_tokens)     # (B, K*D)
            out = torch.cat([vlad, cls_token.squeeze(1)], dim=-1)  # (B, K*D + D)
            out = F.normalize(out, p=2, dim=1)
            return self.project(out)

        elif self.mode == 'fuse_attention':
            vlad = self._netvlad_aggregate(patch_tokens)     # (B, K*D)
            vlad_proj = self.vlad_proj(vlad).unsqueeze(1)    # (B, 1, D)
            fused, _ = self.attention_fuser(vlad_proj, cls_token, cls_token)  # (B, 1, D)
            fused = F.normalize(fused.squeeze(1), p=2, dim=1)  # (B, D)
            return self.project(fused)

        elif self.mode == 'all_tokens':
            all_tokens = torch.cat([cls_token, patch_tokens], dim=1)  # (B, N+1, D)
            vlad = self._netvlad_aggregate(all_tokens)
            return self.project(vlad)




