import torch
import torch.nn.functional as F
import torch.nn as nn

class TransformerND(nn.Module):
    def __init__(self):
        super(TransformerND, self).__init__()

    def forward(self, src, flow):
        """
        Unified 2D / 3D spatial transformer.
        Args:
            src: [B, C, H, W] or [B, C, H, W, D]
            flow: [B, 2, H, W] or [B, 3, H, W, D]
        Returns:
            warped: same shape as src
        """
        dim = flow.shape[1]
        b = flow.shape[0]
        shape = flow.shape[2:]
        size = tuple(shape)

        # --- generate base grid ---
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids).to(torch.float32)  # [dim, H, W(, D)]
        grid = grid.repeat(b, 1, *([1]*dim)).cuda()  # [B, dim, H, W(, D)]
        new_locs = grid + flow

        # --- normalize to [-1,1] ---
        for i in range(dim):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # --- permute and reverse channel order ---
        if dim == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)       # [B, H, W, 2]
            new_locs = new_locs[..., [1, 0]]
        elif dim == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)     # [B, H, W, D, 3]
            new_locs = new_locs[..., [2, 1, 0]]

        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")
        return warped