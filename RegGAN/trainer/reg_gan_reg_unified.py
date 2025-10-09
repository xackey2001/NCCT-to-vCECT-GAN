# reg_gan_reg_unified.py
# Perfectly matched to old yamazaki_reg.py (7-layer ResUNet structure)

import torch
import torch.nn as nn
import torch.nn.functional as F
from RegGAN.trainer.reg_gan_layers_unified import ConvND, DownBlockND, UpBlockND, ResnetBlockND


class Reg(nn.Module):
    """ResUNet Registration Network"""

    def __init__(self, dim=3, height=256, width=256, depth=16, in_channels_a=1, in_channels_b=1):
        super(Reg, self).__init__()
        self.dim = dim
        self.height, self.width, self.depth = height, width, depth
        self.in_ch = in_channels_a + in_channels_b

        Conv = nn.Conv2d if dim == 2 else nn.Conv3d

        # --- Encoder (same 7 down blocks as before) ---
        self.down1 = DownBlockND(dim, self.in_ch, 32, 3, 1, 1, use_norm=False, use_resnet=True)
        self.down2 = DownBlockND(dim, 32, 64, 3, 1, 1, use_norm=True,  use_resnet=True)
        self.down3 = DownBlockND(dim, 64, 64, 3, 1, 1, use_norm=True,  use_resnet=True)
        self.down4 = DownBlockND(dim, 64, 64, 3, 1, 1, use_norm=True,  use_resnet=True)
        self.down5 = DownBlockND(dim, 64, 64, 3, 1, 1, use_norm=True,  use_resnet=True)
        self.down6 = DownBlockND(dim, 64, 64, 3, 1, 1, use_norm=True,  use_resnet=True)
        self.down7 = DownBlockND(dim, 64, 64, 3, 1, 1, use_norm=True,  use_resnet=True)

        # --- Bottleneck (ResNet blocks ×3 same as before) ---
        self.c1 = Conv(64, 128, kernel_size=1, stride=1, padding=0)
        self.res = nn.Sequential(
            ResnetBlockND(dim, 128),
            ResnetBlockND(dim, 128),
            ResnetBlockND(dim, 128),
        )
        self.c2 = Conv(128, 64, kernel_size=1, stride=1, padding=0)

        # --- Decoder (7 up blocks mirror structure) ---
        self.up7 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up6 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up5 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up4 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up3 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up2 = UpBlockND(dim, 64, 64, 64, kernel_size=3, stride=1, padding=1, refine=True)
        self.up1 = UpBlockND(dim, 64, 32, 32, kernel_size=3, stride=1, padding=1, refine=True)

        # --- Refinement and output (same as before) ---
        self.refine = nn.Sequential(
            ResnetBlockND(dim, 32),
            Conv(32, 32, kernel_size=1, stride=1, padding=0)
        )

        self.flow = Conv(32, dim, kernel_size=3, stride=1, padding=1)  # 2D→2ch, 3D→3ch

    def forward(self, img_a, img_b):
        """Forward identical to old yamazaki_reg.py."""
        x = torch.cat([img_a, img_b], dim=1)

        # Encoder
        d1, s1 = self.down1(x)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        d4, s4 = self.down4(d3)
        d5, s5 = self.down5(d4)
        d6, s6 = self.down6(d5)
        d7, s7 = self.down7(d6)

        # Bottleneck
        x = self.c1(d7)
        x = self.res(x)
        x = self.c2(x)

        # Decoder with skip connections
        u7 = self.up7(x, s7)   
        u6 = self.up6(u7, s6)
        u5 = self.up5(u6, s5)
        u4 = self.up4(u5, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)

        # Refinement and flow output
        out = self.refine(u1)
        flow = self.flow(out)
        return flow
