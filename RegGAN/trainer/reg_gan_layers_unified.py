# reg_gan_layers_unified.py
# Unified 2D/3D building blocks (Conv/Down/Up/Attention/Resnet/smoothing_loss)
# - Selects 2D or 3D ops based on `dim` argument (2 or 3)
# - Mirrors APIs from original reg_gan_layers.py (2D) and reg_gan_layers2.py (3D)

from functools import partial
import torch
import torch.nn.functional as F
from torch import nn


# ------------------------- common config -------------------------
alpha = 0.02
resnet_n_blocks = 1
align_corners = False
UPSAMPLE_MODE = {2: 'bilinear', 3: 'trilinear'}


# ------------------------- init & activations -------------------------
def get_init_function(activation, init_function, **kwargs):
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None


# ------------------------- unified conv -------------------------
class ConvND(nn.Module):
    """
    dim-aware Conv + (optional) Norm + Act + (optional) ResBlock
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, bias=True,
                 activation='leaky_relu', init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super().__init__()
        assert dim in (2, 3), "dim must be 2 or 3"
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        Norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d

        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformerND(dim, out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = Norm(out_channels, affine=False, track_running_stats=False) if use_norm else None
        
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        # init
        init_ = get_init_function(activation, init_func, **kwargs)
        init_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x


# ------------------------- encoder / decoder blocks -------------------------
class DownBlockND(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, activation='leaky_relu', init_func='kaiming',
                 use_norm=False, use_resnet=False, skip=True, refine=False,
                 pool=True, pool_size=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.conv_0 = ConvND(dim, in_channels, out_channels, kernel_size, stride, padding,
                             bias=bias, activation=activation, init_func=init_func,
                             use_norm=use_norm, use_resnet=use_resnet)
        self.conv_1 = ConvND(dim, out_channels, out_channels, kernel_size, stride, padding,
                             bias=bias, activation=activation, init_func=init_func,
                             use_norm=use_norm, use_resnet=use_resnet) if refine else None
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size) if dim == 2 else nn.MaxPool3d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            # mimic old yamazaki: for very small z-dim, avoid downsampling z
            if self.dim == 3 and x.size(4) <= 4:
                x = nn.MaxPool3d(kernel_size=(2, 2, 1))(x)
            else:
                x = self.pool(x)
        return (x, skip) if self.skip else x


class UpBlockND(nn.Module):
    def __init__(self, dim, nc_down_stream, nc_skip_stream, nc_out,
                 kernel_size, stride, padding, bias=True,
                 activation='leaky_relu', init_func='kaiming',
                 use_norm=False, refine=False, use_resnet=False,
                 use_add=False, use_attention=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_attention = use_attention
        self.use_add = use_add

        # Up7などskipがない層もあるため、skipチャネル数が0なら単入力Conv
        in_ch = nc_down_stream + nc_skip_stream
        nc_inner = kwargs.get('nc_inner', nc_out)

        self.conv_0 = ConvND(dim, in_ch, nc_inner,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias, activation=activation, init_func=init_func,
                             use_norm=use_norm, use_resnet=use_resnet)
        self.conv_1 = ConvND(dim, nc_inner, nc_inner,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias, activation=activation, init_func=init_func,
                             use_norm=use_norm, use_resnet=use_resnet) if refine else None

        if use_attention:
            self.attention_gate = AttentionGateND(dim, nc_down_stream, nc_skip_stream,
                                                  nc_inner, use_norm=True, init_func=init_func)

        self.up_conv = ConvND(dim, nc_inner, nc_out,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias, activation=activation, init_func=init_func,
                              use_norm=use_norm, use_resnet=False)

        if use_add:
            add_out_ch = 2 if dim == 2 else 3
            self.output = ConvND(dim, nc_out, add_out_ch, kernel_size=1, stride=1, padding=0,
                                 bias=bias, activation=None, init_func='zeros', use_norm=False, use_resnet=False)

    def forward(self, down_stream, skip_stream=None):
        ds = down_stream.size()
        if skip_stream is not None:
            ss = skip_stream.size()
            if self.use_attention:
                skip_stream = self.attention_gate(down_stream, skip_stream)
            if self.dim == 2 and (ds[2] != ss[2] or ds[3] != ss[3]):
                down_stream = F.interpolate(down_stream, (ss[2], ss[3]),
                                            mode=UPSAMPLE_MODE[2], align_corners=align_corners)
            elif self.dim == 3 and (ds[2] != ss[2] or ds[3] != ss[3] or ds[4] != ss[4]):
                down_stream = F.interpolate(down_stream, (ss[2], ss[3], ss[4]),
                                            mode=UPSAMPLE_MODE[3], align_corners=align_corners)
            x = torch.cat([down_stream, skip_stream], 1)
        else:
            x = down_stream

        x = self.conv_0(x)
        if self.conv_1 is not None:
            x = self.conv_1(x)
        if self.use_add:
            return self.output(x) + down_stream
        else:
            return self.up_conv(x)


class AttentionGateND(nn.Module):
    def __init__(self, dim, nc_g, nc_x, nc_inner, use_norm=False, init_func='kaiming', mask_channel_wise=False):
        super().__init__()
        self.dim = dim
        self.conv_g = ConvND(dim, nc_g, nc_inner, 1, 1, 0, bias=True, activation=None, init_func=init_func,
                             use_norm=use_norm, use_resnet=False)
        self.conv_x = ConvND(dim, nc_x, nc_inner, 1, 1, 0, bias=False, activation=None, init_func=init_func,
                             use_norm=use_norm, use_resnet=False)
        self.residual = nn.ReLU(inplace=True)
        self.mask_channel_wise = mask_channel_wise
        # attention_map out channels: full channels if channel-wise, else 1
        out_ch = nc_x if mask_channel_wise else 1
        self.attention_map = ConvND(dim, nc_inner, out_ch, 1, 1, 0, bias=True, activation='sigmoid',
                                    init_func=init_func, use_norm=use_norm, use_resnet=False)

    def forward(self, g, x):
        x_size = x.size()
        g_size = g.size()
        x_resized = x
        g_c = self.conv_g(g)
        x_c = self.conv_x(x_resized)

        if self.dim == 2:
            if x_c.size(2) != g_size[2] or x_c.size(3) != g_size[3]:
                x_c = F.interpolate(x_c, (g_size[2], g_size[3]), mode=UPSAMPLE_MODE[2], align_corners=align_corners)
        else:
            if x_c.size(2) != g_size[2] or x_c.size(3) != g_size[3] or x_c.size(4) != g_size[4]:
                x_c = F.interpolate(x_c, (g_size[2], g_size[3], g_size[4]), mode=UPSAMPLE_MODE[3], align_corners=align_corners)

        combined = self.residual(g_c + x_c)
        alpha = self.attention_map(combined)

        if not self.mask_channel_wise:
            alpha = alpha.repeat(1, x_size[1], *([1] * (self.dim)))

        # final resize to x size (safety)
        a_size = alpha.size()
        if self.dim == 2 and (a_size[2] != x_size[2] or a_size[3] != x_size[3]):
            alpha = F.interpolate(alpha, (x_size[2], x_size[3]), mode=UPSAMPLE_MODE[2], align_corners=align_corners)
        if self.dim == 3 and (a_size[2] != x_size[2] or a_size[3] != x_size[3] or a_size[4] != x_size[4]):
            alpha = F.interpolate(alpha, (x_size[2], x_size[3], x_size[4]), mode=UPSAMPLE_MODE[3], align_corners=align_corners)

        return alpha * x


# ------------------------- ResNet -------------------------
class ResnetTransformerND(nn.Module):
    def __init__(self, dim, c, n_blocks, init_func):
        super().__init__()
        self.dim = dim
        model = []
        for _ in range(n_blocks):
            model += [ResnetBlockND(dim, c, padding_type='reflect', norm_layer=nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d,
                                    use_dropout=False, use_bias=True)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)
        def init_weights(m):
            Conv = nn.Conv2d if dim == 2 else nn.Conv3d
            BN = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
            if isinstance(m, Conv):
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, BN):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class ResnetBlockND(nn.Module):
    def __init__(self, dim, c, padding_type='reflect', norm_layer=None,
                 use_dropout=False, use_bias=True):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        PadRef = nn.ReflectionPad2d if dim == 2 else nn.ReflectionPad3d

        conv_block = []
        if padding_type == 'reflect':
            conv_block += [PadRef(1)]
            p = 0
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] not implemented')

        norm_layer = norm_layer or (nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d)
        conv_block += [
            Conv(c, c, kernel_size=3 if dim == 2 else (3,3,3), padding=p, bias=use_bias),
            norm_layer(c, affine=False, track_running_stats=False),
            nn.ReLU(inplace=False)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            conv_block += [PadRef(1)]
            p = 0
        conv_block += [
            Conv(c, c, kernel_size=3 if dim == 2 else (3,3,3), padding=p, bias=use_bias),
            norm_layer(c, affine=False, track_running_stats=False)
        ]
        self.block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.block(x)

# ------------------------- smoothing loss -------------------------
def smoothing_loss_nd(flow):
    """
    Dim-aware smoothing regularizer.
    2D: mean(dx^2 + dy^2)
    3D: mean(dx^2 + dy^2 + dz^2)/3
    flow shape:
      2D: [B, 2, H, W]
      3D: [B, 3, H, W, D]
    """
    if flow.dim() == 4:  # 2D
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        return (dx*dx).mean() + (dy*dy).mean()
    elif flow.dim() == 5:  # 3D
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        return ((dx*dx).mean() + (dy*dy).mean() + (dz*dz).mean()) / 3.0
    else:
        raise ValueError("flow must be 4D (2D case) or 5D (3D case)")

# Backward-compatible aliases
smooothing_loss = smoothing_loss_nd  # keep the original misspelling for drop-in import
