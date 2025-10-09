# reg_gan_Models_unified.py
# Simplified model definitions for RegGAN (2D/3D unified usage)
# Keeps only: ResidualBlock_2d/3d, Generator_2d/3d, Discriminator_2d/3d

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Residual Blocks ----------------
class ResidualBlock_2d(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_2d, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResidualBlock_3d(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_3d, self).__init__()
 
        conv_block = [nn.ReflectionPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad3d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features)]
 
        self.conv_block = nn.Sequential(*conv_block)
 
    def forward(self, x):
        return x + self.conv_block(x)


# ---------------- Generators ----------------
class Generator_2d(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_2d, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock_2d(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x


class Generator_3d(nn.Module): #model_head + model_body(Residual blocks * 9) + model_tail
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_3d, self).__init__()
 
        # Initial convolution block
        model_head = [nn.ReflectionPad3d((1,1,3,3,3,3)), #ReflectionPad3d((z,z,y,y,x,x,))
                      nn.Conv3d(input_nc, 64, (7,7,3)),
                      nn.InstanceNorm3d(64),
                      nn.ReLU(inplace=True)]
 
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm3d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
 
        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock_3d(in_features)]
 
        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm3d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
 
        # Output layer
        model_tail += [nn.ReflectionPad3d((1,1,3,3,3,3)),
                       nn.Conv3d(64, output_nc, (7,7,3)),
                       nn.Tanh()]
 
        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)
 
    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
 
        return x

# ---------------- Discriminators ----------------
class Discriminator_2d(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_2d, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Discriminator_3d(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_3d, self).__init__()
 
        # A bunch of convolutions one after another
        model = [nn.Conv3d(input_nc, 64, kernel_size=4, stride=(2,2,1), padding=1), 
                 nn.LeakyReLU(0.2, inplace=True)]
 
        model += [nn.Conv3d(64, 128, kernel_size=4, stride=(2,2,1), padding=1), 
                  nn.InstanceNorm3d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
 
        model += [nn.Conv3d(128, 256, kernel_size=4, stride=(2,2,1), padding=1), 
                  nn.InstanceNorm3d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
 
        model += [nn.Conv3d(256, 512, 4, padding=1),
                  nn.InstanceNorm3d(512),
                  nn.LeakyReLU(0.2, inplace=True)]
 
        # FCN classification layer
        model += [nn.Conv3d(512, 1, 4, padding=1)]
 
        self.model = nn.Sequential(*model)
 
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1) 

