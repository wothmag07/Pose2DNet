import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_by_2 = out_channels // 2

        # Define layers
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels_by_2)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_by_2, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels_by_2, self.out_channels_by_2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.out_channels_by_2, self.out_channels, kernel_size=1, stride=1, padding=0)

        if self.in_channels != self.out_channels:
            self.conv_skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        # Main path
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        # Skip connection
        if self.in_channels != self.out_channels:
            skip = self.conv_skip(x)
        else:
            skip = x

        # Add main path and skip connection
        out += skip
        out = self.relu(out)
        return out

class MyUpsample(nn.Module):
    def __init__(self):
        super(MyUpsample, self).__init__()

    def forward(self, x):
        # Unsqueeze along dimensions to insert additional dimensions
        x = x.unsqueeze(3).unsqueeze(5)
        # Use interpolate for upsampling
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        # Squeeze to remove inserted dimensions
        x = x.squeeze(3).squeeze(4)
        return x

class Hourglass(nn.Module):
    def __init__(self, nChannels=256, numReductions=4, nModules=2, poolKernel=(2,2), poolStride=(2,2), upSampleKernel=2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        
        # Define skip connection
        self.skip = nn.Sequential(*[ResidualBottleneckBlock(nChannels, nChannels) for _ in range(nModules)])

        # First pooling and residual blocks
        self.mp = nn.MaxPool2d(poolKernel, poolStride)
        self.afterpool = nn.Sequential(*[ResidualBottleneckBlock(nChannels, nChannels) for _ in range(nModules)])

        if numReductions > 1:
            self.hg = Hourglass(nChannels, numReductions-1, nModules, poolKernel, poolStride)
        else:
            self.num1res = nn.Sequential(*[ResidualBottleneckBlock(nChannels, nChannels) for _ in range(nModules)])

        # Final residual blocks
        self.lowres = nn.Sequential(*[ResidualBottleneckBlock(nChannels, nChannels) for _ in range(nModules)])

        # Upsampling layer
        self.up = nn.Upsample(scale_factor=upSampleKernel, mode='bilinear', align_corners=True)

    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        
        if self.numReductions > 1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        
        out2 = self.lowres(out2)
        out2 = self.up(out2)
        return out2 + out1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class StackedHourGlass(nn.Module):
    def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
        super(StackedHourGlass, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.nJoints = nJoints

        self.start = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.res1 = ResidualBottleneckBlock(64, 128)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = ResidualBottleneckBlock(128, 128)
        self.res3 = ResidualBottleneckBlock(128, self.nChannels)

        self.hourglass = nn.ModuleList([Hourglass(self.nChannels, self.numReductions, self.nModules) for _ in range(self.nStack)])
        self.Residual = nn.ModuleList([nn.Sequential(*[ResidualBottleneckBlock(self.nChannels, self.nChannels) for _ in range(self.nModules)]) for _ in range(self.nStack)])
        self.lin1 = nn.ModuleList([ConvBlock(self.nChannels, self.nChannels) for _ in range(self.nStack)])
        self.chantojoints = nn.ModuleList([nn.Conv2d(self.nChannels, self.nJoints, 1) for _ in range(self.nStack)])
        self.lin2 = nn.ModuleList([nn.Conv2d(self.nChannels, self.nChannels, 1) for _ in range(self.nStack)])
        self.jointstochan = nn.ModuleList([nn.Conv2d(self.nJoints, self.nChannels, 1) for _ in range(self.nStack)])

    def forward(self, x):
        # print(x.shape)
        x = self.start(x)
        # print(x.shape)
        x = self.res1(x)
        # print(x.shape)
        x = self.mp(x)
        # print(x.shape)
        x = self.res2(x)
        # print(x.shape)
        x = self.res3(x)
        # print(x.shape)

        out = []

        for i in range(self.nStack):
            x1 = self.hourglass[i](x)
            # print(x1.shape)
            x1 = self.Residual[i](x1)
            # print(x1.shape)
            x1 = self.lin1[i](x1)
            # print(x1.shape)
            out.append(self.chantojoints[i](x1))
            # print(x1.shape)
            x1 = self.lin2[i](x1)
            # print(x1.shape)
            x = x + x1 + self.jointstochan[i](out[i])
            # print(x.shape)

        return out
