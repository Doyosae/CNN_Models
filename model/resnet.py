import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .dropout import *
        

class Basic_Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride, drop_ratio = 0.0):
        super(Basic_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.drop1 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.drop2 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1, stride = stride, bias = False),
                Scaling_Dropout(drop_ratio = drop_ratio),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.bn2(out)

        out = out + shortcut
        out = F.relu(out)

        return out


class Bottle_Neck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride, drop_ratio = 0.0):
        super(Bottle_Neck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.drop1 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.drop2 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.drop3 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn3   = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1, stride = stride, bias = False),
                Scaling_Dropout(drop_ratio = drop_ratio),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):

        shorcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.drop3(out)
        out = self.bn3(out)

        out = out + shorcut
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, in_channels = 32, num_blocks = [2, 2, 2, 2], num_classes = 10, drop_ratio = 0.0):
        super(ResNet, self).__init__()

        self.in_planes  = in_channels
        self.drop_ratio = drop_ratio
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.drop1 = Scaling_Dropout(drop_ratio = drop_ratio)
        self.bn1   = nn.BatchNorm2d(in_channels)

        self.layer1 = self.make_layers(block, in_channels, num_blocks[0], stride = 1)
        self.layer2 = self.make_layers(block, 2*in_channels, num_blocks[1], stride = 2)
        self.layer3 = self.make_layers(block, 4*in_channels, num_blocks[2], stride = 2)
        self.layer4 = self.make_layers(block, 8*in_channels, num_blocks[3], stride = 2)
        self.linear = nn.Linear(8*in_channels*block.expansion, num_classes)

    def make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.drop_ratio))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels = 64, drop_ratio = 0.0):
    return ResNet(Basic_Block, in_channels, [2, 2, 2, 2], num_classes = 10, drop_ratio = drop_ratio)


def ResNet34(in_channels = 64, drop_ratio = 0.0):
    return ResNet(Basic_Block, in_channels, [3, 4, 6, 3], num_classes = 10, drop_ratio = drop_ratio)


def ResNet50(in_channels = 64, drop_ratio = 0.0):
    return ResNet(Bottle_Neck, in_channels, [3, 4, 6, 3], num_classes = 10, drop_ratio = drop_ratio)


def ResNet101(in_channels = 64, drop_ratio = 0.0):
    return ResNet(Bottle_Neck, in_channels, [3, 4, 23, 3], num_classes = 10, drop_ratio = drop_ratio)


def ResNet152(in_channels = 64, drop_ratio = 0.0):
    return ResNet(Bottle_Neck, in_channels, [3, 8, 36, 3], num_classes = 10, drop_ratio = drop_ratio)