import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from .dropout import *


vgg_config = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def vgg_slim_config (cfg_list, drop_ratio):

    prune_cfg = {'prune' : []}

    for data in cfg_list:
        if type(data) is int:
            prune_cfg["prune"] = prune_cfg["prune"] + [int(drop_ratio * data)]
        else:
            prune_cfg["prune"] = prune_cfg["prune"] + [data]

    return prune_cfg


class VGG (nn.Module):

    def __init__(self, layer_list):
        super(VGG, self).__init__()

        self.convolution, self.last_channel = layer_list
        self.classifier                     = nn.Sequential(nn.Linear(self.last_channel, 512),
                                                            nn.ReLU(True),
                                                            nn.Linear(512, 512),
                                                            nn.ReLU(True),
                                                            nn.Linear(512, 10))
        'Initialize weights'
        for layers in self.modules():
            if isinstance(layers, nn.Conv2d):
                parameters = layers.kernel_size[0] * layers.kernel_size[1] * layers.out_channels
                layers.weight.data.normal_(0, math.sqrt(2. / parameters))
                layers.bias.data.zero_()


    def forward(self, inputs):

        outputs = self.convolution(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)

        return outputs


def make_layers (model_option_config, batch_norm = False, drop_out = False, drop_ratio = 0.0):

    layers      = []
    in_channels = 3

    for convolution_option in model_option_config:

        if convolution_option == 'M':
            layers = layers + [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else: 
            conv2d = nn.Conv2d(in_channels, convolution_option, kernel_size = 3, padding = 1)

            if batch_norm == True and drop_out == False:
                layers = layers + [conv2d, nn.BatchNorm2d(convolution_option), nn.ReLU(inplace = True)]
            elif batch_norm == True and drop_out == True:
                layers = layers + [conv2d, Scaling_Dropout(drop_ratio), nn.BatchNorm2d(convolution_option), nn.ReLU(inplace = True)]
            else:
                layers = layers + [conv2d, nn.ReLU(inplace = True)]

            in_channels = convolution_option
    return nn.Sequential(*layers), model_option_config[-2]