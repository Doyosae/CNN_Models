import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from .dropout import *


vgg_configuration = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def slim_configuration (layer_configuration, prune_ratio):
    prune_configuration = []

    for config in layer_configuration:
        if type(config) is int:
            prune_configuration = prune_configuration + [int(prune_ratio * config)]
        else:
            prune_configuration = prune_configuration + [config]

    return prune_configuration


def make_layers (layer_configuration, batch_norm = True, drop_out = False, drop_ratio = 0.0):
    layers      = []
    in_channels = 3

    for convolution_option in layer_configuration:
        if convolution_option == 'M':
            layers = layers + [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else: 
            conv2d = nn.Conv2d(in_channels, convolution_option, kernel_size = 3, padding = 1)

            if batch_norm == True and drop_out == True:
                layers = layers + [conv2d, Dropout2d(drop_ratio), nn.BatchNorm2d(convolution_option), nn.ReLU(inplace = True)]
            elif batch_norm == True and drop_out == False:
                layers = layers + [conv2d, nn.BatchNorm2d(convolution_option), nn.ReLU(inplace = True)]
            else:
                layers = layers + [conv2d, nn.ReLU(inplace = True)]
            in_channels = convolution_option

    return nn.Sequential(*layers), layer_configuration[-2]


class VGG (nn.Module):

    def __init__(self, layer_list):
        super(VGG, self).__init__()

        self.convolution, self.last_channel = layer_list
        self.classifier                     = nn.Sequential(nn.Linear(self.last_channel, 512),
                                                            nn.ReLU(True),
                                                            nn.Linear(512, 512),
                                                            nn.ReLU(True),
                                                            nn.Linear(512, 10))

    def forward(self, inputs):

        outputs = self.convolution(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)

        return outputs



def VGG11 (batch_norm = True, drop_out = False, drop_ratio = 0.0):
    return VGG(make_layers(vgg_configuration["vgg11"], batch_norm, drop_out, drop_ratio))

def VGG13(batch_norm = True, drop_out = False, drop_ratio = 0.0):
    return VGG(make_layers(vgg_configuration["vgg13"], batch_norm, drop_out, drop_ratio))

def VGG16(batch_norm = True, drop_out = False, drop_ratio = 0.0):
    return VGG(make_layers(vgg_configuration["vgg16"], batch_norm, drop_out, drop_ratio))

def VGG19(batch_norm = True, drop_out = False, drop_ratio = 0.0):
    return VGG(make_layers(vgg_configuration["vgg19"], batch_norm, drop_out, drop_ratio))

def Slim_VGG11 (batch_norm = True, drop_out = False, drop_ratio = 0.0, prune_ratio = 0.0):
    prune_config = slim_configuration(vgg_configuration["vgg11"], prune_ratio)
    return VGG(make_layers(prune_config, batch_norm, drop_out, drop_ratio))

def Slim_VGG13 (batch_norm = True, drop_out = False, drop_ratio = 0.0, prune_ratio = 0.0):
    prune_config = slim_configuration(vgg_configuration["vgg13"], prune_ratio)
    return VGG(make_layers(prune_config, batch_norm, drop_out, drop_ratio))

def Slim_VGG16 (batch_norm = True, drop_out = False, drop_ratio = 0.0, prune_ratio = 0.0):
    prune_config = slim_configuration(vgg_configuration["vgg16"], prune_ratio)
    return VGG(make_layers(prune_config, batch_norm, drop_out, drop_ratio))

def Slim_VGG19 (batch_norm = True, drop_out = False, drop_ratio = 0.0, prune_ratio = 0.0):
    prune_config = slim_configuration(vgg_configuration["vgg19"], prune_ratio)
    return VGG(make_layers(prune_config, batch_norm, drop_out, drop_ratio))