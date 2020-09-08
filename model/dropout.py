import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class Normal_Dropout (nn.Module):

    def __init__ (self, drop_ratio = 1.0):
        super(Normal_Dropout, self).__init__()

        self.drop_ratio = drop_ratio
    
    def forward (self, inputs):

        outputs = nn.Dropout2d(p = self.drop_ratio)(inputs)
        return outputs if self.dropout <1.0 else inputs


class Scaling_Dropout (nn.Module): 
    
    def __init__(self, drop_ratio = 0.3):
        super(Scaling_Dropout, self).__init__()

        self.drop_ratio = drop_ratio

    def forward(self, inputs):

        outputs = (1-self.drop_ratio) * nn.Dropout2d(p = self.drop_ratio)(inputs)
        return outputs if self.drop_ratio < 1.0 else inputs