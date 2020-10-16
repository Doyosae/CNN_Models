import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Dropout2d (nn.Module):

    def __init__ (self, drop_ratio = 1.0):
        super(Dropout2d, self).__init__()
        if drop_ratio > 1.0:
            raise "drop_ratio smaller than 1.0"
            
        self.drop_ratio = drop_ratio

    def forward (self, inputs):
        
        if self.drop_ratio == 0.0:
            return inputs
        elif self.drop_ratio > 0.0 and self.drop_ratio <= 1.0:
            outputs = nn.Dropout2d(p = self.drop_ratio)(inputs)
            return outputs

if __name__ == "__main__":
    print("TEST")
    inputs  = torch.ones(1, 3, 3, 3)
    outputs = Dropout2d(drop_ratio = 1.1)(inputs)
    print(outputs)