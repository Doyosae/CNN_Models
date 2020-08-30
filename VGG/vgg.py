import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class VGG11 (nn.Module):

    def __init__ (self, in_channels = 3, out_channels = 64, inputs_size = (32, 32), drop_prob = 0.5):
        super(VGG11, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.height, _    = inputs_size
        self.drop_prob    = drop_prob
        

        self.module1 = nn.Sequential(nn.Conv2d(in_channels = self.in_channels, out_channels = 1*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(1*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 1*self.out_channels, out_channels = 2*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(2*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module2 = nn.Sequential(nn.Conv2d(in_channels = 2*self.out_channels, out_channels = 4*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(4*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 4*self.out_channels, out_channels = 4*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(4*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module3 = nn.Sequential(nn.Conv2d(in_channels = 4*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module4 = nn.Sequential(nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.classifier = nn.Sequential(nn.Linear(8 * self.out_channels * int(self.height / 16) * int(self.height / 16), 8 * self.out_channels * int(self.height / 16)),
                                    nn.BatchNorm1d(8 * self.out_channels * int(self.height / 16)),
                                    nn.ReLU(),
                                    nn.Linear(8 * self.out_channels * int(self.height / 16), 8 * self.out_channels * int(self.height / 16)),
                                    nn.BatchNorm1d(8 * self.out_channels * int(self.height / 16)),
                                    nn.ReLU(),
                                    nn.Linear(8 * self.out_channels * int(self.height / 16), 10))
        

    def forward(self, inputs):

        outputs = self.module1(inputs)
        outputs = self.module2(outputs)
        outputs = self.module3(outputs)
        outputs = self.module4(outputs)

        'FCN Network'
        outputs = outputs.view(-1, (8 * self.out_channels) * int(self.height / 16) * int(self.height / 16))
        outputs = self.classifier(outputs)

        return outputs


class VGG16 (nn.Module):

    def __init__ (self, in_channels = 3, out_channels = 64, inputs_size = (32, 32), drop_prob = 0.5):
        super(VGG16, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.height, _    = inputs_size
        self.drop_prob    = drop_prob
        

        self.module1 = nn.Sequential(
                                    nn.Conv2d(in_channels = self.in_channels, out_channels = 1*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(1*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 1*self.out_channels, out_channels = 1*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(1*self.out_channels),
                                    nn.ReLU())

        self.module2 = nn.Sequential(
                                    nn.Conv2d(in_channels = 1*self.out_channels, out_channels = 2*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(2*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 2*self.out_channels, out_channels = 2*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(2*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module3 = nn.Sequential(
                                    nn.Conv2d(in_channels = 2*self.out_channels, out_channels = 4*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(4*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 4*self.out_channels, out_channels = 4*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(4*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 4*self.out_channels, out_channels = 4*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(4*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module4 = nn.Sequential(
                                    nn.Conv2d(in_channels = 4*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.module5 = nn.Sequential(
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 8*self.out_channels, out_channels = 8*self.out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.Dropout2d(self.drop_prob),
                                    nn.BatchNorm2d(8*self.out_channels),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2, 2), (2, 2)))

        self.classifier = nn.Sequential(
                                    nn.Linear(8 * self.out_channels * int(self.height / 16) * int(self.height / 16), 8 * self.out_channels * int(self.height / 16)),
                                    nn.BatchNorm1d(8 * self.out_channels * int(self.height / 16)),
                                    nn.ReLU(),
                                    nn.Linear(8 * self.out_channels * int(self.height / 16), 8 * self.out_channels * int(self.height / 16)),
                                    nn.BatchNorm1d(8 * self.out_channels * int(self.height / 16)),
                                    nn.ReLU(),
                                    nn.Linear(8 * self.out_channels * int(self.height / 16), 10))
        

    def forward(self, inputs):

        outputs = self.module1(inputs)
        outputs = self.module2(outputs)
        outputs = self.module3(outputs)
        outputs = self.module4(outputs)
        outputs = self.module5(outputs)

        'FCN Network'
        outputs = outputs.view(-1, (8 * self.out_channels) * int(self.height / 16) * int(self.height / 16))
        outputs = self.classifier(outputs)

        return outputs



if __name__ == "__main__":
      test = VGG11()
      summary(test, (3, 32, 32))
