from model.vgg import *
from model.resnet import *
from model.densenet import *
from torchsummary import summary

test = Slim_VGG11(True, False, 0.0, 0.5)
summary(test, (3, 32, 32))