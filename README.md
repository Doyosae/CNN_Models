# 리팩토링
- Tensorflow 1.14로 만든 많은 코드들을 Tensorflow 2.1 이상 또는 Pytorch 1.6.0 코드로 변환 중  
- 분류 문제를 위한 모델을 업로드할 것이며, 이후의 구조들은 나중에  
# Direcotry (Pytorch)
```
./model
      __init__.py
      dropout.py
      vgg.py
      resnet.py
      densenet.py
```
# Usage
```
from model.vgg import *
from model.resnet import *
from model.densenet import *
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

mode = VGG16(True, False, 0.0)
summary(model, (3, 32, 32))

>>>
Input size (MB): 0.01
Forward/backward pass size (MB): 1.87
Params size (MB): 10.33
Estimated Total Size (MB): 12.21
----------------------------------------------------------------
olution_Network_Family/summary.py ncher 53853 -- /Users/doyosae/Desktop/Git/Conv 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          36,928
       BatchNorm2d-5           [-1, 64, 32, 32]             128
              ReLU-6           [-1, 64, 32, 32]               0
         MaxPool2d-7           [-1, 64, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]          73,856
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,584
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
        MaxPool2d-14            [-1, 128, 8, 8]               0
... ... ...

test = Slim_VGG11(True, False, 0.0, 0.5)
summary(test, (3, 32, 32))

test = ResNet34(in_channels = 64, drop_ratio = 0.5)
sumaary(test, (3, 32, 32))

>>>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         Dropout2d-2           [-1, 64, 32, 32]               0
       BatchNorm2d-3           [-1, 64, 32, 32]             128
            Conv2d-4           [-1, 64, 32, 32]          36,864
         Dropout2d-5           [-1, 64, 32, 32]               0
       BatchNorm2d-6           [-1, 64, 32, 32]             128
... ... ...
```