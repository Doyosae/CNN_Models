from model.vgg import *
from model.resnet import *
from model.densenet import *

if __name__ == "__main__":
    '''
    if drop_ratio == 1.0 is No using nn.Dropout2d
    else... Using nn.Dropout2d, model will learn data
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    vgg_config = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

    # vgg config에서 모델을 로드하는 방법
    test = VGG(make_layers(vgg_config["vgg11"], True, True, 0.0))
    summary(test.to(device), (3, 32, 32))
    
    # vgg 모델에서 drop_ratio를 주고 프루닝한 모델을 로드하는 방법
    prune_list = vgg_slim_config(vgg_config["vgg11"], 0.5)
    test = VGG(make_layers(prune_list["prune"], True, True, 0.0))
    summary(test.to(device), (3, 32, 32))


    test = ResNet18(16, 0.0).to(device)
    summary(test, (3, 32, 32))

    test = ResNet18(64, 0.0).to(device)
    summary(test, (3, 32, 32))


    # growth_rate는 in_channels 수에 비례    
    test = DenseNet121(growth_rate = 16, drop_ratio = 0.0).to(device)
    summary(test, (3, 32, 32))

    test = DenseNet121(growth_rate = 64, drop_ratio = 0.0).to(device)
    summary(test, (3, 32, 32))