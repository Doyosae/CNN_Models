# Introduction  
Residual Network 모델을 구현합니다. 성능 검증으로 활용한 데이터 세트는 Cifar-10 입니다.  
20개의 레이어를 가지면서 디테일이 조금씩 다른 ResNet을 시험하며, 사양 별로 어떠한 성능을 보이는지 알아봅시다.  
# Model Architecture  
- Model 1  
Init  
Backbone Layer는 2개의 Convolution,  
Residual Layer는 1개의 Convolution으로 Skip Connection을 이룹니다.  
각각의 스킵 커넥션 모듈은 2번씩 활용됩니다. 이 모듈들은 총 4종류가 있습니다.  
모듈들의 차이는 필터수의 차이입니다. 32, 64, 128, 256개의 필터들을 가지고,  
스킵 커넥션으로 들어가기 전에 처음 두 번은 일반적인 Convolution 연산을 수행합니다.  
10개의 레이블로 구분하기 위하여, 마지막 레이어는 완전 연결 계층으로 묶었습니다.  
20190804  
RMSprop에서 AdamOptimizer로 변경  
Epoch가 지나면 lr을 0.001에서 0.0005로 낮추었음.  
-------------------------------------------------------------------------------
- Model 2  
series 1과의 가장 큰 차이는 스킵 커넥션 모듈마다의 Backbone Layer에서  
두 번째 Convolution의 필터 수가 배로 늘어나게 바꾼 것입니다.  
마지막 스킵 커넥션에서는 필터 갯수의 차원을 10개로 줄이고,  
Avg Pooling을 수행하여 (8, 8, 10) -> (1, 10) 로 바꾸었습니다.  
하나 더 사소한 변화로는 스킵 커넥션 연산으로 들어가기 전의  
두 개의 Normal Convolution 층을 하나로 줄였습니다.  
20190804  
RMSprop에서 AdamOptimizer로 변경  
if Epoch <= 30: lr = 0.001, elif 30 < Epoch < 90: lr = 0.0005, else: lr = 0.0002  
-------------------------------------------------------------------------------
- Model 3  
series 1, 2와 가장 큰 차이는 Optimizer 함수를 RMSprop에서 AdamOptimizer로 바꾼 것입니다.  
Epoch에 따른 학습률도 달리 적용하였습니다. 사전 실험을 통하여 얻은 결과로  
30 Epoch까지는 Lr = 0.001, 30 Epoch 이후부터는 Lr = 0.0001로 하였습니다.  
구조적 차이도 커졌습니다. 기존의 ResNet에서 보였던 스킵 커넥션의 Residual Network를  
더 추가하여 Backbone Network에 교차하며 연결하였습니다. DenseNet하고의 모양이 비슷하지만,  
한 Residual Network가 다른 Residual Network를 포함하지는 않습니다.  
series 3에서는 Backbone Net에 대하여 체인 형태로 Residual Net이 동작합니다.  
-------------------------------------------------------------------------------
# Accuracy Summary  
- Model 1  
  137 Epoch 부터 안정적인 85 ~ 86%의 Accuracy  
  ![model1](https://github.com/Doyosae/CNN_Models/blob/master/Residual%20Network/Accuracy/model%201.png)
- Model 2  
  81 Epoch 부터 안정적인 87% ~ 89% Accuracy  
- Model 3  
  30 Epoch 부터 안정적인 85 ~ 86% Accuracy  
  ![model3](https://github.com/Doyosae/CNN_Models/blob/master/Residual%20Network/Accuracy/model%203.png)
