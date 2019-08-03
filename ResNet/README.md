Residual Network 모델을 구현합니다. 성능 검증으로 활용한 데이터 세트는 Cifar-10 입니다.  
  
# Introduction  
20개의 레이어를 가지는 다양한 ResNet을 시험하면서 특징 별로 어떠한 성능을 보이는지 알아봅시다.  
  
# Model Architecture  
- Ver 1  
Backbone Layer는 2개의 Convolution, Residual Layer는 1개의 Convoluion으로 Skip Connection을 이룹니다. 각각의 스킵 커넥션 모듈은 2번씩 활용됩니다. 이 모듈들은 총 4종류가 있습니다. 모듈들의 차이는 필터수의 차이입니다. 32, 64, 128, 256개의 필터들을 가지고, 스킵 커넥션으로 들어가기 전에 처음 두 번은 일반적인 Convolution 연산을 수행합니다. 10개의 레이블로 구분하기 위하여, 마지막 레이어는 완전 연결 계층으로 묶었습니다.  
- Ver 2  
Ver 1과의 가장 큰 차이는 스킵 커넥션 모듈마다의 Backbone Layer에서 두 번째 Convolution의 필터 수가 배로 늘어나게 바꾼 것입니다. 그리고 마지막 스킵 커넥션에서 필터 갯수의 차원을 10개로 줄이고, Average Pooling을 수행하여 [8, 8, 10] -> [1, 10] 형태로 바꾸었습니다. 하나 더 사소한 변화로는 스킵 커넥션 연산으로 들어가기 전의 두 개의 Normal Convolution 층을 하나로 줄였습니다.  
  
# Accuracy Summary  
- Ver 1  
100 Epoch에서 안정적으로 76 ~ 78% 정확도에 도달  
- Ver 2
50 Epoch에서 안정적으로 82 ~ 84% 정확도에 도달  
