# Introduction  
Cifar-10 데이터로 다양한 CNN 구조들을 실험합니다.  
잘 알려진 딥러닝 알고리즘들은 이미 만들어져있는 것들을 써도 됩니다. 그럼에도 다시 만들어보는 이유는 두 가지가 있습니다.  
1. 바퀴를 재발명 한다는 마음입니다. 기나긴 코드의 안에는 코딩의 테크닉적 모습도, 딥러닝의 구조적 모습도 공부할 수 있습니다.  
2. 만들어진 것은 만고의 진리가 아닙니다. 얼마든지 변형될 수 있는 가능성을 열어둡니다.  
   직접 작성해보면서 문득 개량의 아이디어가 떠오를지도 모릅니다. 그것을 바로 구현하고 테스트해보십시오!   
   저 또한 코드를 작성하면서 많이 배웠습니다. 앞으로도 이러한 활동들을 계속할 것입니다. 덧붙어 세상에서 가장 쉬운 코드를요.  
   그래서 이제껏 그 어떤 코드 보다도 제가 작성한 코드가 책처럼 술술 읽히는 코드이라 자신합니다.  
  
## 이메일 (E-mail)  
calidris.snipe@gmail.com  
  
## 벤치마크 데이터셋 & 훈련 기본 조건
* Cifar-10  
URL : [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) 
  
* Setting A  
001 <= Epoch < 050, lr = 0.01  
050 <= Epoch < 100, lr = 0.001  
100 <= Epoch < 150, lr = 0.0001  
  
* Setting B  
001 <= Epoch < 050, lr = 0.1  
050 <= Epoch < 100, lr = 0.01  
100 <= Epoch < 150, lr = 0.001  
  
## Repository를 만들면서 참조한 논문들  
* Striving for Simplicity: The All Convolutional Net (ACN)  
  URL : https://arxiv.org/abs/1412.6806?context=cs  
  
* Network In Network (NIN)  
  URL : https://arxiv.org/abs/1312.4400  
  
* Deep Residual Learning for Image Recognition (ResNet)  
  URL : https://arxiv.org/abs/1512.03385  
  
* Going deeper with convolutions (GoogLeNet)  
  URL : https://arxiv.org/abs/1409.4842  

* Rethinking the Inception Architecture for Computer Vision (Inception v2)  
  URL : https://arxiv.org/abs/1512.00567  
  
## Model Summary 
1. ResNet model 1 (150 Epoch로 실험)  
2. GoogLeNet model 2 (150 Epoch로 실험 중)  
  
## Accuracy Summery  
    Accuracy of ResNet (85% ~ 86%)
![ResNet model 1](https://github.com/Doyosae/CNN_Models/blob/master/Residual%20Network/Accuracy/model%201.png)  

    Accuracy of GoogLeNet (80% ~ 81%)
![GoogLeNet model 2](https://github.com/Doyosae/CNN_Models/blob/master/GoogLe%20Network/Accuracy/model%202.png)  
