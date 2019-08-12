# Introduction  
Cifar-10 데이터로 다양한 CNN 구조들을 실험합니다.  
  
## 이메일 (E-mail)  
calidris.snipe@gmail.com  
  
## 벤치마크 데이터셋 & 훈련 기본 조건
* Cifar-10  
URL : [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) 
  
* Type A (Epoch 100)  
Epoch < 50, lr = 0.001  
Epoch < 75, lr = 0.00075  
Epoch < 99, lr = 0.00050  
  
* Type A+ (Epoch 100)  
Epoch < 25, lr = 0.001  
Epoch < 50, lr = 0.0008  
Epoch < 75, lr = 0.0006  
Epoch < 99, lr = 0.0005  
  
* Type B (Epoch 150)  
Epoch < 050, lr = 0.001  
Epoch < 150, lr = 0.0005  
  
* Type C (Epoch 150)  
Epoch < 030, lr = 0.001  
Epoch < 100, lr = 0.00075  
Epoch < 150, lr = 0.00050   
  
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
1. GoogLeNet model 2 (Type A)
1. ResNet model 1 (Type B)  

  
## Accuracy Summery  
    Accuracy of GoogLeNet model 2 Type A (83% ~ 84%)  
![GoogLeNet Type A](https://github.com/Doyosae/CNN_Models/blob/master/GoogLe%20Network/GoogLeNet%20model%202/accuracy/Accuracy%20Graph.png) 
  
    Accuracy of GoogLeNet model 3 Type B (83% ~ 84%)  
![GoogLeNet Type B]
(https://github.com/Doyosae/CNN_Models/blob/master/GoogLe%20Network/GoogLeNet%20modle%203/accuracy/model%203.png)  
  
    Accuracy of ResNet model 1 Type B (85% ~ 86%)  
![ResNet Type B](https://github.com/Doyosae/CNN_Models/blob/master/Residual%20Network/Accuracy/model%201.png)   
