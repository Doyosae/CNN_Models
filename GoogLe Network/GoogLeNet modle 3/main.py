# -*- coding: utf-8 -*-
"""main 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dgdd9A_np1OQ1RvMAjNi0W1SKn5Xhaby
"""

# 필요한 모듈들을 임포트하고 gdrive 경로를 sys.path.insert로 추가한다.
# tensoflow, numpy, matplolib, 나의 모듈들
import os
import sys
from google.colab import drive
drive.mount('/content/gdrive')
sys.path.insert (0, "/content/gdrive/My Drive/Colab Notebooks/GoogLeNet")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import module
from module import module
from module import dataload

# sys.path.insert에 gdrvie 경로가 잘 추가 되었는지 확인
!ls "/content/gdrive/My Drive/Colab Notebooks/GoogLeNet"
print ("\n")
print ("마운트가 잘 되었는지, 기본 폴더 내부에 무엇이 있는지 확인 완료했어요.")

# 세이브 파일을 저장할 경로
save_path = "/content/gdrive/My Drive/Colab Notebooks/GoogLeNet"

# Auxiliary Classifier들은 완전히 Fully Connected Network 으로 연결된다.
# 각각의 결과 배열의 크기는 10이다. 이 값을 실제 라벨과 비교하는 것이 다음 프로세스
def Build_FullyConnectedNetwork (inputs, out_numbers):
    
    outputs = tf.contrib.layers.fully_connected (inputs, out_numbers, activation_fn = None)
    
    if out_numbers > 100:
        outputs = tf.nn.dropout (outputs, rate = RATE)
        
    outputs = tf.contrib.layers.batch_norm (outputs, updates_collections = None, is_training = module.phase)
    
    return outputs

# 필요한 고정상수를 정의한다. 학습을 몇 번 반복할 것인지?
# 배치 사이즈는 얼마나 할 것인지? Boolean 변수 Phase도 있었으나, 다른 모듈로 옮김
Epochs = 100
BatchSize = 128
LearningRate = 0.001

Input_Layer = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Label_Layer = tf.placeholder (tf.float32, shape = [None, 10])
RATE = tf.placeholder (tf.float32)


# 서로 다른 필터 수를 가지는 인셉션 모델을 각각의 클래스로 호출한다.
Inceptionmodule1 = module.Inception (16, 16)
Inceptionmodule2 = module.Inception (80, 80)

Reducntionmodule1 = module.Reduction (32, 64)
Reducntionmodule2 = module.Reduction (160, 320)

Inceptionmodule3 = module.Inception (400, 400)
Redunctionmodule3 = module.Reduction (800, 1600)


# 인셉션 모듈을 이용한 전체 그래프 생성
# 처음에는 일반적인 Stem Convolution 신경망 두 개를 지난다.
outputs = module.Build_Convolution_Network_v2 (Input_Layer)

# 인셉션 모델을 두개 겹친 구조
outputs = Inceptionmodule1.StyleA2 (outputs)

# 리덕션 모듈이 연산하면서 크기를 반으로 줄인다.
outputs = Reducntionmodule1.StyleA (outputs)

# 인셉션 모델을 두개 겹친 구조
outputs = Inceptionmodule2.StyleA2 (outputs)

# 중간 단계 outputs을 빼내 Auxiliary Classifier로 사용할 것
Auxiliary = outputs

# 리덕션 모듈이 연산하면서 크기를 반으로 줄인다.
outputs = Reducntionmodule2.StyleA (outputs)

# 인셉션 모델을 두개 겹친 구조
outputs = Inceptionmodule3.StyleA2 (outputs)

# 최종 outputs을 Classifier
Outputs = outputs


# 각 인셉션 모듈에서 뽑아온 리스트를 AvgPool을 취하고,
# 성분 갯수가 채널 수에만 의존하는 1차원 형태로 변환
AvgPool2 = tf.reduce_mean (Auxiliary, axis = [1, 2])
AvgPool3 = tf.reduce_mean (Outputs, axis = [1, 2])

ShapeList = [np.shape(AvgPool2), np.shape(AvgPool3)]
print (ShapeList)
    

# 인셉션 모델에서 추출한 Auxiliary_outputs을 Fully Connected Layer에 연결
# 그렇게 해서 사이즈 10의 결과들을 합계하여 결과를 계산할 것

# with Auxiliary_outputs2의 신경망 연산
AuxiliaryClassifier = Build_FullyConnectedNetwork (AvgPool2, 100)
AuxiliaryClassifier = tf.nn.relu (AuxiliaryClassifier)
AuxiliaryClassifier = Build_FullyConnectedNetwork (AuxiliaryClassifier,10)


# with Auxiliary_outputs3의 신경망 연산
OutputsClassifier = Build_FullyConnectedNetwork (AvgPool3, 400)
AuxiliaryClassifier = tf.nn.relu (AuxiliaryClassifier)
OutputsClassifier = Build_FullyConnectedNetwork (OutputsClassifier, 200)
AuxiliaryClassifier = tf.nn.relu (AuxiliaryClassifier)
OutputsClassifier = Build_FullyConnectedNetwork (OutputsClassifier, 10)
print (AuxiliaryClassifier, OutputsClassifier)

# (0.5*Auxiliary Classifier) + Outputs를 Summation에 할당
SummarClassifier = (0.5 * AuxiliaryClassifier) + OutputsClassifier


# Training, Summation을 Logits에 할당
SummarClassifier_logits = SummarClassifier
SummarClassifier_softmax = tf.nn.softmax (SummarClassifier)


# Train,   손실 함수 및 훈련 스코프
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = SummarClassifier_logits))
LossTraining1 = tf.train.AdamOptimizer (LearningRate, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)
LossTraining2 = tf.train.AdamOptimizer (LearningRate*0.8, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)
LossTraining3 = tf.train.AdamOptimizer (LearningRate*0.6, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)
LossTraining4 = tf.train.AdamOptimizer (LearningRate*0.5, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)

# Test,   일치성 및 정확도 계산 스코프
# 주의할 점은 정확도 계산은 오로지 마지막 레이어에서 출력된 Outputs Classifier 다룬다.
# 중간 레이어에서 출력된 Auxiliary Classifier는 정확도 계산에 기여하지 않는다.
CorrectPrediction = tf.equal (tf.argmax(Label_Layer, 1), tf.argmax(OutputsClassifier, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

# 손실도와 정확도를 저장할 리스트
LossList = []
AccuList = []

# Graph Sumaary
# 25 Epoch, lr = 0.001
# 50 Epoch, lr = 0.0008
# 75 Epoch, lr = 0.0006
# 100 Epoch, lr = 0.0005
# 세션을 열고 그래프를 실행하여 학습합니다.
with tf.Session () as sess: 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    print ("----------------------------------------------")
    print ("   텐서플로우 세션을 열어서 학습을 시작할게요.")
    print ("----------------------------------------------")
    
    # 전역 상수 Epochs = 100만큼 학습을 시작할게요. 훈련 함수는 다음과 같이 세 가지로 구분했어요.
    # LossTraining1 or LossTraining2 or LossTraining3, 처음에는 lr = 0.01으로 시작해서 50Epc까지는 이 학습률로 훈련해요.
    # 51Epc부터 75Epc까지는 Lr = 0.001*0.75로 학습하고, 75Epc부터 끝까지는 Lr = 0.001*0.5로 학습할꺼에요.
    # 왜냐하면 50Epc 근처에서부터 학습이 급격히 안정화되는걸 관찰했었거든요.
    for Epoch in range (Epochs):
        
        ##############################################################
        if Epoch < 25: 
        
            LossValue = 0.0000

            for i in range (390):

                trainBatch = dataload.Build_NextBatch_Function (BatchSize, dataload.TrainData, dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining1, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                      RATE : 0.3, module.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                                             RATE : 0, module.phase : True}) 

            LossValue = LossValue / 390

            
        ##############################################################
        elif 25 <= Epoch and Epoch < 50:
        
            LossValue = 0.0000

            for i in range (390):

                trainBatch = dataload.Build_NextBatch_Function (BatchSize, dataload.TrainData, dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining2, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                      RATE : 0.3, module.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                                             RATE : 0, module.phase : True}) 

            LossValue = LossValue / 390
            
            
        ##############################################################    
        elif 50 <= Epoch and Epoch < 75:
        
            LossValue = 0.0000

            for i in range (390):

                trainBatch = dataload.Build_NextBatch_Function (BatchSize, dataload.TrainData, dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining3, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                      RATE : 0.3, module.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                                             RATE : 0, module.phase : True}) 
                
            LossValue = LossValue / 390

            
        ##############################################################    
        else:
            
            LossValue = 0.0000      
            
            for i in range (390):
                
                trainBatch = dataload.Build_NextBatch_Function (BatchSize, dataload.TrainData, dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining4, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                      RATE : 0.3, module.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                                             RATE : 0, module.phase : True}) 
                
            LossValue = LossValue / 390
            
            
            
        # 이 라인까지 내려왔다면 한 Epoch 학습이 끝난거에요. 정밀한 추적을 위해 Epoch마다의 Acc를 계산할꺼에요.
        # 테스트 이미지는 총 10,000 장이에요. 1,000장 단위로 배치를 만들고 총 10번 반복하여 테스트해요.
        # 각 배치마다의 Accuracy를 모두 더한 후, (Sum from n=1 to n=10) 다시 10으로 나누어서 Acc의 평균값을 계산해요
        # 트레이닝에는 AuxiliaryClassifier도 사용하지만, 테스트에는 오직 마지막 레이어에서 출력되는 OutputsClassifier만 사용해요
        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산 ----
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주
        
        TestAccuracy = 0.000
        
        for i in range (10):
            testBatch = dataload.Build_NextBatch_Function (1000, dataload.TestData, dataload.SqueezedTestLabel.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {Input_Layer : testBatch[0], 
                                                                           Label_Layer : testBatch[1], 
                                                                           RATE : 0, module.phase : False})
            
        TestAccuracy = TestAccuracy / 10

        print ("- Epoch :   %d회" %(Epoch+1), "  손실도는 :   %f" %LossValue, "   테스트 데이터 정확도:   %.2f" %(100*TestAccuracy), "%")
        LossList.append (LossValue)
        AccuList.append (TestAccuracy)
        
        
        if (Epoch+1) % 25 == 0:
            print ("%d Epoch에서 학습을 한 파라미터 정보를 저장할게요." %(Epoch+1))
            saver.save(sess, save_path + "/save/EpochSave.ckpt")
            print ("파라미터 정보가 정상적으로 저장이 되었어요.")

            # 모델 복원 시, 주의할 점
            # sess.run (tf.global_variables_initializer())
            # saver = tf.train.Saver()
            # saver.restore (sess ~~~)