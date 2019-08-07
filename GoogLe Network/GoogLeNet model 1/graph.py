# 아래 주석 처리된 부분은 저의 환경에서 실행하기 위한 코드입니다.
# colab에서 gdrive를 마운트하고, 모듈들이 있는 파일 경로를 sys.path.insert합니다.

# from google.colab import drive
# drive.mount('/content/gdrive')

# import os
# import sys
# sys.path.insert (0, "/content/gdrive/My Drive/Colab Notebooks/GoogLeNet/")

# # sys.path.insert에 경로가 잘 추가 되어서 들여다볼 수 있는지 되었는지 확인
# !ls /content/gdrive/My\ Drive/Colab\ Notebooks/GoogLeNet


# 필요한 모듈들을 임포트
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# module 폴더를 import하고 내부에서 model과 dataload를 import
import module
from module import model
from module import dataload

# Auxiliary Output은 FullyConnectedNetwork로 연결됩니다. 이를 위한 함수를 만들었습니다.
def Build_FullyConnectedNetwork (inputs, out_numbers):
    
    outputs = tf.contrib.layers.fully_connected (inputs, out_numbers, activation_fn = None)
    outputs = tf.contrib.layers.batch_norm (outputs, updates_collections = None, is_training = model.phase)
    outputs = tf.nn.relu (outputs)
    
    return outputs


# 필요한 고정 상수들을 설정합니다. 그래프를 그립니다.
# 그리고 손실 함수와 최적화 함수도 정의
Epochs = 100
BatchSize = 128
LearningRate = 0.001

Input_Layer = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Label_Layer = tf.placeholder (tf.float32, shape = [None, 10])

# 서로 다른 필터 수를 가지는 인셉션 모델들을 각각의 클래스로 호출합니다.
InceptionModel1 = model.Inception (32, 64)
InceptionModel2 = model.Inception (64, 128)
InceptionModel3 = model.Inception (128, 256)
InceptionModel4 = model.Inception (256, 512)

# Step 1. Convolution Network를 먼저 연산합니다.
outputs = model.Build_Convolution_Network (Input_Layer)

# Step 2. model.Inception 클래스에 스타일1, 스타일2 형태의 인셉션 모듈을 그려놓았습니다.
# 이 그래프에서는 스타일 2 형태의 인셉션만 호출해서 사용합니다.
# 인셉션 모듈이 4개를 거치지만 Reduction A나 여타 다른 연산은 사용하지 않았습니다.
# 후의 model 2 구현에서 사용하겠습니다.
InceptionOutputs1 = InceptionModel1.Style2 (outputs)
OUTPUTS1 = InceptionOutputs1

InceptionOutputs2 = InceptionModel2.Style2 (InceptionOutputs1)
OUTPUTS2 = InceptionOutputs2

InceptionOutputs3 = InceptionModel3.Style2 (InceptionOutputs2)
OUTPUTS3 = InceptionOutputs3

InceptionOutputs4 = InceptionModel4.Style2 (InceptionOutputs3)
InceptionOutputs4 = tf.contrib.layers.max_pool2d (InceptionOutputs4, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
OUTPUTS4 = InceptionOutputs4
print ("마지막 Inception 출력은 맥스 풀링을 연산", np.shape(InceptionOutputs4))

# 각 인셉션 모듈에서 뽑아온 리스트를 AvgPool을 취하여 성분 갯수가 채널 수에만 의존하는 1차원 형태로 변환한다.
AvgPool1 = tf.reduce_mean (OUTPUTS1, axis = [1, 2])
AvgPool2 = tf.reduce_mean (OUTPUTS2, axis = [1, 2])
AvgPool3 = tf.reduce_mean (OUTPUTS3, axis = [1, 2])
AvgPool4 = tf.reduce_mean (OUTPUTS4, axis = [1, 2])
print (np.shape(AvgPool1))
print (np.shape(AvgPool2))
print (np.shape(AvgPool3))
print (np.shape(AvgPool4))


# 각각의 인셉션 모델에서 추출한 결과 중 일부를 Fully Connected Layer에 연결한다.
# 그렇게 해서 사이즈 10의 결과들을 합계하여 결과를 계산할 것
AuxiliaryClass1 = Build_FullyConnectedNetwork (AvgPool2, 64)
SummationOutputs1 = tf.contrib.layers.fully_connected (AuxiliaryClass1, 10, activation_fn = None)

AuxiliaryClass2 = Build_FullyConnectedNetwork (AvgPool3, 192)
AuxiliaryClass2 = Build_FullyConnectedNetwork (AuxiliaryClass2, 64)
SummationOutputs2 = tf.contrib.layers.fully_connected (AuxiliaryClass2, 10, activation_fn = None)

AuxiliaryClass3 = Build_FullyConnectedNetwork (AvgPool4, 768)
AuxiliaryClass3 = Build_FullyConnectedNetwork (AuxiliaryClass3, 384)
SummationOutputs3 = tf.contrib.layers.fully_connected (AuxiliaryClass3, 10, activation_fn = None)

Summation = 0.3 * SummationOutputs1 + 0.3 * SummationOutputs2 + SummationOutputs3
print (np.shape(Summation))


# Summation Logits으로 부여하고, Cross_Entropy를 손실값과 손실함수를 정의
Logits = Summation
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = Logits))
LossTraining1 = tf.train.AdamOptimizer(LearningRate, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)
LossTraining2 = tf.train.AdamOptimizer(LearningRate*0.5, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)
LossTraining3 = tf.train.AdamOptimizer(LearningRate*0.2, beta1 = 0.99, beta2 = 0.999).minimize(Lossfunction)


# Summation Softmax를 씌운 그 Predict값이 Argmax하였을때 얼마나 정답 레이블과 일치하는지를 보는것
Predict = tf.nn.softmax (Summation)
CorrectPrediction = tf.equal (tf.argmax(Label_Layer, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))


# 손실도와 정확도를 저장할 리스트
LossList = []
AccuList = []

# 세션을 열고 그래프를 실행하여 학습합니다.
with tf.Session () as sess:    
    sess.run(tf.global_variables_initializer())
    
    print ("----------------------------------------")
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("----------------------------------------")
    
    # Epochs Step만큼 기계 학습을 시작
    for Epoch in range (Epochs):
        print ("- Epoch :   %d회" %(Epoch+1))
        LossValue = 0.000
        
        if Epoch < 50:
            for i in range (390):
                trainBatch = dataload.Build_NextBatch_Function (BatchSize, 
                                                                dataload.TrainData, 
                                                                dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining1, feed_dict = {Input_Layer : trainBatch[0], 
                                                      Label_Layer : trainBatch[1],
                                                      model.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                             Label_Layer : trainBatch[1],
                                                                             model.phase : True}) 
        elif Epoch >= 50 and Epoch < 750:
            
            for i in range (390):

                trainBatch = dataload.Build_NextBatch_Function (BatchSize, 
                                                                dataload.TrainData, 
                                                                dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining2, feed_dict = {Input_Layer : trainBatch[0], 
                                                      Label_Layer : trainBatch[1], 
                                                      model.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                             Label_Layer : trainBatch[1], 
                                                                             model.phase : True})
        else:
            for i in range (390):

                trainBatch = dataload.Build_NextBatch_Function (BatchSize, 
                                                                dataload.TrainData, 
                                                                dataload.SqueezedTrainLabel.eval())
                sess.run (LossTraining3, feed_dict = {Input_Layer : trainBatch[0], 
                                                      Label_Layer : trainBatch[1], 
                                                      model.phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                             Label_Layer : trainBatch[1], 
                                                                             model.phase : True})
        LossValue = LossValue / 390
        LossList.append (LossValue)
        print ("   손실도는 :            %f" %LossValue)

        
        # 학습하지 않은 테스트 이미지로 학습한 모델에 대해서 그 정확도를 계산한다.
        # 테스트 이미지는 총 10,000 장이다. 1,000장 단위로 배치를 만들고 총 10번 반복하여 테스트한다.
        # 각 배치마다의 Accuracy를 모두 더한 후, (Summation n=10) 다시 10으로 나누어서 Acc의 평균값을 구한다.
        TestAccuracy = 0.000
        
        for i in range (10):
            testBatch = dataload.Build_NextBatch_Function (1000, 
                                                           dataload.TestData, 
                                                           dataload.SqueezedTestLabel.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {Input_Layer : testBatch[0], 
                                                                           Label_Layer : testBatch[1], 
                                                                           model.phase : False})

        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산한다.
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주한다.
        TestAccuracy = TestAccuracy / 10
        AccuList.append (TestAccuracy)
        print("   테스트 데이터 정확도:   %.4f" %TestAccuracy)

plt.plot (LossList, color = "darkred")

plt.plot (AccuList, color = "darkred")
