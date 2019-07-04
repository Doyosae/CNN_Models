# -*- coding: utf-8 -*-
"""
Cifar-10 Model.ipynb
Automatically generated by Colaboratory.
Original file is located at https://colab.research.google.com/drive/1NYl03VWEpHoS-6e8qpwfUmURA0ISqP0d
"""

"""
Cifar 모델 설계에 참고한 논문 (Reference)
Title : Striving For Slmplicity, The All Concolutional Net (Accepted as a workshop contribution at ICLR 2015)
Author : Jost Tobias Springenberg⇤, Alexey Dosovitskiy⇤, Thomas Brox, Martin Riedmiller
Department of Computer Science University of Freiburg

URL : https://arxiv.org/abs/1412.6806?context=cs
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(X_training, Y_training), (X_test, Y_test) = load_data ()

TestData = (X_training, Y_training) ; ValidData = (X_test, Y_test)

BatchSize = 256

X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])
KeepProbDropout = tf.placeholder(tf.float32)

Y_training_OneHotLabeling = tf.squeeze (tf.one_hot (Y_training, 10), axis = 1) ; print (np.shape(Y_training))
# 그 크기를 출력하면 좀 남다르다. 원 핫 인코딩 처리된 10개의 차원이 추가되었다.
Y_test_OneHotLabeling = tf.squeeze (tf.one_hot (Y_test, 10), axis = 1) ; print (np.shape (Y_test))

ImageConvolution = X

def Next_Batch_Function (number, data, labels) :
    DataRange = np.arange(0 , len(data))
    np.random.shuffle(DataRange)
    DataRange = DataRange[ : number]
    
    # 1. 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
    # 2. 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 만든다.
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)

"""
1. tensorflow.kears.datasets.cifar10에서 load_data 명령어로 훈련 셋과 테스트 셋을 받아온다.
2. X, Y의 Placeholder를 설정해주고, Dropout에서 살릴 노드의 확률을 KeepProbDropout 정의
3. 훈련 라벨과 테스트 라벨은 원 핫 인코딩을 해준 다음에 차원을 줄여준다.
"""

"""
1. 합성곱 신경망 필터의 채널 갯수를 3으로 맞춘다.
2. MNIST 이미지에서는 채널 한 개에서 모델을 설계한 것과 차이가 있음
"""

WeightConv1 = tf.Variable (tf.truncated_normal (shape = [3, 3, 3, 96], stddev = 0.01))
BiasConv1 = tf.Variable (tf.truncated_normal (shape = [96], stddev = 0.01))
ActivationConv1 = tf.nn.relu (tf.nn.conv2d (ImageConvolution, WeightConv1, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv1)

WeightConv2 = tf.Variable (tf.truncated_normal (shape = [3, 3, 96, 96], stddev = 0.01))
BiasConv2 = tf.Variable (tf.truncated_normal (shape = [96], stddev = 0.01))
ActivationConv2 = tf.nn.relu (tf.nn.conv2d (ActivationConv1, WeightConv2, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv2)
PoolingLayer2 = tf.nn.max_pool(ActivationConv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding='SAME')
print ("두 번째 컨볼루션 레이어에서의 풀링 사이즈", PoolingLayer2)

WeightConv3 = tf.Variable (tf.truncated_normal (shape = [3, 3, 96, 196], stddev = 0.01))
BiasConv3 = tf.Variable (tf.truncated_normal (shape = [196], stddev = 0.01))
ActivationConv3 = tf.nn.relu (tf.nn.conv2d (PoolingLayer2, WeightConv3, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv3)

WeightConv4 = tf.Variable (tf.truncated_normal (shape = [3, 3, 196, 196], stddev = 0.01))
BiasConv4 = tf.Variable (tf.truncated_normal (shape = [196], stddev = 0.01))
ActivationConv4 = tf.nn.relu (tf.nn.conv2d (ActivationConv3, WeightConv4, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv4)
PoolingLayer4 = tf.nn.max_pool (ActivationConv4, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("네 번째 컨볼루션 레이어에서의 풀링 사이즈", PoolingLayer4)


# Session 2. Convolution Nueral Network가 끝난 후, Fully Coneected Layer로 연결
ImageFlat = tf.reshape(PoolingLayer4, [-1, 8 * 8 * 196])

WeightFCN1 = tf.Variable (tf.truncated_normal (shape = [ 8 * 8 * 196, 2048], stddev = 0.01))
BiasFCN1 = tf.Variable (tf.truncated_normal (shape = [2048], stddev = 0.01))
ActivationFCN1 = tf.nn.relu(tf.matmul(ImageFlat, WeightFCN1) + BiasFCN1)

WeightFCN2 = tf.Variable(tf.truncated_normal (shape = [2048, 1024], stddev = 0.01))
BiasFCN2 = tf.Variable (tf.truncated_normal (shape = [1024], stddev = 0.01))
ActivationFCN2 = tf.nn.relu (tf.matmul(ActivationFCN1, WeightFCN2) + BiasFCN2)
ActivationFCN2Dropout = tf.nn.dropout (ActivationFCN2, KeepProbDropout)

WeightFCN4 = tf.Variable(tf.truncated_normal (shape = [1024, 10], stddev = 0.01))
BiasFCN4 = tf.Variable (tf.truncated_normal (shape = [10], stddev = 0.01))

Hypothesis = tf.matmul (ActivationFCN2Dropout, WeightFCN4) + BiasFCN4
# 1. Hypothesis는 Cross-Entropy-Softmax 손실도에 들어갈 것
Prediction = tf.nn.softmax (Hypothesis)
# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것


# Session 3. Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화
Lossfunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = Hypothesis))
LossTrainingStep = tf.train.RMSPropOptimizer(0.001).minimize(Lossfunction)

# 아래 구문은 정확도를 예측
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Prediction, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))


with tf.Session () as sess :
    sess.run(tf.global_variables_initializer())
    
    print ("학습 시작...")
    print (".........")
    
    # Sessrion 4. range의 범위만큼 Epoch를 수행
    for Epoch in range (5000) :
        
        batch = Next_Batch_Function (BatchSize, X_training, Y_training_OneHotLabeling.eval())
        sess.run (LossTrainingStep, feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 0.8})

        if Epoch % 100 == 0 :
            
            TrainAccuracy = Accuracy.eval (feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout: 1.0})
            PrintLoss = Lossfunction.eval (feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout: 1.0})

            print("Epoch  : %d,   트레이닝 데이터 정확도 : %f,   손실도 : %f" % (Epoch, TrainAccuracy, PrintLoss))

            
    # Last Session. 학습이 모두 끝나고 테스트 데이터를 넣어서 그 정확도를 
    TestAccuracy = 0.000
    
    for i in range (10) :
        TestBatch = Next_Batch_Function(1000, X_test, Y_test_OneHotLabeling.eval())
        TestAccuracy = TestAccuracy + Accuracy.eval (feed_dict = {X : TestBatch[0], Y : TestBatch[1], KeepProbDropout: 1.0})
        
    TestAccuracy = TestAccuracy / 10
    # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후, 모두 더한 다음 10으로 나누는 것
    print("테스트 데이터 정확도: %f" % TestAccuracy)
