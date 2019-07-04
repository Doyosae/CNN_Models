"""
설계 방식은
1. 먼저 두 층의 Depthwise Convolution Layer를 겹치고, 마지막에 한 번 Max Pooling
2. 그 다음 보통의 Convolution Layer를 쌓고, 마지막에 한 번 Max Pooling
3. 그리고 마지막에는 Fully Connected Layer로 연결
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(X_training, Y_training), (X_test, Y_test) = load_data ()


# 1. 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
# 2. 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 만든다.
def Next_Batch_Function (number, data, labels) :
    
    DataRange = np.arange(0 , len(data))
    np.random.shuffle(DataRange)
    DataRange = DataRange[ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


TestData = (X_training, Y_training)
ValidData = (X_test, Y_test)
BatchSize = 256

Y_training_OneHotLabeling = tf.squeeze (tf.one_hot (Y_training, 10), axis = 1)
Y_test_OneHotLabeling = tf.squeeze (tf.one_hot (Y_test, 10), axis = 1)

print (np.shape (Y_training))
print (np.shape (Y_test))

"""
1. tensorflow.kears.datasets.cifar10에서 load_data 명령어로 훈련 셋과 테스트 셋을 받아온다.
2. X, Y의 Placeholder를 설정해주고, Dropout에서 살릴 노드의 확률을 KeepProbDropout 정의
3. 훈련 라벨과 테스트 라벨은 원 핫 인코딩을 해준 다음에 차원을 줄여준다.
"""


X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])
KeepProbDropout = tf.placeholder(tf.float32)
ImageConvolution = X


# Session 1. two step Depthwise Convolution + two step Typical Convolution Layer
WeightConv1 = tf.Variable (tf.truncated_normal (shape = [3, 3, 3, 16], stddev = 0.01))
BiasConv1 = tf.Variable (tf.truncated_normal (shape = [3 * 16], stddev = 0.01))
ActivationConv1 = tf.nn.relu (tf.nn.depthwise_conv2d (ImageConvolution, WeightConv1, strides = [1, 2, 2, 1], padding = 'SAME') + BiasConv1)
print ("1-1. 첫 번째 Depthwise Convolution을 통과한 배열 사이즈   ", ActivationConv1)

WeightConv2 = tf.Variable (tf.truncated_normal (shape = [3, 3, 48, 16], stddev = 0.01))
BiasConv2 = tf.Variable (tf.truncated_normal (shape = [48 * 16], stddev = 0.01))
ActivationConv2 = tf.nn.relu (tf.nn.depthwise_conv2d (ActivationConv1, WeightConv2, strides = [1, 2, 2, 1], padding = 'SAME') + BiasConv2)
print ("1-2. 두 번째 Depthwise Convolution을 통과한 배열 사이즈   ", ActivationConv2)

Pooling1 = tf.nn.max_pool(ActivationConv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("   두 depthwise convolution을 지나고 Max Pooling한 배열 사이즈   ", Pooling1)


WeightConv3 = tf.Variable (tf.truncated_normal (shape = [3, 3, 768, 768], stddev = 0.01))
BiasConv3 = tf.Variable (tf.truncated_normal (shape = [768], stddev = 0.01))
ActivationConv3 = tf.nn.relu (tf.nn.conv2d (Pooling1, WeightConv3, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv3)
print ("2-1. 첫 번째 Convolution Filter를 통과한 배열 사이즈   ", ActivationConv3)

WeightConv4 = tf.Variable (tf.truncated_normal (shape = [3, 3, 768, 3072], stddev = 0.01))
BiasConv4 = tf.Variable (tf.truncated_normal (shape = [3072], stddev = 0.01))
ActivationConv4 = tf.nn.relu (tf.nn.conv2d (ActivationConv3, WeightConv4, strides = [1, 1, 1, 1], padding = 'SAME') + BiasConv4)
print ("2-2. 두 번째 Convolution Filter를 통과한 배열 사이즈   ", ActivationConv3)

Dropout1 = tf.nn.dropout (ActivationConv4, KeepProbDropout)

Pooling2 = tf.nn.max_pool(ActivationConv4, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print ("      두 convolution을 지나고 Max Pooling한 배열 사이즈   ", Pooling2)



# Session 2. Convolution Nueral Network의 Output을 Fully Coneected Layer Input으로 연결
ImageFlat = tf.reshape(Pooling2, [-1, 2 * 2 * 3072])

WeightFCN1 = tf.Variable (tf.truncated_normal (shape = [2 * 2 * 3072, 1152], stddev = 0.01))
BiasFCN1 = tf.Variable (tf.truncated_normal (shape = [1152], stddev = 0.01))
ActivationFCN1 = tf.nn.relu(tf.matmul(ImageFlat, WeightFCN1) + BiasFCN1)

Dropout2 = tf.nn.dropout (ActivationFCN1, KeepProbDropout)

WeightFCN2 = tf.Variable(tf.truncated_normal (shape = [1152, 10], stddev = 0.01))
BiasFCN2 = tf.Variable (tf.truncated_normal (shape = [10], stddev = 0.01))

Hypothesis = tf.matmul (Dropout2, WeightFCN2) + BiasFCN2
# 1. Hypothesis는 Cross-Entropy-Softmax 손실도에 들어갈 것
Prediction = tf.nn.softmax (Hypothesis)
# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것


# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화
Lossfunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = Hypothesis))
LossTrainingStep = tf.train.RMSPropOptimizer(0.0002).minimize(Lossfunction)

# 정확도를 계산하는 연산을 추가합
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Prediction, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))


# Run Tensorflow All Grpah... JDI
with tf.Session () as sess :
    sess.run(tf.global_variables_initializer())
    
    print ("... 학습 시작......")
    print (".........")
    
    # 10000 Step만큼 최적화를 수행
    for Epoch in range (5000):
        batch = Next_Batch_Function (BatchSize, X_training, Y_training_OneHotLabeling.eval())
            
        
        # 100 Step 마다 학습 데이터에 대하여 정확도와 손실을 출력
        if Epoch % 100 == 0 :
            TrainAccuracy = Accuracy.eval (feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 1.0})
            PrintLoss = Lossfunction.eval (feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 1.0})

            print("Epoch  : %d,   트레이닝 데이터 정확도 : %f,   손실도 : %f" % (Epoch, TrainAccuracy, PrintLoss))
            
        # 20% 확률의 Dropout을 이용해서 학습을 진행
        sess.run (LossTrainingStep, feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 0.8})

            
    # 학습이 끝나면 테스트 데이터에 대한 정확도를 출력
    TestAccuracy = 0.000
    
    for i in range (10) :
        TestBatch = Next_Batch_Function (1000, X_test, Y_test_OneHotLabeling.eval())
        TestAccuracy = TestAccuracy + Accuracy.eval (feed_dict = {X : TestBatch[0], Y : TestBatch[1], KeepProbDropout : 1.0})
        
    TestAccuracy = TestAccuracy / 10
    # 방식은...
    # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후
    # 모두 더한 다음 총 배치 수인 10으로 나누는 것
    
    print("테스트 데이터 정확도: %f" % TestAccuracy)
