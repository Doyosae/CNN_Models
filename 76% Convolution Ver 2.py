import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainDataSet, TrainLabelSet), (TestDataSet, TestLabelSet) = load_data ()

def generate_NextBatch_Function (number, data, labels) :
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)

    # 1. 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
    # 2. 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 만든다.

    
TestData = (TrainDataSet, TrainLabelSet)
ValidData = (TestDataSet, TestLabelSet)

TrainLabel_OneHotEncoding = tf.squeeze (tf.one_hot (TrainLabelSet, 10), axis = 1)
TestLabel_OneHOtEncoding = tf.squeeze (tf.one_hot (TestLabelSet, 10), axis = 1)

print ("훈련라벨세트의 크기     ", np.shape (TrainLabelSet))
print ("검사라벨세트의 크기     ", np.shape (TestLabelSet))
print ("원 핫 인코딩을 한 라벨세트의 크기     ", np.shape (TrainLabel_OneHotEncoding))
print ("원 핫 인코딩을 한 라벨세트의 크기     ", np.shape (TestLabel_OneHOtEncoding))

"""
1. tensorflow.kears.datasets.cifar10에서 load_data 명령어로 훈련 셋과 테스트 셋을 받아온다.
2. X, Y의 Placeholder를 설정해주고, Dropout에서 살릴 노드의 확률을 KeepProbDropout 정의
3. 훈련 라벨과 테스트 라벨은 원 핫 인코딩을 해준 다음에 차원을 줄여준다.
"""

X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])
ImageConvolution = X

KeepProbDropout = tf.placeholder(tf.float32)
EPOCH = 10000
BatchSize = 128
LearningRate = 0.001
FilterSize1 = 32
FilterSize2 = 64
FilterSize3 = 256
FilterSize4 = 512
FilterSize5 = 768


# Session 1. Convolution Nueral Network
ConvolutionFilter1 = tf.Variable (tf.truncated_normal (shape = [3, 3, 3, FilterSize1], stddev = 0.01))
ConvolutionFilter2 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize1, FilterSize2], stddev = 0.01))
ConvolutionFilter3 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize2, FilterSize3], stddev = 0.01))
ConvolutionFilter4 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize3, FilterSize4], stddev = 0.01))
ConvolutionFilter5 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize4, FilterSize5], stddev = 0.01))

SameConvoltuionFilter1 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize4, FilterSize4], stddev = 0.01))
SameConvoltuionFilter2 = tf.Variable (tf.truncated_normal (shape = [3, 3, FilterSize4, FilterSize4], stddev = 0.01))

BiasFilter1 = tf.Variable (tf.truncated_normal (shape = [FilterSize1], stddev = 0.01))
BiasFilter2 = tf.Variable (tf.truncated_normal (shape = [FilterSize2], stddev = 0.01))
BiasFilter3 = tf.Variable (tf.truncated_normal (shape = [FilterSize3], stddev = 0.01))
BiasFilter4 = tf.Variable (tf.truncated_normal (shape = [FilterSize4], stddev = 0.01))
BiasFilter5 = tf.Variable (tf.truncated_normal (shape = [FilterSize5], stddev = 0.01))

# Convolution Step 1 (이미지 -> Convolution Filter + Bias -> 그 값을 ReLU 활성함수를 씌우고, 배치 정규화))
Convolution1 = tf.nn.conv2d (ImageConvolution, ConvolutionFilter1, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter1
ActivatedConvolution1 = tf.nn.relu (Convolution1)
BatchNormConv1 = tf.layers.batch_normalization (ActivatedConvolution1)
print ("Convolution Step 1을 지나고 나서의 배열 사이즈", BatchNormConv1)


# Convolution Step 2 (이미지 - Convolution Filter + Bias - Activated ReLU - Max Pooling - Batch Normaliztion)
Convolution2 = tf.nn.conv2d (BatchNormConv1, ConvolutionFilter2, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter2
ActivatedConvolution2 = tf.nn.relu (Convolution2)
PoolingConvolution2 = tf.nn.max_pool (ActivatedConvolution2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
BatchNormConv2 = tf.layers.batch_normalization (PoolingConvolution2)
print ("Convolution Step 2을 지나고 나서의 배열 사이즈", BatchNormConv2)


# Convolution Step 3 (이미지 - Convolution Filter + Bias - Activated ReLU - Max Pooling - Batch Normaliztion)
Convolution3 = tf.nn.conv2d (BatchNormConv2, ConvolutionFilter3, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter3
ActivatedConvolution3 = tf.nn.relu (Convolution3)   
PoolingConvolution3 = tf.nn.max_pool (ActivatedConvolution3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
BatchNormConv3 = tf.layers.batch_normalization (PoolingConvolution3)
print ("Convolution Step 3을 지나고 나서의 배열 사이즈", BatchNormConv3)


# Convolution Step 4 (이미지 - Convolution Filter + Bias - Activated ReLU - Max Pooling - Batch Normaliztion)
Convolution4 = tf.nn.conv2d (BatchNormConv3, ConvolutionFilter4, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter4
ActivatedConvolution4 = tf.nn.relu (Convolution4)
PoolingConvolution4 = tf.nn.max_pool (ActivatedConvolution4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
BatchNormConv4 = tf.layers.batch_normalization (PoolingConvolution4)
print ("Convolution Step 4을 지나고 나서의 배열 사이즈", BatchNormConv4)


# 사이즈를 유지하면서 Convolution Filter를 씌우는 단계
Convolution5 = tf.nn.conv2d (BatchNormConv4, SameConvoltuionFilter1, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter4
ActivatedConvolution5 = tf.nn.relu (Convolution5)
print ("같은 사이즈를 유지하면서 SameConvFilter1을 지난 배열 사이즈", ActivatedConvolution5)
Convolution6 = tf.nn.conv2d (ActivatedConvolution5, SameConvoltuionFilter2, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter4
ActivatedConvolution6 = tf.nn.relu (Convolution6)
print ("같은 사이즈를 유지하면서 SameConvFilter2을 지난 배열 사이즈", ActivatedConvolution6)

# Convolution Step 5 (이미지 - Convolution Filter + Bias - Activated ReLU - Max Pooling - Batch Normaliztion)
Convolution7 = tf.nn.conv2d (ActivatedConvolution6, ConvolutionFilter5, strides = [1, 1, 1, 1], padding = 'SAME') + BiasFilter5
ActivatedConvolution7 = tf.nn.relu (Convolution7)
PoolingConvolution5 = tf.nn.max_pool (ActivatedConvolution7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
BatchNormConv5 = tf.layers.batch_normalization (PoolingConvolution5)
print ("Convolution Step 5을 지나고 나서의 배열 사이즈", BatchNormConv5)

# Session 2. Convolution Nueral Network의 Output을 Fully Coneected Layer로 연결
ImageFlat = tf.reshape(BatchNormConv5, [-1, 2 * 2 * 768])

WeightFCN1 = tf.Variable (tf.truncated_normal (shape = [2 * 2 * 768, 1536], stddev = 0.01))
BiasFCN1 = tf.Variable (tf.truncated_normal (shape = [1536], stddev = 0.01))
ActivatedFCN1 = tf.nn.relu(tf.matmul(ImageFlat, WeightFCN1) + BiasFCN1)
Dropout = tf.nn.dropout (ActivatedFCN1, KeepProbDropout)

WeightFCN2 = tf.Variable(tf.truncated_normal (shape = [1536, 10], stddev = 0.01))
BiasFCN2 = tf.Variable (tf.truncated_normal (shape = [10], stddev = 0.01))


# 1. Hypothesis는 Cross-Entropy-Softmax 손실도에 들어갈 것
Hypothesis = tf.matmul (Dropout, WeightFCN2) + BiasFCN2

# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
Prediction = tf.nn.softmax (Hypothesis)


# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
Lossfunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (labels = Y, logits = Hypothesis))
LossTrainingStep = tf.train.RMSPropOptimizer(LearningRate).minimize(Lossfunction)

# 정확도를 계산하는 연산을 추가합니다.
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Prediction, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

with tf.Session () as sess :
    sess.run(tf.global_variables_initializer())
    
    print ("학습 시작...")
    print (".........")
    
    # 10000 Step만큼 최적화를 수행, 200 Step마다 학습 데이터에 대해여 정확도와 손실을 출력
    for Epoch in range (4500) :
        
        batch = generate_NextBatch_Function (BatchSize, TrainDataSet, TrainLabel_OneHotEncoding.eval())
       
        if Epoch % 200 == 0 :
            TrainAccuracy = sess.run (Accuracy, feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 1.0})
            PrintLoss = sess.run (Lossfunction, feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 1.0})

            print("Epoch  : %d,   트레이닝 데이터 정확도 : %f,   손실도 : %f" % (Epoch, TrainAccuracy, PrintLoss))
            
        # 20% 확률의 Dropout을 이용해서 학습을 진행
        sess.run (LossTrainingStep, feed_dict = {X : batch[0], Y : batch[1], KeepProbDropout : 0.7})

            
    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
    TestAccuracy = 0.000
    
    for i in range (10) :
        TestBatch = generate_NextBatch_Function (1000, TestDataSet, TestLabel_OneHOtEncoding.eval())
        TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {X : TestBatch[0], Y : TestBatch[1], KeepProbDropout : 1.0})
        
    TestAccuracy = TestAccuracy / 10
    
    # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후, 모두 더한 다음 10으로 나누는 것
    print("테스트 데이터 정확도: %f" % TestAccuracy)
