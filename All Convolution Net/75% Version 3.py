import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainDataSet, TrainLabelSet), (TestDataSet, TestLabelSet) = load_data ()
    
TestData = (TrainDataSet, TrainLabelSet)
ValidData = (TestDataSet, TestLabelSet)

TrainLabel_OneHotEncoding = tf.squeeze (tf.one_hot (TrainLabelSet, 10), axis = 1)
TestLabel_OneHOtEncoding = tf.squeeze (tf.one_hot (TestLabelSet, 10), axis = 1)

print ("Train Label Set의 크기           ", np.shape (TrainLabelSet))
print ("Valid Label Set의 크기           ", np.shape (TestLabelSet))
print ("원 핫 인코딩을 한 라벨세트의 크기     ", np.shape (TrainLabel_OneHotEncoding))
print ("원 핫 인코딩을 한 라벨세트의 크기     ", np.shape (TestLabel_OneHOtEncoding))



# 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
# 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 생성
def generate_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


# Convolution 연산을 실행하는 Build 함수
def build_Convolution_Filter (size, channel, number, strides, data) : 
   
    Weight = tf.Variable (tf.truncated_normal (shape = [size, size, channel, number], stddev = 0.01))
    Bias = tf.Variable (tf.truncated_normal (shape = [number], stddev = 0.01))
    Convolution = tf.nn.relu (tf.nn.conv2d (data, Weight, strides = [1, strides, strides, 1], padding = 'SAME') + Bias)
    
    return Convolution

    
# Batch Nomalization을 적용한 Convolution Build 함수    
def build_Convolution_Filter_BatchNorm (size, channel, number, data) :
    
    Weight = tf.Variable (tf.truncated_normal (shape = [size, size, channel, number], stddev = 0.01))
    Convolution = tf.nn.relu (tf.nn.conv2d (data, Weight, strides = [1, 1, 1, 1], padding = 'SAME'))
    ActivatedResult = tf.layers.batch_normalization (Convolution, center = True, scale = True, training = True)
    
    return ActivatedResult
    

# Max Pooling Build 함수    
def build_MaxPooling_Function (size, step, data) :  
        
    Pool = tf.nn.max_pool (data, ksize = [1, size, size, 1], strides = [1, step, step, 1], padding = 'SAME')
        
    return Pool


# 해당 레이어의 사이즈를 출력하고 싶을때 호출 할 것, 위치한 레이어의 이름과 레이어 넘버만 넣으면 자동으로 출력    
def print_ArraySize (number, data) :
    print ("%d번째 레이어의 배열 사이즈     " %number, data)

X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])
rate = tf.placeholder(tf.float32)

EPOCH = 10000
BatchSize = 256
LearningRate = 0.001



# Session 1. Convolution Neural Network
# 함수를 호출할 때 필요한 정보는 순서대로... Size, Channel, Number, Strides, Data
Conv1 = build_Convolution_Filter (3, 3, 96, 1, X)
print_ArraySize(1, Conv1)

Conv2 = build_MaxPooling_Function (3, 2, build_Convolution_Filter (3, 96, 96, 1, Conv1))
print_ArraySize(2, Conv2)

Conv3 = build_Convolution_Filter (3, 96, 192, 1, Conv2)
print_ArraySize(3, Conv3)

Conv4 = build_MaxPooling_Function (3, 2, build_Convolution_Filter (3, 192, 192, 1, Conv3))
print_ArraySize(4, Conv4)

Conv5 = build_Convolution_Filter (3, 192, 384, 1, Conv4)
print_ArraySize(5, Conv5)

Conv6 = build_MaxPooling_Function (3, 2, build_Convolution_Filter (3, 384, 384, 1, Conv5))
print_ArraySize(6, Conv6)

Conv7 = build_Convolution_Filter (3, 384, 768, 1, Conv6)
print_ArraySize(7, Conv7)

Conv8 = build_MaxPooling_Function (3, 2, build_Convolution_Filter (3, 768, 768, 1, Conv7))
print_ArraySize(8, Conv8)



# Session 2. Convolution Nueral Network의 Output을 Fully Coneected Layer로 연결
FlatedImage = tf.reshape (Conv8, [-1, 2*2*768])

Weight1 = tf.Variable (tf.truncated_normal (shape = [2*2*768, 1024], stddev = 0.01))
Weight2 = tf.Variable (tf.truncated_normal (shape = [1024, 10], stddev = 0.01))
Bias1 = tf.Variable (tf.truncated_normal (shape = [1024], stddev = 0.01))
Bias2 = tf.Variable (tf.truncated_normal (shape = [10], stddev = 0.01))

Activated1 = tf.nn.relu (tf.matmul (FlatedImage, Weight1) + Bias1)
Dropout = tf.nn.dropout (Activated1, rate)

# 1. Hypothesis는 Cross-Entropy-Softmax 손실도에 들어갈 것
# Cross Entropy를 손실함수(Loss function)으로 정의하고, AdamOptimizer 이용해서 손실도를 minimize
Hypothesis = tf.matmul (Dropout, Weight2) + Bias2
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Y, logits = Hypothesis))
LossTrainingStep = tf.train.AdamOptimizer(LearningRate).minimize(Lossfunction)


# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
# 정확도를 계산하는 연산을 추가합니다.
Prediction = tf.nn.softmax (Hypothesis)
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Prediction, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

with tf.Session () as sess :
    
    sess.run(tf.global_variables_initializer())
    
    print ("학습 시작...")
    print (".........")
    
    # 5000 Step만큼 최적화를 수행, 200 Step마다 학습 데이터에 대해여 정확도와 손실을 출력
    for Epoch in range (5000) :
        
        batch = generate_NextBatch_Function (BatchSize, TrainDataSet, TrainLabel_OneHotEncoding.eval())
       
        if Epoch % 200 == 0 :
            
            TrainAccuracy = sess.run (Accuracy, feed_dict = {X : batch[0], Y : batch[1], rate : 1.0})
            PrintLoss = sess.run (Lossfunction, feed_dict = {X : batch[0], Y : batch[1], rate : 1.0})

            print("Epoch  : %d,   트레이닝 데이터 정확도 : %f,   손실도 : %f" % (Epoch, TrainAccuracy, PrintLoss))
            
        # 30% 확률의 Dropout을 이용해서 학습을 진행
        sess.run (LossTrainingStep, feed_dict = {X : batch[0], Y : batch[1], rate : 0.7})

            
    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
    TestAccuracy = 0.000
    
    for i in range (10) :
        TestBatch = generate_NextBatch_Function (1000, TestDataSet, TestLabel_OneHOtEncoding.eval())
        TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {X : TestBatch[0], Y : TestBatch[1], rate : 1.0})
        
    TestAccuracy = TestAccuracy / 10
    
    # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후, 모두 더한 다음 10으로 나누는 것
    print("테스트 데이터 정확도: %f" % TestAccuracy)
