import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainDataSet, TrainLabelSet), (TestDataSet, TestLabelSet) = load_data ()
    
TestData = (TrainDataSet, TrainLabelSet)
ValidData = (TestDataSet, TestLabelSet)

TrainLabel_OneHotEncoding = tf.squeeze (tf.one_hot (TrainLabelSet, 10), axis = 1)
TestLabel_OneHOtEncoding = tf.squeeze (tf.one_hot (TestLabelSet, 10), axis = 1)

print ("훈련 이미지의 크기           ", np.shape (TrainDataSet))
print ("훈련 라벨링의 크기           ", np.shape (TrainLabel_OneHotEncoding))
print ("검사 이미지의 크기           ", np.shape (TestDataSet))
print ("검사 라벨링의 크기           ", np.shape (TestLabel_OneHOtEncoding))


# Session 1. 필요한 함수를 정의한다. 배치 셔플 함수, 신경망 함수
# 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
# 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 생성
def Build_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


def Build_NetworkNetwork_Function (inputs) :
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = Size1, kernel_size = 5, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size1, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size1, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    print (outputs)

    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size2, kernel_size = 3,
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size2, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size2, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    print (outputs)

    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size3, kernel_size = 3, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = Size3, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    outputs = tf.contrib.layers.batch_norm(outputs, is_training = is_training)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 10, kernel_size = 1, 
                                        stride = 1, padding = "SAME", activation_fn = None)
    print (outputs)


    outputs = tf.reduce_mean(outputs, axis = [1, 2])
    print ("After Avg_Pooling Size is... ... ", outputs)

    outputs = tf.reshape (outputs, [-1, 10])

    Logits = outputs
    Predict = tf.nn.softmax (outputs)

    return Logits, Predict


# Session 2. 필요한 상수, 변수, 함수 호출, 훈련, 손실도, 정확도 계산하는 구문을 입력
Size1 = 256
Size2 = 128
Size3 = 64
Size4 = 64
Size5 = 32
Size6 = 10

Epochs = 20
BatchSize = 128
LearningRate = 0.001

X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])

# bool 자료형은 True, False을 반환하는 1비트 자료형
# 배치 정규화 과정에서 트레이닝 시에 배치 정규화를 하는지 안하는지 여부에 관여할 때 쓰인다.
# bool형 자료형을 is_training의 feed_dict 인자로 전달한 후 True를 
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

Logits, Predict = Build_NetworkNetwork_Function (X)

# 1. 손실도를 계산하고 그것을 최소화하는 학습을 진행할 것
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Y, logits = Logits))
LossTraining = tf.train.RMSPropOptimizer(LearningRate, decay = 0.9, momentum = 0.9).minimize(Lossfunction)

# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))


with tf.Session () as sess :
    
    sess.run(tf.global_variables_initializer())
    
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("......................")
    
    # Epochs Step만큼 기계 학습을 시작
    for Epoch in range (Epochs) :
        
        print ("Epoch is... ... %d회" %Epoch)  
        PrintLossfunction = 0.000
        
        for i in range (390) :
            # IsTraining의 bool 자료형을 True 값으로 전달한다.
            trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, TrainLabel_OneHotEncoding.eval())
            PrintLossTraining = sess.run (LossTraining, feed_dict = {X : trainBatch[0], Y : trainBatch[1], 
                                                                     keep_prob : 0.7, is_training : True})
            PrintLossfunction += sess.run (Lossfunction, feed_dict = {X : trainBatch[0], Y : trainBatch[1], 
                                                                      keep_prob : 1.0, is_training : True})
            
        PrintLossfunction = PrintLossfunction / 390
        print ("손실도   %f" %PrintLossfunction)
        

        # 한 번 Epoch가 끝날 때마다 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
        TestAccuracy = 0.000

        for i in range (10) :
            testBatch = Build_NextBatch_Function (1000, TestDataSet, TestLabel_OneHOtEncoding.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {X : testBatch[0], Y : testBatch[1], 
                                                                           keep_prob : 1.0, is_training : True})

        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산한다.
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주한다.
        TestAccuracy = TestAccuracy / 10
        print("테스트 데이터 정확도: %f" % TestAccuracy)
