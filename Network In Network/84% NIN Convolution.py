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
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

Logits, Predict = Build_NetworkNetwork_Function (X)

Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Y, logits = Logits))
LossTraining = tf.train.RMSPropOptimizer(LearningRate, decay = 0.9, momentum = 0.9).minimize(Lossfunction)


# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
CorrectPrediction = tf.equal (tf.argmax(Y, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

with tf.Session () as sess :
    
    sess.run(tf.global_variables_initializer())
    
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("......................")
    
    # 5000 Step만큼 최적화를 수행, 200 Step마다 학습 데이터에 대해여 정확도와 손실을 출력
    for Epoch in range (Epochs) :
        
        print ("Epoch is... ... %d회" %Epoch)  
        PrintLossfunction = 0.000
        
        for i in range (390) :
            
            # IsTraining을 True로 켜고 진행
            trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, TrainLabel_OneHotEncoding.eval())
            PrintLossTraining = sess.run (LossTraining, feed_dict = {X : trainBatch[0], Y : trainBatch[1], 
                                                                     keep_prob : 0.7, is_training : True})
            PrintLossfunction = PrintLossfunction + sess.run (Lossfunction, feed_dict = {X : trainBatch[0], Y : trainBatch[1], 
                                                                                         keep_prob : 1.0, is_training : True})
            
        PrintLossfunction = PrintLossfunction / 390
              
        print ("손실도   %f" %PrintLossfunction)
        
        
        
        # 한 번 Epoch가 끝날 때마다 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
        TestAccuracy = 0.000

        for i in range (10) :

            testBatch = Build_NextBatch_Function (1000, TestDataSet, TestLabel_OneHOtEncoding.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {X : testBatch[0], Y : testBatch[1], 
                                                                           keep_prob : 1.0, is_training : True})


        # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후, 모두 더한 다음 10으로 나누는 것
        TestAccuracy = TestAccuracy / 10
        print("테스트 데이터 정확도: %f" % TestAccuracy)
