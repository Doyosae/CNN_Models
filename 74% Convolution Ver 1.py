import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(X_training, Y_training), (X_test, Y_test) = load_data ()

TestData = (X_training, Y_training) ; ValidData = (X_test, Y_test)

BatchSize = 256

X = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Y = tf.placeholder (tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)

Y_training_OneHotLabeling = tf.squeeze (tf.one_hot (Y_training, 10), axis = 1)
Y_test_OneHotLabeling = tf.squeeze (tf.one_hot (Y_test, 10), axis = 1)

print (np.shape (Y_training))
print (np.shape (Y_test))

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

outputs = tf.contrib.layers.conv2d (X, num_outputs = 96, kernel_size = 3, stride = 1, padding = "SAME", activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 96, kernel_size = 3, stride = 1, padding = "SAME", activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
print ("두 번째 컨볼루션 레이어에서의 풀링 사이즈", outputs)

outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 196, kernel_size = 3, stride = 1, padding = "SAME", activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 196, kernel_size = 3, stride = 1, padding = "SAME", activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
print ("네 번째 컨볼루션 레이어에서의 풀링 사이즈", outputs)

# Session 2. Convolution Nueral Network가 끝난 후, Fully Coneected Layer로 연결
outputs = tf.reshape (outputs, [-1, 8 * 8 * 196])
print (outputs)

outputs = tf.contrib.layers.fully_connected (outputs, num_outputs = 2048, activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.nn.dropout (outputs, rate = 1 - keep_prob)
outputs = tf.contrib.layers.fully_connected (outputs, num_outputs = 1024, activation_fn=None)
outputs = tf.nn.relu (outputs)
outputs = tf.nn.dropout (outputs, rate = 1 - keep_prob)
outputs = tf.contrib.layers.fully_connected (outputs, num_outputs = 10, activation_fn=None)
                                   
# 1. Hypothesis는 Cross-Entropy-Softmax 손실도에 들어갈 것
# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
Hypothesis = outputs
print (Hypothesis)
Prediction = tf.nn.softmax (Hypothesis)
print (Prediction)

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
        sess.run (LossTrainingStep, feed_dict = {X : batch[0], Y : batch[1], keep_prob : 0.3})

        if Epoch % 100 == 0 :
            
            TrainAccuracy = Accuracy.eval (feed_dict = {X : batch[0], Y : batch[1], keep_prob: 1.0})
            PrintLoss = Lossfunction.eval (feed_dict = {X : batch[0], Y : batch[1], keep_prob: 1.0})

            print("Epoch  : %d,   트레이닝 데이터 정확도 : %f,   손실도 : %f" % (Epoch, TrainAccuracy, PrintLoss))

            
    # Last Session. 학습이 모두 끝나고 테스트 데이터를 넣어서 그 정확도를 출력
    TestAccuracy = 0.000
    
    for i in range (10) :
        TestBatch = Next_Batch_Function(1000, X_test, Y_test_OneHotLabeling.eval())
        TestAccuracy = TestAccuracy + Accuracy.eval (feed_dict = {X : TestBatch[0], Y : TestBatch[1], keep_prob: 1.0})
        
    # 테스트 데이터 10000개를 1000개 단위의 배치로 잘라서 각 배치마다의 정확도를 측정한 후, 모두 더한 다음 10으로 나누는 것
    TestAccuracy = TestAccuracy / 10
    print("테스트 데이터 정확도: %f" % TestAccuracy)
