import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainDataSet, TrainLabelSet), (TestDataSet, TestLabelSet) = load_data ()

Train_Labeling = tf.squeeze (tf.one_hot (TrainLabelSet, 10), axis = 1)
Test_Labeling  = tf.squeeze (tf.one_hot (TestLabelSet, 10), axis = 1)

print ("훈련 이미지의 크기           ", np.shape (TrainDataSet))
print ("훈련 라벨링의 크기           ", np.shape (Train_Labeling))
print ("검사 이미지의 크기           ", np.shape (TestDataSet))
print ("검사 라벨링의 크기           ", np.shape (Test_Labeling))


# 1. 보통의 컨볼루션 연산을 수행하는 함수
def Build_ConvolutionNetwork (inputs, number1):
    
    outputs = tf.contrib.layers.conv2d(
                                inputs, 
                                num_outputs = number1, 
                                kernel_size = 3, 
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs, 
                                updates_collections = None, 
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    return outputs

# 2. Residual Net 연산을 실행할 함수
def Build_ResidualNetwork (inputs, number1):
    
    outputs = tf.contrib.layers.conv2d(
                                inputs, 
                                num_outputs = number1, 
                                kernel_size = 3, 
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs, 
                                updates_collections = None, 
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    return outputs

# 3. 가장 중요한 Backbone Net 함수, 이 내부에서 Residual Net을 호출한다.
def Build_BackboneNetwork (inputs, number1, number2):
    
    ResidualOutputs1 = Build_ResidualNetwork (inputs, number1)
    
    outputs = tf.contrib.layers.conv2d(
                                inputs, 
                                num_outputs = number1, 
                                kernel_size = 3, 
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs, 
                                updates_collections = None, 
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    ResidualOutputs2 = Build_ResidualNetwork (outputs, number2)
    
    outputs = tf.contrib.layers.conv2d(
                                outputs, 
                                num_outputs = number1, 
                                kernel_size = 3,                      
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs,
                                updates_collections = None,
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    outputs = outputs + ResidualOutputs1
    
    outputs = tf.contrib.layers.conv2d(
                                inputs, 
                                num_outputs = number2, 
                                kernel_size = 3, 
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs, 
                                updates_collections = None, 
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    outputs = outputs + ResidualOutputs2
    
    outputs = tf.contrib.layers.conv2d(
                                outputs, 
                                num_outputs = number2, 
                                kernel_size = 3,                      
                                stride = 1, 
                                padding = "SAME", 
                                activation_fn = None, 
                                biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                outputs,
                                updates_collections = None,
                                is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    return outputs

# 맥스 풀링 함수
def Build_MaxPooling_Function (inputs):

    outputs = tf.contrib.layers.max_pool2d(inputs, 
                                           kernel_size = [3, 3], 
                                           stride = [2, 2], 
                                           padding = "SAME")
    print (outputs)

    return outputs

# 배치 사이즈에 맞게 데이터셋을 셔플하는 함수
def Build_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


"""
필요한 고정 상수들을 설정, 그래프를 빌드한다.
그리고 손실 함수와 최적화 함수도 정의
"""
Epochs = 80
BatchSize = 128
LearningRate = 0.001

Input_Layer = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Label_Layer = tf.placeholder (tf.float32, shape = [None, 10])
phase = tf.placeholder(tf.bool)


outputs = Build_ConvolutionNetwork (Input_Layer, 32)

outputs = Build_BackboneNetwork (outputs, 32, 64)
outputs = Build_BackboneNetwork (outputs, 64, 128)
outputs = Build_MaxPooling_Function (outputs)

outputs = Build_BackboneNetwork (outputs, 128, 196)
outputs = Build_BackboneNetwork (outputs, 196, 320)
outputs = Build_MaxPooling_Function (outputs)

outputs = Build_ConvolutionNetwork (outputs, 10)
print (outputs)

outputs = tf.reduce_mean(outputs, axis = [1, 2])
print ("After Avg_Pooling Size is... ... ", outputs)

Logits = outputs
Predict = tf.nn.softmax (outputs)

# 1. Epoch가 30보다 작으면 LossTraining1 실행하고, Epoch가 30보다 크면 LossTraining2 실행한다.
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = Logits))
LossTraining1 = tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.95).minimize(Lossfunction)
LossTraining2 = tf.train.AdamOptimizer(LearningRate*0.1, beta1 = 0.9, beta2 = 0.95).minimize(Lossfunction)
    
# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
CorrectPrediction = tf.equal (tf.argmax(Label_Layer, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

LossValueList = []
AccuracyList = []

# 세션을 열고 그래프를 실행하여 학습합니다.
with tf.Session () as sess:    
    
    sess.run(tf.global_variables_initializer())
    
    print ("----------------------------------------")
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("----------------------------------------")
    
    # Epochs Step만큼 기계 학습을 시작
    for Epoch in range (Epochs):
        
        print ("- Epoch is... ... %d회" %Epoch)
        LossValue = 0.000
        
        if Epoch < 30:
        
            for i in range (390):

                trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, Train_Labeling.eval())
                sess.run (LossTraining1, feed_dict = {Input_Layer : trainBatch[0], 
                                                      Label_Layer : trainBatch[1], 
                                                      phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                              Label_Layer : trainBatch[1], 
                                                                              phase : True})
                
        else:
            
            for i in range (390):

                trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, Train_Labeling.eval())
                sess.run (LossTraining2, feed_dict = {Input_Layer : trainBatch[0], 
                                                      Label_Layer : trainBatch[1], 
                                                      phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                              Label_Layer : trainBatch[1], 
                                                                              phase : True})
            
        LossValue = LossValue / 390
        LossValueList.append (LossValue)
        print ("손실도는 대략... %f" %LossValue)
            

        # 한 번 Epoch가 끝날 때마다 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
        TestAccuracy = 0.000

        for i in range (10):
            
            testBatch = Build_NextBatch_Function (1000, TestDataSet, Test_Labeling.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {Input_Layer : testBatch[0], 
                                                                           Label_Layer : testBatch[1], 
                                                                           phase : False})

        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산한다.
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주한다.
        TestAccuracy = TestAccuracy / 10
        AccuracyList.append (TestAccuracy)
        print("테스트 데이터 정확도: %.4f" %TestAccuracy)
