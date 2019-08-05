import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainDataSet, TrainLabelSet), (TestDataSet, TestLabelSet) = load_data ()
    
TrainData = (TrainDataSet, TrainLabelSet)
TestDat   = (TestDataSet, TestLabelSet)

SqueezedTrainLabel = tf.squeeze (tf.one_hot (TrainLabelSet, 10), axis = 1)
SqueezedTestLabel  = tf.squeeze (tf.one_hot (TestLabelSet, 10),  axis = 1)


def Build_Convolution_Network (inputs):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = 16, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 16, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)   
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    print ("컨볼루션 신경망 출력 크기 :   ", np.shape(outputs))
    
    return outputs


def Build_Inception_Modular1 (inputs, size1):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular2 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular3 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular4 (inputs, size1):
    
    outputs = tf.contrib.layers.max_pool2d (inputs, kernel_size = [3, 3], stride = [1, 1], padding = "SAME")
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.nn.relu (outputs)
    
    return outputs


# 데이터셋으로 부터 배치를 만드는 함수
def Build_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


# 위에서 함수로 정의한 인셉션 구성요소 모듈들을 이용하여 두 가지 종류의 인셉션 모델을 클래스로 정의한다.
# 스타일 1은 인셉션 모델 하나만 거치는 함수이고,
# 스타일 2는 인셉션 모델 두 개를 거치는 함수이다.
class Inception ():
    
    # 이 셀프 인자들의 값은 각각 32, 64, 64를 입력받을 것
    def __init__ (self, size1, size2):
        
        self.size1 = size1 #32
        self.size2 = size2 #64
        
        
    # 클래스 아래 함수들의 입력 인자에는 항상 self가 있다.
    # 이 셀프 인자가 들어가면 그 중 해당하는 셀프.사이즈 변수가 인셉션 모듈러 함수의 변수로 들어간다.
    def Style1 (self, inputs):

        ModularOutput1 = Build_Inception_Modular1 (inputs, self.size1) #32
        ModularOutput2 = Build_Inception_Modular2 (inputs, self.size1, self.size2) #32, 64
        ModularOutput3 = Build_Inception_Modular3 (inputs, self.size1, self.size2) #32, 64
        ModularOutput4 = Build_Inception_Modular4 (inputs, self.size1) #32

        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Style1 모듈의 출력 크기 :    ", np.shape(outputs))

        return outputs


    def Style2 (self, inputs):
        
        ModularOutput1 = Build_Inception_Modular1 (inputs, self.size1) #32
        ModularOutput2 = Build_Inception_Modular2 (inputs, self.size1, self.size2) #32, 64
        ModularOutput3 = Build_Inception_Modular3 (inputs, self.size1, self.size2) #32, 64
        ModularOutput4 = Build_Inception_Modular4 (inputs, self.size1) #32
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        
        ModularOutput1 = Build_Inception_Modular1 (outputs, self.size1) #32
        ModularOutput2 = Build_Inception_Modular2 (outputs, self.size1, self.size2) #32, 64
        ModularOutput3 = Build_Inception_Modular3 (outputs, self.size1, self.size2) #32, 64
        ModularOutput4 = Build_Inception_Modular4 (outputs, self.size1) #32
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Style2 모듈의 출력 크기 :    ", np.shape(outputs))
        
        return outputs

    
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

# 서로 다른 필터 수를 가지는 인셉션 모델들을 개별 클래스로 호출한다.
InceptionModel1 = Inception(32, 64)
InceptionModel2 = Inception(64, 128)
InceptionModel3 = Inception(128, 256)

outputs = Build_Convolution_Network (Input_Layer)
outputs = InceptionModel1.Style2(outputs)
OUTPUTS1 = outputs
print ("\n")

outputs = InceptionModel2.Style2(outputs)
OUTPUTS2 = outputs
print ("\n")

outputs = InceptionModel3.Style2(outputs)
OUTPUTS3 = outputs
print ("\n")

# 각 인셉션 모듈에서 뽑아는 리스트를 AvgPool을 취하여 성분 갯수가 채널 수에만 의존하는 1차원 형태로 바꾸어준다.
AvgPool1 = tf.reduce_mean(OUTPUTS1, axis = [1, 2])
AvgPool2 = tf.reduce_mean(OUTPUTS2, axis = [1, 2])
AvgPool3 = tf.reduce_mean(OUTPUTS3, axis = [1, 2])
print (np.shape(AvgPool2))
print (np.shape(AvgPool1))
print (np.shape(AvgPool3))


# 1차원 형태로 바꾼 텐서 리스트들을 각각의 크기에 맞게 Fully Connected Layer를 구현한다.
HiddenOutput1 = tf.contrib.layers.fully_connected (AvgPool1, 100, activation_fn = None)
HiddenOutput1 = tf.contrib.layers.batch_norm(HiddenOutput1, updates_collections = None, is_training = phase)
HiddenOutput1 = tf.nn.relu(HiddenOutput1)
output1 = tf.contrib.layers.fully_connected (HiddenOutput1, 10, activation_fn = None)

HiddenOutput2 = tf.contrib.layers.fully_connected (AvgPool2, 200, activation_fn = None)
HiddenOutput2 = tf.contrib.layers.batch_norm(HiddenOutput2, updates_collections = None, is_training = phase)
HiddenOutput2 = tf.nn.relu(HiddenOutput2)
output2 = tf.contrib.layers.fully_connected (HiddenOutput2, 10, activation_fn = None)

HiddenOutput3 = tf.contrib.layers.fully_connected (AvgPool3, 300, activation_fn = None)
HiddenOutput3 = tf.contrib.layers.batch_norm(HiddenOutput3, updates_collections = None, is_training = phase)
HiddenOutput3 = tf.nn.relu(HiddenOutput3)
output3 = tf.contrib.layers.fully_connected (HiddenOutput3, 10, activation_fn = None)


# 세 가지 output들을 모두 더해서 그것의 평균을 구한다.
RealOutput = (output1 + output2 + output3) / 3
print (np.shape(output))


# RealOutput을 Logits으로 부여하고, Cross_Entropy를 손실값과 손실함수를 정의
Logits = RealOutput
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = Logits))
LossTraining1 = tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.95).minimize(Lossfunction)
LossTraining2 = tf.train.AdamOptimizer(LearningRate*0.5, beta1 = 0.9, beta2 = 0.95).minimize(Lossfunction)


# RealOutput에 Softmax를 씌운 그 Predict값이 Argmax하였을때 얼마나 정답 레이블과 일치하는지를 보는것
Predict = tf.nn.softmax (RealOutput)
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
        
        print ("- Epoch :   %d회" %(Epoch+1))
        LossValue = 0.000
        
        if Epoch < 30:
        
            for i in range (390):

                trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, SqueezedTrainLabel.eval())
                sess.run (LossTraining1, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1],
                                                      phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                             Label_Layer : trainBatch[1],
                                                                             phase : True})
                
        else:
            
            for i in range (390):

                trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, SqueezedTrainLabel.eval())
                sess.run (LossTraining2, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], 
                                                      phase : True})
                LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], 
                                                                             Label_Layer : trainBatch[1], 
                                                                             phase : True})
            
        LossValue = LossValue / 390
        LossValueList.append (LossValue)
        print ("   손실도는 :            %f" %LossValue)

        
        # 학습하지 않은 테스트 이미지로 학습한 모델에 대해서 그 정확도를 계산한다.
        # 테스트 이미지는 총 10,000 장이다. 1,000장 단위로 배치를 만들고 총 10번 반복하여 테스트한다.
        # 각 배치마다의 Accuracy를 모두 더한 후, (Summation n=10) 다시 10으로 나누어서 Acc의 평균값을 구한다.
        TestAccuracy = 0.000
        
        for i in range (10):
            
            testBatch = Build_NextBatch_Function (1000, TestDataSet, SqueezedTestLabel.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {Input_Layer : testBatch[0], 
                                                                           Label_Layer : testBatch[1], 
                                                                           phase : False})

        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산한다.
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주한다.
        TestAccuracy = TestAccuracy / 10
        AccuracyList.append (TestAccuracy)
        print("   테스트 데이터 정확도:   %.4f" %TestAccuracy)
