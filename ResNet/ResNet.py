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

class Define_ResNet_Class ():
    
    def __init__ (self, kernel_size, channel_size):
        
        self.kernel_size1 = kernel_size
        self.kernel_size2 = kernel_size + 2
        self.channel_size = channel_size

    def Build_BackBoneNetwork_Fucntion (self, inputs, phase):

        outputs = tf.contrib.layers.conv2d (inputs, num_outputs = self.channel_size, kernel_size = self.kernel_size1, 
                                            stride = 1, padding = "SAME", activation_fn = None, biases_initializer = None)
        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
        outputs = tf.nn.relu (outputs)
        outputs = tf.contrib.layers.conv2d (outputs, num_outputs = self.channel_size, kernel_size = self.kernel_size1, 
                                            stride = 1, padding = "SAME", activation_fn = None, biases_initializer = None)
        outputs = tf.contrib.layers.batch_norm (outputs, updates_collections = None, is_training = phase)
        outputs = tf.nn.relu (outputs)

        return outputs
    
    
    def Build_ResidualNetwork_Fucntion (self, inputs, phase):

        outputs = tf.contrib.layers.conv2d (inputs, num_outputs = self.channel_size, kernel_size = self.kernel_size2, 
                                            stride = 1, padding = "SAME", activation_fn = None, biases_initializer = None)
        outputs = tf.nn.relu (outputs)
        outputs = tf.contrib.layers.batch_norm (outputs, updates_collections = None, is_training = phase)

        return outputs
    
    
def Build_MaxPooling_Function (inputs):
    
    outputs = tf.contrib.layers.max_pool2d (inputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    print (outputs)
    
    return outputs


def Build_FullyConnectedNetwork_Function (inputs, input_size):
    
    outputs = tf.reshape (inputs, [-1, input_size])
    outputs = tf.contrib.layers.fully_connected (outputs, num_outputs = int(input_size/2), activation_fn = None)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.fully_connected (outputs, num_outputs = 10, activation_fn = None)
    
    return outputs


def Build_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


# 필요한 상수와 변수 지정, 클래스 지정
Epochs = 30
BatchSize = 128
LearningRate = 0.001

Input_Layer = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
Label_Layer = tf.placeholder (tf.float32, shape = [None, 10])
phase = tf.placeholder(tf.bool)


# 클래스 호출, 입력 값은 (커널 사이즈, 필터의 갯수)
# 백본 네트워크는 커널 사이즈 3 by 3, 레지듀얼 네트워크는 5 by 5로 정의
ResNetLayer_64 = Define_ResNet_Class (3, 64)
ResNetLayer_96 = Define_ResNet_Class (3, 96)
ResNetLayer_128 = Define_ResNet_Class (3, 128)
ResNetLayer_256 = Define_ResNet_Class (3, 256)

# Build Model
# ResNet Class No 1.
# 필터 64개를 가지는 ResNet 모듈
outputs1 = ResNetLayer_64.Build_BackBoneNetwork_Fucntion (Input_Layer, phase)
outputs2 = ResNetLayer_64.Build_ResidualNetwork_Fucntion (Input_Layer, phase)
outputs = outputs1 + outputs2
outputs = Build_MaxPooling_Function (outputs)

# ResNet Class No 2.
# 필터 96개를 가지는 ResNet 모듈
outputs1 = ResNetLayer_96.Build_BackBoneNetwork_Fucntion (outputs, phase)
outputs2 = ResNetLayer_96.Build_ResidualNetwork_Fucntion (outputs, phase)
outputs = outputs1 + outputs2
outputs = Build_MaxPooling_Function (outputs)

# ResNet Class No 3.
# 필터 128개를 가지는 ResNet 모듈
outputs1 = ResNetLayer_128.Build_BackBoneNetwork_Fucntion (outputs, phase)
outputs2 = ResNetLayer_128.Build_ResidualNetwork_Fucntion (outputs, phase)
outputs = outputs1 + outputs2
outputs = Build_MaxPooling_Function (outputs)

# ResNet Class No 4.
# 필터 256개를 가지는 ResNet 모듈
outputs1 = ResNetLayer_256.Build_BackBoneNetwork_Fucntion (outputs, phase)
outputs2 = ResNetLayer_256.Build_ResidualNetwork_Fucntion (outputs, phase)
outputs = outputs1 + outputs2
outputs = Build_MaxPooling_Function (outputs)

# 마지막 Fully Connectec Network 모듈을 호출
outputs = Build_FullyConnectedNetwork_Function (outputs, 1024)

Logits = outputs
Predict = tf.nn.softmax (outputs)
print (Logits)
print (Predict)


# 1. 손실도를 계산하고 그것을 최소화하는 학습을 진행할 것
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = Logits))
LossTraining = tf.train.RMSPropOptimizer(LearningRate, decay = 0.9, momentum = 0.9).minimize(Lossfunction)

# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
CorrectPrediction = tf.equal (tf.argmax(Label_Layer, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

# LossValue와 Accuracy를 저장할 리스트 정의
LossValueList = []
AccuracyList = []


with tf.Session () as sess:
    
    sess.run(tf.global_variables_initializer())
    
    print ("----------------------------------------")
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("----------------------------------------")
    
    # Epochs Step만큼 기계 학습을 시작
    for Epoch in range (Epochs):
        
        print ("- Epoch is... ... %d회" %Epoch)
        LossValue = 0.000

        
        for i in range (390):
            
            trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, Train_Labeling.eval())
            sess.run (LossTraining, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], phase : True})
            LossValue = LossValue + sess.run (Lossfunction, feed_dict = {Input_Layer : trainBatch[0], Label_Layer : trainBatch[1], phase : True})
            
            
        LossValue = LossValue / 390
        LossValueList.append (LossValue)
        print ("손실도는 대략... %f" %LossValue)
            

        # 한 번 Epoch가 끝날 때마다 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력
        TestAccuracy = 0.000

        for i in range (10):
            
            testBatch = Build_NextBatch_Function (1000, TestDataSet, Test_Labeling.eval())
            TestAccuracy = TestAccuracy + sess.run (Accuracy, feed_dict = {Input_Layer : testBatch[0], Label_Layer : testBatch[1], phase : False})

        # 테스트 데이터 10,000개를 1,000개 단위의 배치로 잘라서 각 배치의 Acc를 계산한다.
        # 10개의 Acc를 모두 더한 후, 10으로 나눈 Avg Acc를 Epoch 당 테스트 정확도로 간주한다.
        TestAccuracy = TestAccuracy / 10
        AccuracyList.append (TestAccuracy)
        print("테스트 데이터 정확도: %f" %TestAccuracy)
