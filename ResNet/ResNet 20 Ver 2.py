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


# 1, 2 레이어 빌드 이 함수 한 번만 호출해서 1, 2 레이어를 구성한다.
def Build_Convolution_Layer_1 (inputs):
    
    outputs = tf.contrib.layers.conv2d(
                                    inputs,
                                    num_outputs = 32, 
                                    kernel_size = 5, 
                                    stride = 1, 
                                    padding = "SAME", 
                                    activation_fn = None, 
                                    biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(
                                    outputs, 
                                    updates_collections = None, 
                                    is_training = phase)
    outputs = tf.nn.relu (outputs)
    print ("Convolution output size :   " %outputs)
    
    return outputs


# Convolution 1 X 2"
def Build_convolution_Layer_2 (inputs):

    def Build_BackBoneNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 32, 
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
        outputs = tf.contrib.layers.conv2d(
                                    outputs, 
                                    num_outputs = 64, 
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


    def Build_ResidualNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 64, 
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
    
    
    for index in range (2):
        
        if (index == 0):
            Result1 = Build_BackBoneNetwork_Fucntion (inputs)
            Result2 = Build_ResidualNetwork_Fucntion (inputs)
            ResultSum = Result1 + Result2
            
        else:
            Result1 = Build_BackBoneNetwork_Fucntion (ResultSum)
            Result2 = Build_ResidualNetwork_Fucntion (ResultSum)
            ResultSum = Result1 + Result2
            
    return ResultSum

# Convolution 3 X 3"
def Build_convolution_Layer_3 (inputs):

    def Build_BackBoneNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 64, 
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
        outputs = tf.contrib.layers.conv2d(
                                    outputs, 
                                    num_outputs = 128, 
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


    def Build_ResidualNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 128, 
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
    
    
    for index in range (2):
        
        if (index == 0):
            Result1 = Build_BackBoneNetwork_Fucntion (inputs)
            Result2 = Build_ResidualNetwork_Fucntion (inputs)
            ResultSum = Result1 + Result2
            
        else:
            Result1 = Build_BackBoneNetwork_Fucntion (ResultSum)
            Result2 = Build_ResidualNetwork_Fucntion (ResultSum)
            ResultSum = Result1 + Result2
            
    return ResultSum

# Convolution 4 X 2"
def Build_convolution_Layer_4 (inputs):

    def Build_BackBoneNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 128, 
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
        outputs = tf.contrib.layers.conv2d(
                                    outputs, 
                                    num_outputs = 256, 
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


    def Build_ResidualNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 256, 
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
    
    
    for index in range (2):
        
        if (index == 0):
            Result1 = Build_BackBoneNetwork_Fucntion (inputs)
            Result2 = Build_ResidualNetwork_Fucntion (inputs)
            ResultSum = Result1 + Result2
        
        else:
            Result1 = Build_BackBoneNetwork_Fucntion (ResultSum)
            Result2 = Build_ResidualNetwork_Fucntion (ResultSum)
            ResultSum = Result1 + Result2

    return ResultSum

# Convolution 5 X 2", 
def Build_convolution_Layer_5 (inputs):

    def Build_BackBoneNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 256, 
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
        outputs = tf.contrib.layers.conv2d(
                                    outputs, 
                                    num_outputs = 10, 
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


    def Build_ResidualNetwork_Fucntion (inputs):

        outputs = tf.contrib.layers.conv2d(
                                    inputs, 
                                    num_outputs = 10, 
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
    
    
    for index in range (2):
        
        if (index == 0):
            Result1 = Build_BackBoneNetwork_Fucntion (inputs)
            Result2 = Build_ResidualNetwork_Fucntion (inputs)
            ResultSum = Result1 + Result2
            
            
        else:
            Result1 = Build_BackBoneNetwork_Fucntion (ResultSum)
            Result2 = Build_ResidualNetwork_Fucntion (ResultSum)
            ResultSum = Result1 + Result2
            
    return ResultSum

def Build_MaxPooling_Function (inputs):

    outputs = tf.contrib.layers.max_pool2d(inputs, 
                                           kernel_size = [3, 3], 
                                           stride = [2, 2], 
                                           padding = "SAME")
    print (outputs)

    return outputs
    
    
def Build_FullyConnectedNetwork_Function (inputs):
    
    outputs = tf.contrib.layers.flatten (inputs)
    outputs = tf.contrib.layers.fully_connected (outputs, 
                                                 num_outputs = 1024, 
                                                 activation_fn = None)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.fully_connected (outputs, 
                                                 num_outputs = 10, 
                                                 activation_fn = None)
    
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


outputs = Build_Convolution_Layer_1 (Input_Layer) # 1
outputs = Build_convolution_Layer_2 (outputs) # 4
outputs = Build_convolution_Layer_3 (outputs) # 4

outputs = Build_convolution_Layer_4 (outputs) # 4
outputs = Build_MaxPooling_Function (outputs) # 1

outputs = Build_convolution_Layer_5 (outputs) # 4
outputs = Build_MaxPooling_Function (outputs) # 1

# 마지막 outputs는 Average Pooling 기법으로 차원수를 도리어 줄인다.
outputs = tf.reduce_mean(outputs, axis = [1, 2])
output = tf.reshape (outputs, [-1, 10])
print ("After Avg_Pooling Size is... ... ", outputs)


Logits = outputs
Predict = tf.nn.softmax (outputs)

# 1. 손실도를 계산하고 그것을 최소화하는 학습을 진행할 것
Lossfunction = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels = Label_Layer, logits = Logits))
LossTraining = tf.train.RMSPropOptimizer(LearningRate, decay = 0.9, momentum = 0.9).minimize(Lossfunction)

    
# 2. CorrectPrediction에 들어가서 정확도를 예측하는데에 쓰일 것
CorrectPrediction = tf.equal (tf.argmax(Label_Layer, 1), tf.argmax(Predict, 1))
Accuracy = tf.reduce_mean (tf.cast (CorrectPrediction, tf.float32))

LossValueList = []
AccuracyList = []

with tf.Session () as sess:
    
    sess.run(tf.global_variables_initializer())
    
    print ("----------------------------------------")
    print ("텐서플로우 세션을 열어서 학습을 시작합니다.")
    print ("----------------------------------------")
    
    # Epochs Step만큼 기계 학습을 시작
    for Epoch in range (Epochs + 170):
        
        print ("- Epoch is... ... %d회" %Epoch)
        LossValue = 0.000

        
        for i in range (390):
            
            trainBatch = Build_NextBatch_Function (BatchSize, TrainDataSet, Train_Labeling.eval())
            sess.run (LossTraining, feed_dict = {Input_Layer : trainBatch[0], 
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