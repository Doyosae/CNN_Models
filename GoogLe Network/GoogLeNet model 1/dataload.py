import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data

# load_data로부터 cifar10의 훈련, 검증 데이터셋을 분리
(TrainData, TrainLabel), (TestData, TestLabel) = load_data ()

# 라벨 데이터셋을 One-Hot 인코딩 처리
SqueezedTrainLabel = tf.squeeze (tf.one_hot (TrainLabel, 10), axis = 1)
SqueezedTestLabel  = tf.squeeze (tf.one_hot (TestLabel, 10),  axis = 1)

# 데이터셋을 받아서 Batch를 만드는 함수
def Build_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)

# Test
Pi = 3.14159

if __name__ == "__main__":

    print ("훈련 이미지의 크기           ", np.shape (TrainData))
    print ("훈련 라벨링의 크기           ", np.shape (SqueezedTrainLabel))
    print ("검증 이미지의 크기           ", np.shape (TestData))
    print ("검증 라벨링의 크기           ", np.shape (SqueezedTestLabel))