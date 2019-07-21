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


# 1. 입력으로 들어가는 데이터 세트의 크기만큼 np.arange를 이용하여 List를 생성 (Cifar-10 데이터는 50,000개 이므로 50,000 리스트 생성)
# 2. 이제 이 리스트의 원소들을 shuffle 해준다. 랜덤으로 정렬된 원소의 인덱스에 해당하는 데이터들을 뽑아서 ShuffleSet를 새로 만든다.
def generate_NextBatch_Function (number, data, labels) :
    
    DataRange = np.arange (0 , len(data))
    np.random.shuffle (DataRange)
    DataRange = DataRange [ : number]
    
    DataShuffle = [data[i] for i in DataRange]
    LabelsShuffle = [labels[i] for i in DataRange]

    return np.asarray(DataShuffle), np.asarray(LabelsShuffle)


# 3. BatchSize는 256으로 하고, Batch Set 함수를 호출한 뒤에 return 받은 batch의 크기를 출력해볼 것
batch = generate_NextBatch_Function (256, TrainDataSet, TrainLabel_OneHotEncoding)
print ("batch[0]의 크기", np.shape(batch[0]))
print ("batch[1]의 크기", np.shape(batch[1]))


# 4. Session을 열어서 특별히 batch[1] 내용들을 출력해볼 것
sess = tf.Session()
sess.run (tf.global_variables_initializer())

for k in range (10) : 
    LabelingPrint = sess.run (batch[1][k])
    print (LabelingPrint)
    
sess.close()
