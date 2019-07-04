import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
(TrainingInputData, TrainingOutputData), (TestingInputData, TestingOutputData) = load_data()




# 1. 데이터 구조가 어떻게 생겼는지 출력해본다.
TestData = (TrainingInputData, TrainingOutputData)
ValidData = (TestingInputData, TestingOutputData)
 
print ("훈련 데이터의 입력 사이즈", np.shape (TrainingInputData))      # 훈련 데이터의 사이즈
print ("결과 데이터의 출력 사이즈", np.shape (TrainingOutputData))     # 훈련 결과 이미지의 사이즈
print ("검사 데이터의 입력 사이즈", np.shape (TestingInputData))       # 테스트 데이터의 사이즈
print ("검사 데이터의 출력 사이즈", np.shape (TestingOutputData))      # 테스트 결과 이미지의 사이즈




# 2. 라벨링 데이터를 One - Hot - Encoding으로 바꾸는 구문
TrainingOutputData = tf.squeeze (tf.one_hot (TrainingOutputData, 10), axis = 1)
print (np.shape (TrainingOutputData))

TestingOutputData = tf.squeeze (tf.one_hot (TestingOutputData, 10), axis = 1)
print (np.shape (TestingOutputData))

# 라벨링된 데이터가 tensorflow의 배열 형태라서 Session을 실행해야 출력이 가능하다.
# 라벨링된 데이터가 One Hot Encoding으로 잘 바뀌었는지 확인할 것
sess = tf.Session()
sess.run (tf.global_variables_initializer())

for k in range (10) : 
    LabelingPrint = sess.run (TrainingOutputData[k])
    print (LabelingPrint)
    
sess.close()




# 3. 학습 데이터의 자료 크기를 출력해볼 것 with len 명령어
TotalBatchSize = len (TrainingInputData)
print (TotalBatchSize)




# 4. RGB 이미지를 그레이스케일로 변환해보는 예시
np.shape (TrainingInputData)
RedScaleImage = TrainingInputData [ :, :, :, 0]
GreenScaleImage = TrainingInputData [ :, :, :, 1]
BlueScaleImage = TrainingInputData [ :, :, :, 2]
print (np.shape(RedScaleImage))
print (np.shape(GreenScaleImage))
print (np.shape(BlueScaleImage))

# RGB 스케일 이미지들의 행렬값들을 모두 더한다.
PreprocessingGREYScaleImage = RedScaleImage + GreenScaleImage + BlueScaleImage
AfterPreprocessingGREYScaleImage = PreprocessingGREYScaleImage / 3

plt.subplot (4, 4, 1)
plt.imshow (AfterPreprocessingGREYScaleImage[0])
plt.subplot (4, 4, 2)
plt.imshow (RedScaleImage[0])
plt.subplot (4, 4, 3)
plt.imshow (GreenScaleImage[0])
plt.subplot (4, 4, 4)
plt.imshow (BlueScaleImage[0])
plt.subplots_adjust(hspace = 0.5, wspace = 1)
