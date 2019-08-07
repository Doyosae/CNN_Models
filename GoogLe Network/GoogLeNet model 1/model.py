import tensorflow as tf
import numpy as np

# 모듈이 잘 호출되었는지 테스트
Pi = 3.14159

# 모듈로 임포트할 때 Model.phase로 호출
phase = tf.placeholder(tf.bool)


# Inception 모델에서 처음에는 일반적인 컨볼루션을 연산한다.
# 이 모델에서는 필터 수 32개와 두 개의 맥스 풀링을 적용하였다. 
def Build_Convolution_Network (inputs):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = 32, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 32, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)   
    outputs = tf.contrib.layers.max_pool2d (outputs, kernel_size = [3, 3], stride = [2, 2], padding = "SAME")
    print ("컨볼루션 신경망 출력 크기 :   ", np.shape(outputs))
    
    return outputs



# Inception v2 모델을 구성하는 네 가지 함수를 구현
# Inception 모델 내부에는 Batch Normalization이 존재하지 않는다. 따라서 bias의 초기화가 필요하다.
def Build_Inception_Modular1 (inputs, size1):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular2 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular3 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Modular4 (inputs, size1):
    
    outputs = tf.contrib.layers.max_pool2d (inputs, kernel_size = [3, 3], stride = [1, 1], padding = "SAME")
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


# 위에서 함수로 정의한 인셉션 구성요소 모듈들을 이용하여 두 가지 종류의 인셉션 모델을 클래스로 정의한다.
# 스타일 1은 인셉션 모델 하나만 거치는 함수이고,
# 스타일 2는 인셉션 모델 두 개를 거치는 함수이다.
class Inception ():
    
    # 이 셀프 인자들의 값을 입력받고 필요한 스타일에 전달한다.
    def __init__ (self, size1, size2):
        
        self.size1 = size1
        self.size2 = size2
        
        
    # 클래스 아래 함수들의 입력 인자에는 항상 self가 있다.
    # 이 셀프 인자가 들어가면 그 중 해당하는 셀프.사이즈 변수가 인셉션 모듈러 함수의 변수로 들어간다.
    def Style1 (self, inputs):

        ModularOutput1 = Build_Inception_Modular1 (inputs, self.size1)
        ModularOutput2 = Build_Inception_Modular2 (inputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Modular3 (inputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Modular4 (inputs, self.size1)

        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Style1 모듈의 출력 크기 :    ", np.shape(outputs))

        return outputs


    def Style2 (self, inputs):
        
        ModularOutput1 = Build_Inception_Modular1 (inputs, self.size1)
        ModularOutput2 = Build_Inception_Modular2 (inputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Modular3 (inputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Modular4 (inputs, self.size1)
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        
        ModularOutput1 = Build_Inception_Modular1 (outputs, self.size1)
        ModularOutput2 = Build_Inception_Modular2 (outputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Modular3 (outputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Modular4 (outputs, self.size1)
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Style2 모듈의 출력 크기 :    ", np.shape(outputs))
        
        return outputs


# if __name__ == "__main__" 코드는 이 모듈을 직접 실행할 때에 작동한다.
# 다른 코드에서 import 할때는 작동하지 않는다. 따라서 코드의 테스트 목적으로 써도 된다.
if __name__ == "__main__":
    
    Inputs = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
    InceptionModel = Inception (32, 64)
    Outputs = InceptionModel.Style1 (Inputs)
     
    print ("NULL")
