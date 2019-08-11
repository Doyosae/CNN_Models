import tensorflow as tf
import numpy as np

# 모듈이 잘 호출되었는지 보기 위해서 아래 상수를 호출
Pi = 3.14159


# 모듈로 임포트할 때 Model.phase로 호출
phase = tf.placeholder(tf.bool)


# Stem Convolution
# 1. Inception 모델 구조에서 처음에는 일반적인 컨볼루션을 연산한다.
# 이 함수에서는 필터 수 32개와 두 개의 맥스 풀링을 적용하였다. 
def Build_Convolution_Network_v1 (inputs):
    
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

# 2. Inception 모델 구조에서 처음에는 일반적인 컨볼루션을 연산한다.
# 이 함수는 Redunction 모듈과 함께 사용할 것을 권장한다.
# 필터 수 32개를 유지하되, 맥스 풀링 구문은 제거 하였다.
def Build_Convolution_Network_v2 (inputs):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = 32, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)
    
    outputs = tf.contrib.layers.conv2d (outputs, num_outputs = 32, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, biases_initializer = None)
    outputs = tf.contrib.layers.batch_norm(outputs, updates_collections = None, is_training = phase)
    outputs = tf.nn.relu (outputs)   
    print ("컨볼루션 신경망 출력 크기 :   ", np.shape(outputs))
    
    return outputs

#######################################################################################################################

# Inception v2 모델을 구성하는 네 가지 함수를 구현
# Inception 모델 내부에는 Batch Normalization이 존재하지 않는다. 따라서 bias의 초기화가 필요하다.
def Build_Inception_Branch1 (inputs, size1):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Branch2 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Branch3 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


def Build_Inception_Branch4 (inputs, size1):
    
    outputs = tf.contrib.layers.max_pool2d (inputs, kernel_size = [3, 3], stride = [1, 1], padding = "SAME")
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs

#######################################################################################################################

# 1번의 3 x 3 컨볼루션 연산, 스트라이드 2
def Build_Reduction_Branch1 (inputs, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, 
                                        num_outputs = size2, 
                                        kernel_size = 3, 
                                        stride = 2, 
                                        padding = "SAME", 
                                        activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs


# 1번의 1 x 1 컨볼루션 연산, 스트라이드 1
# 1번의 3 x 3 컨볼루션 연산, 스트라이드 2
def Build_Reduction_Branch2 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 2, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs

# 1번의 1 x 1 컨볼루션 연산, 스트라이드 1
# 1번의 3 x 3 컨볼루션 연산, 스트라이드 1
# 1번의 3 x 3 컨볼루션 연산, 스트라이드 2
def Build_Reduction_Branch3 (inputs, size1, size2):
    
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 1, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size1, kernel_size = 3, stride = 1, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    outputs = tf.contrib.layers.conv2d (inputs, num_outputs = size2, kernel_size = 3, stride = 2, 
                                        padding = "SAME", activation_fn = None, 
                                        biases_initializer = tf.random_uniform_initializer(-0.1, 0.1))
    outputs = tf.nn.relu (outputs)
    
    return outputs

# 1번의 맥스 풀링 연산, 스트라이드 2
def Build_Redunction_MaxPool_Branch (inputs, size1):
    
    outputs = tf.contrib.layers.max_pool2d (inputs, 
                                            kernel_size = [3, 3], 
                                            stride = [2, 2], 
                                            padding = "SAME")
    outputs = tf.nn.relu (outputs)
    
    return outputs


#######################################################################################################################
#######################################################################################################################


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
    def StyleA1 (self, inputs):

        ModularOutput1 = Build_Inception_Branch1 (inputs, self.size1)
        ModularOutput2 = Build_Inception_Branch2 (inputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Branch3 (inputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Branch4 (inputs, self.size1)

        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Inception v2 Style1 모듈의 출력 크기 :    ", np.shape(outputs))

        return outputs


    def StyleA2 (self, inputs):
        
        ModularOutput1 = Build_Inception_Branch1 (inputs, self.size1)
        ModularOutput2 = Build_Inception_Branch2 (inputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Branch3 (inputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Branch4 (inputs, self.size1)
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        
        ModularOutput1 = Build_Inception_Branch1 (inputs, self.size1)
        ModularOutput2 = Build_Inception_Branch2 (inputs, self.size1, self.size2)
        ModularOutput3 = Build_Inception_Branch3 (inputs, self.size1, self.size2)
        ModularOutput4 = Build_Inception_Branch4 (inputs, self.size1)
        
        outputs = tf.concat ([ModularOutput1, ModularOutput2, ModularOutput3, ModularOutput4], -1)
        print ("Inception v2 Style2 모듈의 출력 크기 :    ", np.shape(outputs))
        
        return outputs


# 위에서 함수로 정의한 리덕션 구성요소 모듈들을 이용하여 두 가지 종류의 리덕션 모델을 클래스로 정의한다.
# Redunction_StyleA 기본적인 Redunction 모델이다.
# Redunction_StyleB 현재 설계 중......
class Reduction ():
    
    # 이 셀프 인자들의 값을 입력받고 필요한 스타일에 전달한다.
    def __init__ (self, size1, size2):
        
        self.size1 = size1
        self.size2 = size2

    def StyleA (self, inputs):

        Output1 = Build_Redunction_MaxPool_Branch (inputs, self.size1)
        Output2 = Build_Reduction_Branch1 (inputs, self.size1)
        Output3 = Build_Reduction_Branch3 (inputs, self.size1, self.size2)

        outputs = tf.concat ([Output1, Output2, Output3], -1)
        print ("Reductoin A 모듈의 출력 크기 :    ", np.shape(outputs))

        return outputs

    # Redunction_StyleB 설계 중...

# if __name__ == "__main__" 코드는 이 모듈을 직접 실행할 때에 작동한다.
# 다른 코드에서 import 할때는 작동하지 않는다. 따라서 코드의 테스트 목적으로 써도 된다.
if __name__ == "__main__":
    
    Inputs = tf.placeholder (tf.float32, shape = [None, 32, 32, 3])
    
    # 인셉션에서 첫 번째 인자는 1 x 1 컨볼루션의 채널수이다.
    # 입력 채널보다 작아도 된다.
    # 그러나 리덕션은 그렇게 사용하지 않는다.
    InceptionModel1 = Inception (16, 16)
    InceptionModel2 = Inception (80, 80)

    ReducntionModel1 = Reduction (32, 64)
    ReducntionModel2 = Reduction (160, 320)

    InceptionModel3 = Inception (400, 400)
    RedunctionModel3 = Reduction (800, 1600)


    outputs = Build_Convolution_Network_v2 (Inputs)

    outputs = InceptionModel1.Style2 (outputs)
    outputs = ReducntionModel1.Redunction_StyleA (outputs)

    outputs = InceptionModel2.Style2 (outputs)
    outputs = ReducntionModel2.Redunction_StyleA (outputs)

    outputs = InceptionModel3.Style2 (outputs)
    outputs = RedunctionModel3.Redunction_StyleA (outputs)
     

    print ("NULL")