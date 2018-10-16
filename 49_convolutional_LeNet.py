# =-=-=-=-=-=-=-=-=-=-=- Пример ГСНС – LeNet

# Ян Лекун (Yann le Cun) предложил (см. статью Y. LeCun, Y. Bengio «Convolutional Networks for Images, Speech, and Time­Series», Brain
# Theory Neural Networks, vol. 3361, 1995) семейство сверточных се- тей, получившее название LeNet, обученных распознаванию рукописных
# цифр из набора MNIST и устойчивых к простым геометрическим преобразованиям и искажению.
# Основная идея состоит в наличии чередующихся слоев, реализующих операции свертки и max­пулинга. Операции свертки основаны на тщательно
# подобранных локальных рецептивных полях с весами, разделяемыми между несколькими картами признаков.
# Последние слои полносвязные – как в традиционном МСП со скрытыми слоями и функцией активации softmax в выходном слое.

# Для определения сети LeNet в Keras используется модуль двумерной сверточной сети:
from tensorflow import keras

from keras.layers.convolutional import Conv2D
filters=2
kernel_size=3
# keras.layers.convolutional.Conv2D( lters, kernel_size, padding='valid')
keras.layers.Conv2D(filters, kernel_size, padding='valid')

# Здесь  filters – число сверточных ядер (например, размерность выхода), kernel_size – одно целое число или кортеж (либо список)
# из двух целых чисел, задающих ширину и высоту двумерного окна свертки (если указано одно число, то ширина и высота одинаковы),
# а padding='same' означает, что используется дополнение. Существует два режима: padding='valid' означает, что свертка вычисляется
# только там, где фильтр целиком помещается в области входа, поэтому выход оказывается меньше входа, а padding='same' – что размер
# выхода такой же (same), как размер входа, для чего входная область дополняется нулями по краям.

from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose

# keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
# Здесь pool_size=(2, 2) – кортеж из двух целых чисел, определяющих коэффициенты уменьшения изображения по вертикали и по горизонтали.
# Таким образом, (2, 2) означает, что изображение уменьшается вдвое в обоих направлениях.
# Наконец, параметр strides=(2, 2) определяет шаг обработки.
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

# Затем определяется сеть LeNet
#de ne the ConvNet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # CONV => RELU => POOL
