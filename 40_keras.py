# -=-=-=-=-=-=-=-=- Загрузка данных MNIST

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    print('Загрузить данные Mnist из пути Path\n')
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)

    return images, labels

X_train , y_train = load_mnist('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/mnist/', kind = 'train')

print('Тренировка - строки: %d, столбцы: %d' % (X_train.shape[0], X_train.shape[1]))

X_test , y_test = load_mnist('/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/mnist/', kind = 't10k')
print('Тестирование - строки: %d, столбцы: %d' % (X_test.shape[0], X_test.shape[1]))

# -=-=-=-=-=-=- Реализация многослойного персептрона на Keras

# Приведем массив изображений MNIST в 32-разрядный формат:
import theano
theano.config.floatX='float32'

X_train=X_train.astype(theano.config.floatX)
X_test=X_test.astype(theano.config.floatX)

# Далее необходимо преобразовать метки классов (целые числа от 0 до 9) в формат прямого кода
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.utils import np_utils
print('Первые 3 метки:', y_train[:3])

y_train_ohe=np_utils.to_categorical(y_train)
print('\nПервые 3 метки (прямой код):\n', y_train_ohe[:3])
