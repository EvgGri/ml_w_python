# Перцептрон – это простой алгоритм, который получает входной вектор x, содержащий n значений (x1, x2, ..., xn), которые часто
# называются входными признаками, или просто признаками, и выдает на выходе 1 (да) или 0 (нет).


# Здесь w – вектор весов, wx – скалярное произведение, а b – смещение.

# Исходным строительным блоком Keras является модель, а простейшая модель называется последовательной.
# В Keras последовательная модель представляет собой линейный конвейер (стек) слоев нейронной сети.
# В следующем фрагменте определен один слой с 12 нейронами, который ожидает получить 8 входных переменных (признаков).

from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()

# На этапе инициализации каждому нейрону можно назначить вес. Keras предлагает несколько вариантов:
# random_uniform: веса инициализируются равномерно распределенными случайными значениями из диапазона (–0.05, 0.05)
# random_normal: веса инициализируются нормально рас- пределенными случайными значениями со средним 0 и стандартным отклонением 0.05
# zero: все веса инициализируются нулями
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))

# Воспользуемся библиотекой Keras, чтобы определить сеть, распознающую рукописные цифры из набора MNIST.
# Начнем с очень простой нейросети и постепенно будем ее улучшать.
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671) # для воспроизводимости результатов

# сеть и ее обучение
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # количество результатов = числу цифр
OPTIMIZER = SGD() # СГС-оптимизатор, обсуждается ниже в этой главе N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # какая часть обучающего набора зарезервирована для контроля

# данные: случайно перетасованы и разбиты на обучающий и тестовый набор
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train содержит 60000 изображений размера 28x28 --> преобразуем в массив 60000 x 784
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
