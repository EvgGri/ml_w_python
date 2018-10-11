# -=-=-=-=-=-=-=- Улучшение простой сети в Keras посредством добавления скрытых слоев
# Первое улучшение – включить в сеть дополнительные слои.
# После входного слоя поместим первый плотный слой с N_HIDDEN нейронами и функцией активации relu.
# Этот слой называется скрытым, потому что он напрямую не соединен ни с входом, ни с выходом.
# После первого скрытого слоя добавим еще один, также содержащий N_HIDDEN нейронов, а уже за ним будет расположен выходной слой с 10 нейронами,
# которые возбуждаются, если распознана соответствующая цифра.

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Для воспроизводимости результатов
np.random.seed(1671)

# Сеть и ее обучение
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
# Количество результатов = числу цифр
NB_CLASSES = 10
