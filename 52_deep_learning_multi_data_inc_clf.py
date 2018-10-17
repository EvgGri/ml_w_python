# =-=-=-=-=-=-=-=-=-=-=-=- Повышение качества распознавания набора CIFAR-10 путем пополнения данных

# Еще один способ повысить качество – сгенерировать дополнительные обучающие изображения.
# Идея состоит в том, чтобы взять стандартный набор данных CIFAR и пополнить его, подвергнув изображения различным преобразованиям:
# вращению, параллельному переносу, масштабированию, отражению относительно горизонтальной и вертикальной оси, перестановке каналов и т. д.
# Приведем соответствующий код:

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np

NUM_TO_AUGMENT=5
# загрузить набор данных
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# пополнение
print("Augmenting training set images.")
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2,
                             horizontal_flip=True,  fill_mode='nearest')
