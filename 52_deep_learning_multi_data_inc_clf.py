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
# Аргумент rotation_range – это диапазон углов в градусах (0–180), на которые можно поворачивать изображения (случайным образом).
# Аргументы width_shift и height_shift – диапазоны случайного параллельного переноса по горизонтали и по вертикали.
# Аргумент zoom_range задает диапазон случайного масштабирования изображений, horizontal_ ip говорит, что случайным образом
# отобранную половину изображений нужно отразить относительно вертикальной оси, а  fill_mode определяет стратегию вычисления новых
# пикселей, образующихся при повороте или параллельном переносе.
print("Augmenting training set images.")
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2,
                             horizontal_flip=True,  fill_mode='nearest')

xtas, ytas = [], []
for i in range(X_train.shape[0]):
    num_aug = 0
    x = X_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in datagen.flow(x, batch_size=1,
        save_to_dir='data/images/', save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1
