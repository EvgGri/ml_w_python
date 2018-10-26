# -=-=-=-=-=-=-= Применение Keras adversarial для создания ПСС, подделывающей MNIST

# Keras adversarial (https://github.com/bstriner/keras-adversarial) – написанный на Python пакет с открытым исходным кодом, предназначенный для построения ПСС.
# Его автор – Бен Страйнер (Ben Striner) (https://github.com/bstriner и https://github.com/bstriner/ keras-adversarial/blob/master/LICENSE.txt).
# Поскольку версия Keras 2.0 появилась совсем недавно, я рекомендую скачать самую последнюю версию этого пакета:

# git clone --depth=50 --branch=master https://github.com/bstriner/keras-adversarial.git

# И установить его:
# python setup.py install

# from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras import backend as K

# Если генератор G и дискриминатор D основаны на одной и той же модели M, то их можно объединить в состязательную модель; она получает тот же вход, что и M,
# но цели и показатели качества различны для G и D. В библиотеке определена следующая функция создания модели:

dversarial_model = AdversarialModel(base_model=M, player_params=[generator.trainable_weights, discriminator.trainable_weights],
                                    player_names=["generator", "discriminator"])

# Если генератор G и дискриминатор D основаны на разных моделях, то можно воспользоваться такой функцией:
adversarial_model = AdversarialModel(player_models=[gan_g, gan_d], player_params=[generator.trainable_weights, discriminator.trainable_weights],
                                     player_names=["generator", "discriminator"])

# -=-=-=-=-=-=-=-=-= Рассмотрим пример вычислений для MNIST:

import matplotlib as mpl

# Эта строка позволяет использовать mpl без определения DISPLAY
mpl.use('Agg')

# Ниже рассматривается открытый исходный код
# (https://github. com/bstriner/keras-adversarial/blob/master/examples/example_gan_convolutional.py).
# В нем используется синтаксис Keras 1.x, но код работаети с Keras 2.x благодаря набору вспомогательных функций в файле legacy.py.
# Содержимое файла legacy.py приведено в приложении, а также по адресу
# https://github.com/bstriner/keras-adversarial/blob/ master/keras_adversarial/legacy.py

# Сначала импортируется ряд модулей. Мы уже встречались со всеми, кроме LeakyReLU, специальной версии ReLU, которая допускает малый
# градиент, когда нейрон не активен. Экспериментально показано, что в ряде случаев функция LeakyReLU может улучшить качество ПСС
# (см. B. Xu, N. Wang, T. Chen, M. Li «Empirical Evaluation of Recti ed Activations in Convolutional Network», arXiv:1505.00853, 2014).
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.datasets import mnist

import pandas as pd
import numpy as np

# Затем импортируются специальные модули для ПСС
import keras.backend as K
from keras_adversarial.legacy import Dense, BatchNormalization, Convolution2D
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix
