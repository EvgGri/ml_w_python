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

dversarial_model = AdversarialModel(base_model=M, player_params=[generator.trainable_weights, discriminator.trainable_weights], player_names=["generator", "discriminator"])

# Если генератор G и дискриминатор D основаны на разных моделях, то можно воспользоваться такой функцией:
