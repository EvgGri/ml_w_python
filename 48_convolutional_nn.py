# =-=-=-=-=-=-=-=-=-=-=- Глубокое обучение с применением сверточных сетей

# В предыдущих главах мы обсуждали плотные сети, где каждый нейрон связан со всеми нейронами соседних слоев.
# Мы применили плотные сети к классификации рукописных цифр из набора данных MNIST. В этом контексте каждому пикселю
# входного изображения сопоставляется отдельный нейрон, так что всего получается 784 (28 × 28 пикселей) входных нейронов.
# Однако при такой стратегии игнорируется пространственная структура и связи внутри изображения.
# Так, следующий фрагмент кода преобразует растровые изображения всех цифр в плоский вектор, что приводит к потере информации о
# пространственной локализации.

from __future__ import print_function
import numpy as np
from keras.datasets import mnist

# Для воспроизводимости результатов
np.random.seed(1671)

# Данные: случайно перетасованы и разбиты на обучающий и тестовый набор
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train содержит 60000 изображений размера 28x28 --> преобразуем в массив 60000 x 784
# Во входном слое с каждым пикселем изображения ассоциирован один нейрон, т. е. всего получается 28 × 28 = 784 нейрона
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Сверточные сети задействуют пространственную информацию и потому хорошо подходят для классификации изображений.
# В них используется специальная архитектура, инспирированная данными, полученными в физиологических экспериментах со зрительной корой.
# Как уже отмечалось, наша зрительная система состоит из нескольких уровней коры, причем каждый последующий распознает все более крупные
# структуры в поступающей информации. Сначала мы видим отдельные пиксели, затем различаем в них простые геометрические формы,
# а затем – все более сложные элементы: предметы, лица, тела людей и животных и т. п.

# В этой главе мы рассмотрим следующие темы: 
# 1. глубокие сверточные нейронные сети
# 2. классификация изображений

# =-=-=-=-=-=-=-=-=-=-=- Глубокая сверточная нейронная сеть
# Глубокая сверточная нейронная сеть (ГСНС) состоит из большого числа слоев. Обычно в ней чередуются слои двух типов – сверточные и пулинговые.
# Глубина фильтра возрастает слева направо. На последних этапах обычно используется один или несколько полносвязных слоев.

# В основе сверточных сетей лежат три идеи:
# 1. локальное рецептивное поле; 
# 2. разделяемые веса;
# 3. пулинг.
#
# Рассмотрим их поочередно.


# =-=-=-=-=-=-=-=-=-=-=- Локальные рецептивные поля
# Для сохранения пространственной информации удобно представлять каждое изображение матрицей пикселей.
# Тогда для кодирования локальной структуры можно просто соединить подматрицу соседних входных нейронов с одним скрытым нейроном
# следующего слоя, который и представляет одно локальное рецептивное поле.
# Эта операция, называемая сверткой, и дала название типу сетей.

# Используя перекрывающиеся подматрицы, мы сможем закодировать больше информации.
# Предположим, к примеру, что размер каждой подматрицы равен 5 × 5 и что эти подматрицы используются для обработки изображений
# размера 28 × 28 из набора MNIST. Тогда мы сумеем создать 23 × 23 нейронов локального рецептивного поля в следующем скрытом слое.
# Действительно, подматрицу можно сдвинуть только на 23 позиции, а затем она уйдет за границу изображения.
# В Keras размер одной подматрицы, называемый длиной шага (stride length), является гиперпараметром, который можно настроить
# в процессе конструирования сетей.

# Определим карту признаков при переходе от одного слоя к другому. Конечно, можно завести несколько карт признаков, которые обучаются
# независимо. Например, для обработки изображений из набора MINST можно начать с 28 × 28 входных нейронов, а затем организовать k карт
# признаков размера 23 × 23 (с шагом 5 × 5) в следующем скрытом слое.

# =-=-=-=-=-=-=-=-=-=-=- Разделяемые веса и смещения
# Допустим, мы хотим отойти от строкового представления пикселей и получить возможность обнаруживать один и тот же признак независимо от того,
# в каком месте изображения он находится. На ум сразу приходит мысль воспользоваться общим набором весов и смещений для всех нейронов в скрытых слоях.
# Тогда каждый слой обучится распознавать множество позиционно­независимых признаков в изображении.

# Если входное изображение имеет размер (256, 256) с тремя каналами в порядке tf (TensorFlow), то его можно представить тензором (256, 256, 3).
# Отметим, что в режиме th (Theano) индекс канала глубины равен 1, а в режиме tf (TensoFlow) – 3.

from keras.layers.convolutional import Conv2D

# В Keras, чтобы добавить сверточный слой с 32 выходами и фильтром размера 3 × 3, мы пишем:
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))

# То же самое можно записать и по­другому:
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(256, 256, 3)))

# Это значит, что свертка с ядром 3 × 3 применяется к изображению размера 256 × 256 с тремя входными каналами (входными фильтрами), и в результате
# получается 32 выходных канала (выходных фильтра).

# =-=-=-=-=-=-=-=-=-=-=- Пулинговые слои
# Допустим, мы хотим агрегировать выход карты признаков. И в этом случае можно воспользоваться пространственной смежностью выходов, порожденных из одной карты признаков,
# и агрегировать значения подматрицы в одно выходное значение, которое дает сводное описание смысла, ассоциированного с данной физической областью.

# =-=-=-=-=-=-=-=-=-=-=- Max-пулинг
# Часто применяется max-пулинг, когда просто берется максимальный отклик в области. В Keras, чтобы определить слой max­ пулинга размера 2 × 2, мы пишем
from keras.layers.convolutional import MaxPooling2D
model.add(MaxPooling2D(pool_size = (2, 2)))

# =-=-=-=-=-=-=-=-=-=-=- Усредненный пулинг
# Другой вариант – усредненный пулинг, когда берется среднее арифметическое откликов в некоторой области.
# В Keras реализовано еще много пулинговых слоев, их полный перечень приведен на странице https://keras.io/layers/pooling/.
# Все операции пулинга сводятся к тому или иному способу агрегирования значений в заданной области.

# =-=-=-=-=-=-=-=-=-=-=- Основные выводы
# Мы изложили основные понятия сверточных сетей. В СНС операции свертки и пулинга применяются в одном направлении (время) для звуковых и текстовых данных,
# в двух направлениях (ширина и высота) для изображений и в трех направлениях (ширина, высота, время) для видео. В случае изображений перемещение фильтра по
# входной матрице порождает карту, дающую отклики фильтра для каждого положения в пространстве. Иначе говоря, сверточная сеть состоит из нескольких собранных
# в стопку фильтров, которые обучаются распознавать конкретные визуальные признаки независимо от того, в каком месте изображения они находятся.
# В начальных слоях сети признаки простые, а затем становятся все более сложными.