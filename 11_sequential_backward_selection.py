# Алгоритмы последовательного отбора признаков

# Эти алгоритмы принадлежат к семейству жадных алгоритмов поиска, которые используя информацию из d-мерного пространства признаков,
# переходят к k-мерному пространству признаков, где d<k.
# Основная мотивация алгоритмов - отобрать наиболее релевантные признаки для решения задачи, чтобы увеличить вычислительную эффективность
# и уменьшить ошибку обобщения модели.

# SBS последовательно удаляет признаки из полнопризнакового пространства. Нужно определить функционал, который мы хотим минимизировать.
# Эта функция - просто разница в качестве классификатора до удаления признака и после удаления. На каждом шаге мы удаляем тот признак,
# который приводит к наименьшей потере качества классификатора.

# =-=-=-=-=-=-=-=-=- Загрузка данных -=-=-=-=-=-=-=-=-

# Разбивка на тренировочное и тестовое подмножество

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()
data.target[[10, 80, 140]]
list(data.target_names)


df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']
df_wine.head()
df_wine['Метка класса'] = pd.DataFrame(data.target)
df_wine.describe()
# Размер data frame'a
df_wine.shape

# Разбиение на train и test выборки
X, y = df_wine.iloc[:,0:13].values, df_wine['Метка класса'].values

# Альтернативная задача массива данных
# X = df_wine.ix[:,['Алкоголь','Яблочная кислота','Зола',
#                  'Щелочность золы','Магний','Всего фенола','Флаваноиды',
#                  'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
#                  'Оттенок','OD280 / OD315 разбавленных вин','Пролин']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Масштабирование признаков(нормирование, приведение к диапазону 0-1)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# С практической точки зрения лучше использовать стандартизацию признаков (приведеение к нормальному распределению с единичной дисперсией).
# Причина в том, что многие линейные модели, такие, как логистическая регрессия и метод опорных векторов инициализируют веса нулями или
# близкими к 0 значениями, но вид нормального распределения упрощает извлечение весов.

# Кроме того, стандартизация содержит полезную информацию о выбросах и делает алгоритм менее чувствительным к выбросам, в отличие от
# минимаксного масштабирования, которое шкалирует данные в ограниченном диапазоне значений.

# стандартизация признаков из модуля preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# вычисление параметров распределения для train данных (стандартное отклонение и мат. ожидание)
# для каждого набора признаков. После вызова trasform мы стандартизируем тестовые и тренинговые данные.
# Для стандартизации тестового набора мы используем теже самые параметры, вычисленные для train набора.
# Поэтому значения в тренировочном и тестовом наборе сопоставимы.
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# =-=-=-=-=-=-=-=-=- Непосредственная реализация SBS -=-=-=-=-=-=-=-=-

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split # cross validation
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features,
                 scorring=accuracy_score,
                 test_size=0.25, random_state=1):
                 self.scoring=scoring
                 self.estimator=clone(estimator)
                 self.k_features=k_features
                 self.test_size=test_size
                 self.random_state=random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # Стартовая размерность пространства признаков
    dim = X_train.shape[1]

    # Индексы столбцов окончательного подмножества
    self.indicies_ = tuple(range(dim))
    
