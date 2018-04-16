# Алгоритмы последовательного отбора признаков

# Эти алгоритмы принадлежат к семейству жадных алгоритмов поиска, которые используя информацию из d-мерного пространства признаков,
# переходят к k-мерному пространству признаков, где d<k.
# Основная мотивация алгоритмов - отобрать наиболее релевантные признаки для решения задачи, чтобы увеличить вычислительную эффективность
# и уменьшить ошибку обобщения модели.

# Используя случайный лес, можно измерить важность признака как усредненное уменьшение неоднородности, вычисленное из всех деревьев в лесе,
# не делая никаких допущений по поводу наличия или отуствия линейной разделимости данных.

# Удобно то, что в библиотеке Scikit-Learn для леса решений уже реализовано аккумулирование важности признаков, и к ним можно обращаться
# через атрибут feature_importances_ после подгонки классификатора RandomForestClassifier


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

# =-=-=-=-=-=-=-=-=- Непосредственная реализация отбора признаков через RandomForestClassifier -=-=-=-=-=-=-=-=-

# Натренируем лес из 10 000 деревьев на нашем наборе данных и упорядочим 13 признаков по их соответствующим мерам важности.
# Отметим, что модели на основе деревьев решений в стандартизации и нормализации не нуждаются.
# Но мы проверим, изменится ли важность признаков для стандартизованного набора данных или нет.
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("{:3d})".format(f), "{:25.25}".format(feat_labels[indices[f]]), "{:10.6f}".format(importances[indices[f]]))

# Выведем график, в котором разные признаки из набора данных упорядочены по их относительной важности,
# отметим то, что важности признаков нормализованы, т.е. в сумме они дают единицу.
import matplotlib.pyplot as plt
plt.title('Важности признаков')
plt.bar(range(X_train.shape[1]), importances[indicies], color='lightblue', align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# =-=-=-=-=-=-=-=-=- Непосредственная реализация отбора признаков через RandomForestClassifier на стандартизованных данных -=-=-=-=-=-=-=-=-

# Натренируем лес из 10 000 деревьев на нашем наборе данных и упорядочим 13 признаков по их соответствующим мерам важности.
# Отметим, что модели на основе деревьев решений в стандартизации и нормализации не нуждаются.
# Но мы проверим, изменится ли важность признаков для стандартизованного набора данных или нет.
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train_std, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train_std.shape[1]):
    print("{:3d})".format(f), "{:25.25}".format(feat_labels[indices[f]]), "{:10.6f}".format(importances[indices[f]]))

# Выведем график, в котором разные признаки из набора данных упорядочены по их относительной важности,
# отметим то, что важности признаков нормализованы, т.е. в сумме они дают единицу.
import matplotlib.pyplot as plt
plt.title('Важности признаков')
plt.bar(range(X_train_std.shape[1]), importances[indicies], color='lightblue', align = 'center')
plt.xticks(range(X_train_std.shape[1]), feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train_std.shape[1]])
plt.tight_layout()
plt.show()
# Видим, что важность признаков стандартизованном наборе данных не изменилась.