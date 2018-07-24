# Bagging = bootstrap aggregating, агрегирование бутстрап-выборок
# Вместо того, чтобы для подгонки ансамблевых классификаторов использовать один и тот же тренировочный набор данных,
# мы используем бутстрап-выборки (случайные образцы с возвратом).
# Каждая бутстрап-выборка затем используется для подгонки классификатора C_jself.

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

# Выберем только классы 2 и 3
df_wine = df_wine[df_wine['Метка класса'] != 1]

df_wine.describe()
# Размер data frame'a
df_wine.shape

# Разбиение на train и test выборки
X, y = df_wine[['Алкоголь', 'Оттенок']].values, df_wine['Метка класса'].values

# Приведем метки классов в двоичный формат
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# В sikit-learn алгоритм уже реализован алгоритм бэггинг-классификатора BaggingClassifier. В данном примере в качестве базового
# классификатора мы будем использовать  неподрезанное дерево решений и создадим ансамбль из 500 деревьев решений, подогнанных на
# разных бутстрап-выборках из тренировочного набора.

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples = 1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Верность дерева решений на тренировочном/тестовом наборах %.3f/%.3f' % (tree_train, tree_test))


# Масштабирование признаков(нормирование, приведение к диапазону 0-1)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
