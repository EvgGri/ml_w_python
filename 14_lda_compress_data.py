# linead disriminant analysis (LDA)
# Сжатие данных с учителем путем линейного дискриминантного анализа

# Линейный дискриминантный анализ, он же канонический, может использоваться в качестве метода для выделения признаков в целях увеличения
# эффективности и уменьшения степени переподгонки из-за проблемы проклятия размерности в нерегуляризованных моделях.

# Основная задача LDA - найти подпространство признаков, которое оптимизирует разделимость классов.
# И метод PCA и LDA являются методами линейного преобразования, которые могут использоваться для снижения числа размерностей
# в наборе данных, но PCA - алгоритм без учителя, LDA - алгоритм с учителем.

# Одно из допущений в LDA состоит в том, что данные нормально распределены. Кроме того, мы также допускаем, что классы имеют идентичные
# ковариационные матрицы и что признаки статистически независимы друг от друга. Однако, если даже эти допущения нарушаются, метод
# LDA может все еще работать достаточно хорошо.

# Основные шаги алгоритма:

# 1. Стандартизировать d-мерный набор данных
# 2. Для каждого из классов вычислить d-мерный вектор средних
# 3. Создать матрицу разброса между классами S_b и матрицу разброса внутри классов S_w
# 4. Вычислить собственные векторы и соответствующие собственные значения матрицы {(S_w)^-1} * S_b
# 5. Выбрать k собственных векторов, которые соответствуют k самым большим собственным значениям для построения d x k - матрицы преобразования W;
#    собственные векторы являются столбцами этой матрицы.
# 6. Спроецировать образцы на новое подпространство признаков при помощи матрицы преобразования W

# При использовании LDA мы делаем допущение, что признаки нормально распределены и независимы друг от друга. Кроме того, алгоритм LDA делает допущение,
# что ковариационные матрицы для индивидуальных классов идентичны. Однако, даже если есть нарушения данных допущений, алгоритм все еще хорошо работает.

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

# =-=-=-=-=-=-=-= Применять LDA только нормализации данных!
# Непосредственное применение метода LDA из пакета scikit-sklearn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# После этого посмотрим, как классификатор логистической регрессии обрабатывает более изкоразмерный тренровочный набор данных после преобразования LDA:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plot_decision_regions(X_train_lda, y_train, clf = lr, res=0.02)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# Понизив силу регуляризации, можно было бы сместить границы решения, в результате чего модели логистической регресси будут правильно классифицировать все
# образцы в тренировочном наборе данных.
# Посмотрим результаты применения LDA на тестовом наборе данных.
X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, clf = lr, res=0.02)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# Как видно на получившемся графике, классификатор на основе логистической регрессии в состоянии получить идеальную оценку верности классификации
# образцов на тестовом наборе данных, используя для этого всего лишь двумерное подпространство признаков вместо исходного набора из 13 признаков.
