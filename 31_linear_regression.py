# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()

 # Exploratory data analysis, EDA -  Разведочный анализ
 # Создадим матрицу точечных графиков:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()
# Вернуть стандартные стили
# sns.reset_orig()

# Корреляционная матрица - это квадратная матрица, которая содержит линеные коэффицLinearиенты корреляции Пирсона, которые измеряют линейную
# зависимость между парами признаков. Коэффициенты корреляции ограничены диапазоном [-1,1].
# Вопреки мнению, что во время тренировки линейной регрессионной модели необходимо, чтобы объясняющие, либо целевые переменные были
# распределены нормально, допущение о нормальности распределения является необходимым условияем для определенных статистических тестов.
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()
# Для того, чтобы выполнить подгонку линейной регрессионной модели, нас интересуют те признаки, которые имеют высокую корреляцию с нашей целевой
# переменной MEDV

# Реализация обычной регрессионной модели методом наименьших квадратов (OLS - Ordinary least squares)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

X=df[['RM']].values
y=df[['MEDV']].values

slr.fit(X,y)
print('Наклон: %.3f' % slr.coef_[0])
print('Пересечение: %.3f' % slr.intercept_)

def lin_regplot(X,y, model):
    plt.scatter(X,y,c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X,y,slr)

plt.xlabel('Среднее число комнат [RM]')
plt.ylabel('Цена в тыс. долл. [MEDV]')
plt.show()

# Подгонка стабильной регрессионной модели алгоритмом RANSAC
# Выбросы могут оказывать сильное воздействие на линейные модели. Небольшое подмножество в данных может оказать очень сильное влияние
# на результаты моделирования. Как альтернатива статистическим методам, позволяющим исключить выбросы, предлагается испольовать
# устойчивый метод регрессии с использованием алгоритма RANSAC (Random Sample Consensus, т.е. консенсус на подмножестве случайных образцов).
# Он выполняет подгонку регрессионной модели на подмножестве данных, так называемых не-выбросах (inliers), т.е. на хороших точках данных.

# Общая схема алгоритма:
# 1. Выбрать случайное число образцов в качестве не-выбросов и выполнить подгонку модели
# 2. Проверить все остальные точки данных на подогнанной модели и добавить те точки, которые попадают в пределы заданного аналитиком диапазона
# для не-выбросов
# 3. Выполнить повторную подгонку модели с использованием всех не-выбросов
# 4. Оценить ошибку подогнанной модели относительно не-выбросов
# 5. Завершить алгоритм, в случае если качество соответствует определенному заданному пользователю порогу либо если было достигнуто
# фиксированное число итераций

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X,y)
# max_trials - максиимальное число итераций
# min_samples - минимальное число случайно отобранных образцов
# residual_metric - параметр метрики остатков, лямбда функция рассчитывает абсолютные вертикальные расстояния между точками образцов и
# подогнанной линией
# residual_threshold - разрешаем включать в подмножество не-выбросов образцы с вертикальным расстоянием не больше 5 единиц

# После подгонки модели RANSAC получим не-выбросы и выбросы из подогнанной линейной регрессионой модели RANSAC и построим совместный график с
# линейной подгонкой
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Не-выбросы')
plt.scatter(X[outlier_mask], y[outlier_mask], c='red', marker='s', label='Выбросы')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Среднее число комнат [RM]')
plt.ylabel('Средняя цена в тыс. долл. [MEDV]')
plt.legend(loc='upper left')
plt.show()
# Если мы распечатаем наклон (угловой коэффициент) и точку пересечения модели, увидим, что линия линейной регрессии отличается от подгонки,
# которую мы получили в предыдущем разделе без модели RANSAC
print('Наклон: %.3f' % ransac.estimator_.coef_[0])
print('Пересечение: %.3f' % ransac.estimator_.intercept_)

# Оценка качества работы линейных регрессионных моделей
