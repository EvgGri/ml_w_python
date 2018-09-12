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
