# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()

# Смоделируем связь между ценами на жилье и LSTAT (процентом населения с более низким статусом) с использованием полинома второй степени
# и третье степени и сравним его с линейной подгонкой
from sklearn.preprocessing import PolynomialFeatures
X=df[['LSTAT']].values
y=df[['MEDV']].values

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# Создаем полиномиальные признаки
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Линейная подгонка
import numpy as np
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
