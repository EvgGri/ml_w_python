# -=-=-=-=-=-=-=-=-=- Обработка нелинейных связей при помощи случайных лесов
# Случайный лес - ансамбль из двух и более деревьев решений, это сумма кусочно-линейных функций.
# Другими словами, мы подразделяем входное пространство на на области меньшего размера, которые становятся более управляемыми.

# Нам не требуется преобразование признаков при использовании алгоритма дерева решений, если мы имеем дело с нелинейностью.
# Мы выращиваем дерево решений путем итеративного расщепления, пока не будут выполнены необходимые критерии остановки.
# В задаче классификации мы максимизировали прирост информации (Information Gain, IG), а энтропию определили, как меру неоднородности.
# Мы хотим найти расщепление признака, которое сокращает неоднородности в дочерних узлах.

# При использовании дерева решений для регрессии мы заменим энтропию на MSE.
# В контексте регрессии на основе дерева решений MSE часто упоминается, как внутриузловая дисперсия.

# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()


from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()

# Реализация обычной регрессионной модели методом наименьших квадратов (OLS - Ordinary least squares)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()


slr.fit(X,y)
print('Наклон: %.3f' % slr.coef_[0])
print('Пересечение: %.3f' % slr.intercept_)

def lin_regplot(X,y, model):
    plt.scatter(X,y,c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

import matplotlib.pyplot as plt
lin_regplot(X,y,slr)

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('log(% населения с более низким статусом [LSTAT])')
plt.ylabel('$\sqrt{Цена в тыс. долл. [MEDV]}$')
plt.show()
