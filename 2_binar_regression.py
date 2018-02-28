
# Алгоритм обучения персептрона никогда не сходится на данных, которые не полностью линейно разделимы.
# Есть более мощные линейные классификаторы, которые сходятся к минимуму стоимости, даже, если данные линейно не разделимы.
# Это происходит из-за того, что веса постоянно обновляются (персептрон), а это происходит из-за того, что в каждой эпохе есть
# как минимум один не верно классифицированный образец.

# Для задачи линейной и бинарной классификации существует алгоритм логистической регресии.
# logit(p)=log(p/(1-p)) - функция логит, логарифм отношения шансов, функция логит принимает значения в диапазоне
# от 0 до 1 и преобразовывает их в значения по всему диапазону вещественного числа, которые можно использовать для
# выражения линейной связи между значениями признаков и логарифмами отношения шансов.
# p/(1-p) - функция отношения шансов в пользу отдельного события.
# Далее нас на самом деле интересует вероятность того, что определенный образец принадлежит отдельно взятому классу,
# т.е.обратная функция логит регрессии. Ее также называют логистической функцией или сигмойдой.

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# Определяем диапазон и шаг приращения
z = np.arange(-7, 7 , 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0 , color = 'k')
plt.axhspan(0.0 , 1.0 , facecolor='1.0' , alpha=1.0, ls='dotted')
plt.axhline(y=0.5 , ls='dotted', color='k')
plt.yticks([0.0, 0.5 , 1.0])
plt.ylim(-0.1 , 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

# В задаче логистической регрессии функция активации становится сигмоидальной функцией.
# Выход из сигмоидальной функции затем интерпретируется как вероятность принадлежности отдельно взятого образца классу 1 при
# наличии его признаков x, параметризованных весами w.

# Например, если для отдельно взятого образца цветков мы вычисляем phi(z)=0.8, то это означает, то шанс, что этот образец
# является цветком ирис разноцветный, составляет 80%.
# Аналогичным образом вероятность, что этот цветок ириса является ирисом щетинистым, P(y=0|x,w) = 1-P(y=1|x,w)=0.2%

# Предсказанную вероятность затем можно просто просто конвертировать кванизатором (единичной ступенчатой функцией) в бинарный результат.
# Если phi(z)>=0.5 тогда y=1, в противном случае =0

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Алгоритм логистической регрессии в sklearn

# Подготовка данных
# Импорт основных библиотек
from sklearn import datasets
import numpy as np

# прогружаем стандартную библиотеку
iris = datasets.load_iris()

# длина и ширина лепестков цветка ириса
X = iris.data[:,[2,3]]
# метки классов, которые присутствуют
y = iris.target
# все закодировано в числовом формате для производительности
print(np.unique(y))

# оценка модели на ранее не встречавшихся данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# стандартизация признаков из модуля preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# вычисление параметров распределения для train данных (стандартное отклонение и мат. ожидание)
# для каждого набора признаков. После вызова trasform мы стандартизируем тестовые и тренинговые данные.
# Для стандартизации тестового набора мы используем теже самые параметры, вычисленные для train набора.
# Поэтому значения в тренировочном и тестовом наборе сопоставимы.
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Непосредственно анализ данных

# Параметры для построения логистической регресии
# C : float, optional (default=1.0)
# Inverse of regularization strength; must be a positive float. Like in support vector machines,
# smaller values specify stronger regularization.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_combined_std, y_combined, clf = lr, res=0.02)
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc = 'upper left')
plt.show()

# Мы можем предсказывать вероятность принадлежности образцов классам при помощи метода predict_proba.
# Например, можно предсказывать вероятности первого образца ириса щетинистого:

lr.predict_proba(X_test_std)[0]

# Изменяем формат представления вывода для "обычных" чисел
np.set_printoptions(precision=3)
print(lr.predict_proba(X_test_std)[0])
# Приведенный ниже массив говорит о том, что модель предсказывает с вероятностью 94%  принадлежность
# образца к классу ириса виргинского, 6% к классу ирис разноцветный
['{:.2f}'.format(i) for i in lr.predict_proba(X_test_std)[0]]


# Определяем формат вывода массива данных
# def ndprint(a, format_string ='{0:.2f}'):
#     print [format_string.format(v,i) for i,v in enumerate(a)]
# ndprint(x)

# Решение проблемы переобучения
# Часто встречается ситуация, когда модель хорошо работает на тренировочных данных, но плохо работает
# на ранее не встречавшихся данных.
