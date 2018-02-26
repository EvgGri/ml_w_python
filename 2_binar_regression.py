
# Алгоритм обучения персептрона никогда не сходится на данных, которые не полностью линейно разделимы.
# Есть более мощные линейные классификаторы, которые сходятся к минимуму стоимости, даже, если данные линейно не разделимы.
# Это происходит из-за того, что веса постоянно обновляются (персептрон), а это происходит из-за того, что в каждой эпохе есть
# как минимум один не верно классифицированный образец.

# Для задачи линейной и бинарной классификации существует алгоритм логистической регресии.
# logit(p)=log(p/(1-p)) - функция логит, логарифм отношения шансов, функция логит принимает значения в диапазоне
# от 0 до 1 и преобразовывает их в значения по всему диапазону вещественного числа, которые можно использовать для
# выражения линейной связи между значениями признаков и логарифмами отношения шансов.
# p/(1-p) - функция отношения шансов в пользу отдельного события

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
