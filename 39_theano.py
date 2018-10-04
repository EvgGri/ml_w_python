import theano
from theano import tensor as T

# Инициализация
x1=T.scalar()
w1=T.scalar()
w0=T.scalar()
z1=w1*x1+w0

# Компиляция
net_input=theano.function(inputs=[w1, x1, w0], outputs=z1)

# Исполнить
print('Чистый вход: %.2f' % net_input(2.0, 1.0, 0.5))

# Настройка для вычисления не на процессоре, а на gpu
print(theano.config.floatX)
theano.config.floatX = 'float32'
print(theano.config.floatX)

# Bash - команда
# export THEANO_FLAGS=floatX=float32

# Применить формат только к конкретному сценарию
# THEANO_ FLAGS=floatX=float32 python your_script.py

# Настройка переключения между CPU и GPU
print(theano.config.device)

# Код на cpu из Bash
# THEANO_FLAGS=device=cpu,floatX=float64 python your_script.py

# Код на gpu из Bash
# THEANO_FLAGS=device=gpu,floatX=float32 python your_script.py

# Для постоянного использования на gpu и 32-битную систему, в домашнем каталоге создать файл .theanorc и записать туда:
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n" >> ~/.theanorc

# Если не MacOS или Linux, то можно вручную создать файл
[global]
floatX=float32
device=gpu

# -=-=-=-=-=-=-=- Работа с матричными структурами
import numpy as np

# инициализировать
# если Theano используется в 64-разрядном режиме, то вам нужно использовать dmatrix вместо fmatrix

x=T.fmatrix(name='x')
x_sum=T.sum(x, axis=0)

# скомпилировать
calc_sum=theano.function(inputs=[x], outputs=x_sum)

# выполнить(сначала Python)
ary=[[1,2,3],[1,2,3]]
print('Сумма столбца:', calc_sum(ary))


# выполнить(массив NumPy)
ary=np.array([[1,2,3],[1,2,3]], dtype=theano.config.floatX)
print('Сумма столбца:', calc_sum(ary))

# Работая с библиотекой Theano, нам нужно выполнить всего три основных шага: определить переменную, скомпилировать код и выполнить его.
# Предыдущий пример показывает, что Theano может работать, как с типами Python, так и с типами NumPy