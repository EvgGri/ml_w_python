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
