# Зачастую придится работать с крупными наборами данных, которые могут превышать память используемого компьютера.
# Для этого рассмотрим обучение вне ядра (out of core learning), т.е. с использованием внешней памяти.
# Воспользуемся функцией partial_fit классификатора на основе стохастического градиентного спуска SGDClassifierself,
# чтобы передавать поток документов непосредственно из нашего локального диска и тренировать логистическую регрессионную
# модель с использованием небольших мини-пакетов документов.

# Определим функцию лексемизации tokenizer, которая очищает необработанные текстовые данные.

# Прочитаем базы отзывов о кинофильмах
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
df=pd.read_csv('./data/movie_data.csv')
stop = stopwords.words('english')

# В отзыве содержится лишняя информация, удалим все лишние за исключением символов-эмоций (эмограммы) вроде ':)'
# Для этого воспользуемся библиотекой регулярных выражений Python re
import re
def tokenizer(text):
    text=re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Определим генераторную функцию stream_docs, которая считывает и выдает по одному документу за раз
def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv) # пропускаем заголовок
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Первый документ должен вернуть кортеж, состоящий из текста отзыва и соответствующей метки класса
next(stream_docs(path='./data/movie_data.csv'))
# Определем функцию get_minibatch, которая принимает поток документов из функции stream_docs и возвращает отдельно взятое число докуметов,
# заданных в параметре size
def get_minibatch(doc_stream, size):
