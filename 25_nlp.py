# Анализ мнений (NLP - Natural Language Processing).

# Как использовать алгоритмы машинного обучения для классификации документов, основываясь на их направленности: отношении автора.
# Анализ мнений - популярная подотрасль более широкой области - обработки ествественного языка, предназначенная для анализа направленности
# документов (полярности документов).

# Набор данных киноотзывов состоит из 50 000 полярных отзывов о кинофильмах, помеченных, как положительные или отрицательные.
# В данном случае, положительный отзыв означает, что в IMDB фильм получил более 6 звезд от пользователей, а отрицательный, что фильм получил
# менее 5 звезд.


# Соберем все отдельные текстовые документы в единый файл.
# pbar - объект индикатора выполнения. После этого идет цикл итераций по каталогам train & test внутри основного каталога и прочли
# отдельные текстовые файлы из подкаталогов pos & neg, которые мы собрали и записали в DataFrame с меткой 1-положительные, 0- отрицательные.
import pyprind
import pandas as pd
import os

pbar=pyprind.ProgBar(50000)
labels={'pos':1, 'neg': 0}
df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos', 'neg'):
        path='./data/aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt=infile.read()
            df=df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

# Поскольку метки классов сейчас в исходном DataFrame отсортированы, перемешаем элементы, используя функцию permutation.
# Сохраним наш "зашафленный" DataFrame в единый файл, чтобы не терять время на сбор всех файлов в подкаталоге.
import numpy as np
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('./data/movie_data.csv', index=False)

df=pd.read_csv('./data/movie_data.csv')
df.head(3)
