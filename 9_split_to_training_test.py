# Разбивка на тренировочное и тестовое подмножество

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
data = load_wine()
data.target[[10, 80, 140]]
list(data.target_names)


df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']
df_wine.head()
df_wine['Метка класса'] = pd.DataFrame(data.target)
df_wine.describe()
