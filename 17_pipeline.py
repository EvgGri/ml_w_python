# Подгонка модели с произвольным числом шагов преобразований.
# Класс-конвейер Pipeline библиотеки Scikit-learn

# Данные о раке молочное железы в штате Висконсин
# Первые 2 столбца в наборе данных содержат уникальные идентификационные номера образцов и соотвествующих диагнозов
# M - malignant, злокачественная, B - benign доброкачественная, столбцы 3-32 содержат 30 вещественных признаков из изображений опухолей

import pandas as pd
import numpy as np
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
# df = pd.read_csv(url, header=None)

url = '/Users/grigorev-ee/Work/AnacondaProjects/My_projects/ml_w_python/data/cancer.base'
df = pd.read_csv(url)


names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean',
         'fractal_dimension_mean', 'radius_se', 'texture_se',
         'perimeter_se', 'area_se', 'smoothness_se',
         'compactness_se', 'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst',
         'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst',
         'fractal_dimension_worst']

df = df[names]

# Присвоим 30 признакоы массиву X библиотеки NumPy.
# При помощи LabelEncoder преобразуем метки классов из их исходного строкового представления (M и B) в целочисленное.
from sklearn.preprocessing import LabelEncoder
X = df.iloc[:,2:].values
y = df.iloc[:,1].values

print(y[:36])

le = LabelEncoder()
y=le.fit_transform(y)

# M-злокачественная опухоль (malignant), B-доброкачественная (benign). Выведем первые 36 элементов из набора данных.
print(y[:36])

# Выведем пример того, как работает кодировщик на тестовом наборе данных
le.transform(['M','M','B'])

# Прежде, чем в следующем подразделе мы построим наш первый модельный конвейер, рзделим выборку на обучающую и тренировочную.
# Оценка модели на ранее не встречавшихся данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Совмещение преобразователей и оценщиков в конвейере
print(X_train[:,1])

# Вместо раздельного выполнения шагов по подгонке и преобразованию тренировочного и тестового наборов данных мы можем расположить бъекты
# StandardScaler, PCA и LogisticRegression друг за другом в конвейере.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)
print('Верность на тестовом наборе: %.3f', pipe_lr.score(X_test, y_test))

# Использование K-блочной перекрестной проверки для оценки качества модели
from sklearn .model_selection import cross_validate
scores = cross_validate(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('Оценки перекрестно-проверочной верности: %s' % scores)
score_avg = np.mean(scores['test_score'])
score_std = np.std(scores['test_score'])
print('Перекрестно-проверочная верность: %.3f +/- %.3f' % (score_avg, score_std))
