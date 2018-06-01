# Подгонка модели с произвольным числом шагов преобразований.
# Класс-конвейер Pipeline библиотеки Scikit-learn

# Данные о раке молочное железы в штате Висконсин
# Первые 2 столбца в наборе данных содержат уникальные идентификационные номера образцов и соотвествующих диагнозов
# M - malignant, злокачественная, B - benign доброкачественная, столбцы 3-32 содержат 30 вещественных признаков из изображений опухолей

import pandas as pd

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
