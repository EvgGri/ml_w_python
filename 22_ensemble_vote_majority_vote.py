from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:,[1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Натренируем три разных классификатора - классификатор на основе логистической регрессии, дерева решений и к-ближайших соседей и посмотрим
# на их индивидуальные качественные характеристики путем 10-блочной перекрестной проверки на тренировочном наборе данных, перед тем, как
# объединить их в ансамблевый классификатор.
