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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Логистическая регрессия', 'Дерево решений', 'К ближайших соседей']
print('10-блочная перекрестная проверка:\n')

for clf, label in zip([pipe1,clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
    print("ROC/AUC: {:.2f} (+/- {:.2f}) {:5}".format(scores.mean(), scores.std(), label))

print('* ROC AUC = площадь под ROC-кривой')

# -=-=-=-=-=-=-=-=-=-=-=-=- Реализация класса -=-=-=-=-=-=-=-=-
# Реализация классификатора на основе мажоритарного голосования
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """Ансамблевый классификатор на основе мажоритарного голосования
    Параметры:
    classifiers - классификаторы ансамбля, форма = [n_classifiers]
    vote - str, ['classlabel', 'classprobability'], если метка classlabel, то берется argmax меток классов, если classprobability, то
    для прогнозирования метки используется argmax сумму вероятностей
    weighta - веса, важность классификаторов, по умолчанию None
    """

def __init__(self, classifiers, vote='classlabel', weights = None):
    self.classifiers = classifiers
    self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
    self.vote = vote
    self.weights = weights

def fit(self, X, y):
    """Выполнить подгонку классификаторов
    Х - разреженная матрица с тренировочными образцами [n_samples, n_features]
    y - вектор целевых меток классов [n_samples]

    Возвращает объект self
    """
    # Использовать объект LabelEncoder, чтобы гарантировать, что
    # метки классов начинаются с 0, что важно для
    # вызова np.argmax в self.predict

    self.lablenc_ = LabelEncoder()
    self.lablenc_.fit(y)
    self.classes_ = self.lablenc_.classes_
    self.classifiers_ = []
    for clf in self.classifiers:
        fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
        self.classifiers_.append(fitted_clf)

    return self

def predict(self, X):
    """Спрогнозировать метки классов для Х
    
    """

mv_clf = MajorityVoteClassifier(classifiers=[pipe1,clf2, pipe3])
