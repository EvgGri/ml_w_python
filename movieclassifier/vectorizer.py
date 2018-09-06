# Нам не нужно консервировать хэширующий векторизатор HashingVectorizer, поскольку он не требует выполнения подгонки. Вместо этого можно
# создать новый сценарный файл Python, из которого можно импортировать векторизатор в наш текущий сеанс Python.
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
