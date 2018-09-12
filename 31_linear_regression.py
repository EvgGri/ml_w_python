# Набор данных, содержащий информацию о зданиях в пригороде Бостона.
import pandas as pd
url='./data/BostonHousing.xls'

df=pd.read_excel(url, index_col=None)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PT','B','LSTAT','MV']


df.head()
