import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn import linear_model

df=pd.read_csv('/Users/khadeejah/Desktop/datasets-9401-13260-Salary_Data.csv')
df.head()
X = df.drop('Salary',axis='columns')
X.head(1)

y= df.Salary
y.head(1)

reg = linear_model.LinearRegression()
reg.fit(X,y)

reg.predict([[3]])

#savin model
import pickle
pickle.dump(reg,open('model.pkl','wb'))

model= pickle.load(open('model.pkl','rb'))
model.predict([[4]])

print(model.predict([[1.8]]))