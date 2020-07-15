import numpy as np # linear algebra
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('/Users/khadeejah/desktop/datasets_loan.csv')

df.isnull().sum()
df1=df.dropna()
df2 = df1.replace(to_replace = ['Yes','No'],value = ['1','0'])
df2 = df1.replace(to_replace = ['Y','N'],value = ['1','0'])

X = df2[['ApplicantIncome','CoapplicantIncome','LoanAmount','Self_Employed','Loan_Amount_Term']]
y=df2.Loan_Status
X = X.replace(to_replace = ['Yes','No'],value = ['1','0'])
y=df2.Loan_Status


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
Classifier=RandomForestClassifier()
Classifier.fit(X_train,y_train)

y_pred = Classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)

pickle_out=open('classifier.pkl','wb')
pickle.dump(Classifier,pickle_out)
pickle_out.close()
Classifier.predict([[4000,6000,120,1,120]])
