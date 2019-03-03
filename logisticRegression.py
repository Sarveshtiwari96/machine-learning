import pandas as pd
df_iris = pd.read_csv('iris.csv')
print(df_iris)
col_names=df_iris.columns.values
print('col_names')
df1=df_iris[col_names[0:4]]
print(df1)
df2=df_iris[col_names[-1]]
print(df2)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.7)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
a=lg_model.fit(x_train,y_train)
print(a)
print(a.coef_)
print(a.intercept_)
model_predict= lg_model.predict(x_test)
print(model_predict)
print(y_test)
from sklearn.metrics import accuracy_score
t=accuracy_score(y_test,model_predict)
print(t)