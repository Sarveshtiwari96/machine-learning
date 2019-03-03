import pandas as pd
df_iris = pd.read_csv('iris.csv')
print(df_iris)
col_names = df_iris.columns.values
print(col_names)
df1=df_iris[col_names[0:4]]
print(df1)
df2=df_iris[col_names[-1]]
print(df2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.75,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

lk_model = KNeighborsClassifier(n_neighbors=3)

a=lk_model.fit(x_train,y_train)
print(a)
model_predict= lk_model.predict(x_train)
print(model_predict)
print(y_train)
from sklearn.metrics import accuracy_score
f=accuracy_score(y_train,model_predict)
print(f)




