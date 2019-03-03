import pandas as pd
df_iris = pd.read_csv('iris.csv')
print(df_iris)

col_names= df_iris.columns.values
print(col_names)
df1=df_iris[col_names[0:4]]
print(df1)
df2=df_iris[col_names[-1]]
print(df2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.7,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.tree import DecisionTreeClassifier
lb_model = DecisionTreeClassifier()
b = lb_model.fit(x_train,y_train)
print(b)
model_predict= lb_model.predict(x_train)
print(model_predict)
print(y_train)

from sklearn.metrics import accuracy_score
d=accuracy_score(y_train,model_predict)
print(d)



