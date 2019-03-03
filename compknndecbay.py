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
x_train ,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.7,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

lk_model = KNeighborsClassifier(n_neighbors=3)

a=lk_model.fit(x_train,y_train)
print(a)
model_predict1= lk_model.predict(x_test)
# print("line 25:",model_predict1)
print(y_train)
from sklearn.metrics import accuracy_score
f=accuracy_score(y_train,model_predict1)
print(f)



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

from sklearn.naive_bayes import GaussianNB
lb_model = GaussianNB()
b = lb_model.fit(x_train,y_train)
print(b)
model_predict2= lb_model.predict2(x_test)
print(model_predict2)
print(y_train)

from sklearn.metrics import accuracy_score52
g=accuracy_score(y_train,model_predict2)
print(g)



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
ld_model = DecisionTreeClassifier()
d = lb_model.fit(x_train,y_train)
print(d)
model_predict= lk_model.predict(x_train)
print(model_predict)
print(y_train)

from sklearn.metrics import accuracy_score
e=accuracy_score(y_train,model_predict)
print(e)


# print(e,"Decisiontree")
# print(f,"knn")
# print(g,"nb")


