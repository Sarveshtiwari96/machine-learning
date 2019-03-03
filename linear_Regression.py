import pandas as pd
df_boston = pd.read_excel('boston.xls')
print(df_boston)

col_names = df_boston.columns.values
print('col_names')
df1 = df_boston[col_names[0:13]]
print(df1)
df2 = df_boston[col_names[-1]]
print(df2)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.7,random_state=1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
r = lr_model.fit(x_train,y_train)
print(r)
print(r.coef_)
print(r.intercept_)
model_predict = lr_model.predict(x_test)
print(model_predict)
print(y_test)