import pandas as pd
df_diabetes = pd.read_csv('diabetes.csv')
# print(df_diabetes)

col_names= df_diabetes.columns.values
# print(col_names)
df1=df_diabetes[col_names[0:8]]
# print(df1)
df2=df_diabetes[col_names[-1]]
# print(df2)

for i in range(0,len(df_diabetes.columns)-1):
    abc = df_diabetes.iloc[:,i].mean()
    for m in range(0,df_diabetes.iloc[:,i].count()-1):

        if(df_diabetes.iloc[:,i][m] == 0):
            df_diabetes.iloc[:,i][m] = df_diabetes.iloc[:, i][m].replace(0, abc)
            print(df_diabetes.iloc[:,i][m])




# df_diabetes.to_csv('diabetes1.csv',index=False)
# df3 = df.fillna(df1.mean())
# print(df3)


# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(df1,df2,train_size=0.7,random_state=1)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
#
# from sklearn.ensemble import RandomForestClassifier
# lb_model = RandomForestClassifier()
# b = lb_model.fit(x_train,y_train)
# print(b)
# model_predict= lb_model.predict(x_train)
# print(model_predict)
# print(y_train)
#
# from sklearn.metrics import accuracy_score
# d=accuracy_score(y_train,model_predict)
# print(d)



