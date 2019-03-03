import pandas as pd
df_iris = pd.read_csv('iris.csv')
print(df_iris)
col_names=df_iris.columns.values
print(col_names)
df1=df_iris[col_names[0:4]]
print(df1)
#df2=df_iris[col_names[-1]]
#print(df2)


from sklearn.cluster import KMeans
lkm_model = KMeans(2)
lkm_model.fit(df1)

print(lkm_model.labels_)
print(lkm_model.cluster_centers_)
