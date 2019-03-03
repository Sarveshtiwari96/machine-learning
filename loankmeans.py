import pandas as pd
df_loands=pd.read_csv('loands.csv')
print(df_loands)
col_names=df_loands.columns.values
print(col_names)
df1=df_loands[col_names[0:12]]
print(df1)

from sklearn.cluster import KMeans
lkmd_model = KMeans(2)
lkmd_model.fit(df1)

print(lkmd_model.labels_)
print(lkmd_model.cluster_centers_)
