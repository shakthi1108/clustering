import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv("C:/Users/shakgane/Downloads/Advertising.csv")

#plt.scatter(df['newspaper'], df['sales'])

scalar = MinMaxScaler()
scalar.fit(df[['sales']])
df['sales'] = scalar.transform(df['sales'].values.reshape(-1,1))

scalar.fit(df[['newspaper']])
df['newspaper'] = scalar.transform(df['newspaper'].values.reshape(-1,1))
#print(df)    

km = KMeans(n_clusters=4)

y_pred = km.fit_predict(df[['newspaper','sales']])
#print(y_pred)

df['cluster'] = y_pred

#print(km.cluster_centers_)
#
#print(df.head())

#df1=df[df.cluster==0]
#df2=df[df.cluster==1]
#df3=df[df.cluster==2]
#df4=df[df.cluster==3]
#
#plt.scatter(df1['newspaper'], df1['sales'], color = 'yellow')
#plt.scatter(df2['newspaper'], df2['sales'], color = 'r')
#plt.scatter(df3['newspaper'], df3['sales'], color = 'k')
#plt.scatter(df4['newspaper'], df4['sales'], color = 'pink')
#plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'blue', marker = '*', label = 'centroid')
#
#plt.xlabel('newspaper')
#plt.ylabel('sales')
#plt.legend()
#
#sse = []
#sse.append(km.inertia_)
#print(sse)

k_range = range(1,10)
sse = []

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['newspaper','sales']])
    sse.append(km.inertia_)
    
#print(sse)
plt.plot(k_range, sse)
plt.xlabel('k range')
plt.ylabel('mean square error')
plt.title('Error for different k values')

