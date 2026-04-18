from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("wine-clustering.csv")
df.head()

plt.scatter(df['Color_Intensity'], df['OD280'])
plt.xlabel('Color_Intensity')
plt.ylabel('OD280')
plt.title('Color_Intensity vs OD280')
plt.show()

plt.scatter(df['Flavanoids'], df['Total_Phenols'])
plt.xlabel('Flavanoids')
plt.ylabel('Total_Phenols')
plt.title('Flavanoids vs Total_Phenols')
plt.show()

plt.scatter(df.Alcohol,df['Alcohol'])
plt.xlabel('Alcohol')
plt.ylabel('Malic_Acid')

plt.scatter(df.Alcohol,df['Alcohol'])
plt.xlabel('Alcohol')
plt.ylabel('Proline')

plt.scatter(df['Hue'], df['Color_Intensity'])
plt.xlabel('Hue')
plt.ylabel('Color_Intensity')
plt.title('Hue vs Color_Intensity')
plt.show()

plt.scatter(df['Proanthocyanins'], df['Proline'])
plt.xlabel('Proanthocyanins')
plt.ylabel('Proline')
plt.title('Proanthocyanins vs Proline')
plt.show()

plt.scatter(df['Magnesium'], df['Ash_Alcanity'])
plt.xlabel('Magnesium')
plt.ylabel('Ash_Alcanity')
plt.title('Magnesium vs Ash_Alcanity')
plt.show()

plt.scatter(df['Flavanoids'], df['Proline'])
plt.xlabel('Flavanoids')
plt.ylabel('Proline')
plt.title('Flavanoids vs Proline')
plt.show()

km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(df[['Flavanoids','Proline']])
y_predicted

df['cluster']=y_predicted
df.head()

km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]
plt.scatter(df1['Flavanoids'],df1['Proline'],color='green')
plt.scatter(df2['Flavanoids'],df2['Proline'],color='red')
plt.scatter(df3['Flavanoids'],df3['Proline'],color='black')
plt.scatter(df4['Flavanoids'],df4['Proline'],color='blue')
plt.scatter(df5['Flavanoids'],df5['Proline'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',s=200,label='centroid')
plt.xlabel('Flavanoids')
plt.ylabel('Proline')
plt.legend()

sse = []
k_rng = range(1,10)
print(k_rng)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Flavanoids','Proline']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

df

df['cluster'].value_counts()


for i in range(0,max(df.cluster)+1):
  new_df=df[df['cluster']==i]
  print(new_df)
  filename='new'+str(i)+'.csv'
  new_df.to_csv(filename)