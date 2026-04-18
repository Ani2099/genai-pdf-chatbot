

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)
data=pd.read_csv('IRIS.csv')
data['species'] = data['species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])



data.head(150)

data.isna().sum()

data.describe()

data.groupby('species').mean()


data['species'].value_counts()

sns.countplot(x='species',data=data)

plt.figure(2)
plt.subplot(121)
sns.distplot(data['sepal_length'])
plt.subplot(122)
data['sepal_length'].plot.box(figsize=(15,5))

sns.pairplot(data)

corr_matrix=data.corr()
sns.heatmap(corr_matrix)

x=data.drop(columns='species',axis=1)
y=data['species']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg=LogisticRegression()
lg.fit(x_train,y_train.astype('int'))



l_pred=lg.predict(x_test.astype('int'))

l_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test.astype(int),l_pred)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test.astype(int),l_pred))

print(classification_report(y_test.astype(int),l_pred))

from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(x_train,y_train.astype(int))

pred=dt.predict(x_test)
pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test.astype(int),pred)

**Deployemnt**

data.columns


import csv
x_test1 = [['sepal_length','sepal_width', 'petal_length', 'petal_width']]
data_rows = []
for i in range (2):
      sepal_length=float(input("enter sepal_length"))
      sepal_width=float(input("enter sepal_width"))
      petal_length=float(input("enter petal_length"))
      petal_width=float(input("enter petal_width"))
      data_rows.append([sepal_length,sepal_width,petal_length,petal_width])

x_test1.extend(data_rows)


f=open("real_data.csv","w")
t=csv.writer(f)
t.writerows(x_test1)
f.close()



real=pd.read_csv('real_data.csv')
#real1=real.drop("name",axis=1)
real = real.iloc[0:]
l_pred1=lg.predict(real)
l_pred1

for i in range(len(l_pred1)):
    if l_pred1[i]==0:
        print('Iris-setosa' )
    elif l_pred1[i]==1:
        print(' Iris-versicolor')
    else :
        print( 'Iris-virginica')


