#https://jubilant-cylindrical-master.anvil.app/
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

EQ=pd.read_csv('eq.csv')

EQ.head()

EQ.isna().sum()

EQ.describe()

EQ.groupby('alert').mean()

sns.countplot(x='alert',data=EQ)

plt.figure(2)
plt.subplot(121)
sns.distplot(EQ['alert'])
plt.subplot(122)
EQ['alert'].plot.box(figsize=(16,6))

sns.pairplot(EQ)

corr_matrix=EQ.corr()
sns.heatmap(corr_matrix)

x=EQ.drop('alert',axis=1)
y=EQ['alert']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg.fit(x_train,y_train)

l_pred=lg.predict(x_test)
l_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test,l_pred)

from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(x_train,y_train)

t_pred=dt.predict(x_test)
t_pred

from sklearn.metrics import accuracy_score
accuracy_score(y_test,t_pred)

!pip install anvil-uplink

import anvil.server

anvil.server.connect('server_A4YWRIN7FBHOFWAI2V5LQOF7-M5LJIGQOHA5OBSSW')

@anvil.server.callable
def CHECK(magnitude,depth,cdi,mmi,sig):
  magnitude=float(magnitude)
  depth=float(depth)
  cdi=float(cdi)
  mmi=float(mmi)
  sig=float(sig)
  real_pred=lg.predict([[magnitude,depth,cdi,mmi,sig]])
  print(real_pred[0])
  if str(real_pred[0])=="1":
    return True
  else:
    return False

anvil.server.wait_forever