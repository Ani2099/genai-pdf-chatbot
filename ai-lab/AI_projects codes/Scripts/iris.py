import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

Iris=pd.read_csv('Iris.csv')

Iris.head()

Iris.isna().head()

Iris.describe()

Iris.groupby('Species').mean()

Iris['Species'].value_counts()

sns.countplot(x='Species',data=Iris)

plt.figure(2)
plt.subplot(121)
sns.distplot(Iris['Species'])
plt.subplot(122)
Iris['Species'].plot.box(figsize=(16,5))

sns.pairplot(Iris)

corr_matrix=Iris.corr()
sns.heatmap(corr_matrix)

x=Iris.drop(['Species', 'Id'],axis=1)
y=Iris['Species']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg.fit(x_train,y_train.astype('int'))

l_pred=lg.predict(x_test)

l_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test.astype('int'),l_pred)

Iris.columns

!pip install anvil-uplink

import anvil.server

anvil.server.connect('server_3JECFGZVNUYIHIOV7O5UUKS4-63YWJ6776C2J5YE2')

@anvil.server.callable
def CHECK(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
  SepalLengthCm=float(SepalLengthCm)
  SepalWidthCm=float(SepalLengthCm)
  PetalLengthCm=float(PetalLengthCm)
  PetalWidthCm=float(PetalWidthCm)
  real_pred=lg.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
  print(real_pred[0])
  if str(real_pred[0])=="0":
    return "Iris-setosa"
  elif str(real_pred[0])=="1":
    return "Iris-versicolor"
  else:
    return "Iris-virginica"

anvil.server.wait_forever

Iris.columns