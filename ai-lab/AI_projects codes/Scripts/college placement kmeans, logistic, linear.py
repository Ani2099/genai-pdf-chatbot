from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("college_placement.csv")
df.fillna('NA', inplace=True)
df.head()



plt.scatter(df['CGPA'], df['Placement_Readiness_Score'])
plt.xlabel('CGPA')
plt.ylabel('Placement_Readiness_Score')
plt.title('CGPA vs Placement_Readiness_Score')
plt.show()

plt.scatter(df['Confidence_Level'], df['Placement_Readiness_Score'])
plt.xlabel('Confidence_Level')
plt.ylabel('Placement_Readiness_Score')
plt.title('Confidence_Level vs Placement_Readiness_Score')
plt.show()

plt.scatter(df.CGPA,df['Confidence_Level'])
plt.xlabel('CGPA')
plt.ylabel('Confidence_Level')

plt.scatter(df['10th_Percentage'],df['CGPA'])
plt.xlabel('10th_Percentage')
plt.ylabel('CGPA')

plt.scatter(df['Backlogs'], df['Placement_Readiness_Score'])
plt.xlabel('Backlogs')
plt.ylabel('Placement_Readiness_Score')
plt.title('Backlogs vs Placement_Readiness_Score')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['CGPA','Placement_Readiness_Score']])
y_predicted

df['cluster']=y_predicted
df.head()

km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',s=200,label='centroid')
plt.xlabel('CGPA')
plt.ylabel('Placement_Readiness_Score')
plt.legend()
plt.show()

df['cluster']=y_predicted
df.head()

sse = []
k_rng = range(1,10)
print(k_rng)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['CGPA','Placement_Readiness_Score']])
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

dfnew= pd.read_csv ("new0.csv")
dfnew.head(20)

def categorize_student_revised(row):
    if (row['Programming_Score'] > 8 and
        row['Confidence_Level'] > 8.5 and
        row['Technical_Skills_Score'] > 8.5):
        return 'above 80%'

    elif (row['Programming_Score'] > 7 and row['Programming_Score'] <= 8 and
          row['Confidence_Level'] > 7.5 and row['Confidence_Level'] <= 8.5 and
          row['Technical_Skills_Score'] > 7.5 and row['Technical_Skills_Score'] <= 8.5):
        return 'between 60-80%'

    elif (row['Programming_Score'] > 6 and row['Programming_Score'] <= 7 and
          row['Confidence_Level'] > 6.5 and row['Confidence_Level'] <= 7.5 and
          row['Technical_Skills_Score'] > 6.5 and row['Technical_Skills_Score'] <= 7.5):
        return 'between 50-60%'

    else:
        return 'below 50%'


category_mapping = {
    'above 80%': 0,
    'between 60-80%': 1,
    'between 50-60%': 2,
    'below 50%': 3
}

dfnew['category'] = dfnew.apply(categorize_student_revised, axis=1)
dfnew['category'] = dfnew['category'].replace(category_mapping)
display(dfnew.head())
display(dfnew['category'].value_counts())

display(dfnew['category'].value_counts())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='category',data=dfnew)

corr_matrix = dfnew.drop(columns=['Unnamed: 0']).select_dtypes(include='number').corr()
sns.heatmap(corr_matrix)

dfnew_cleaned = dfnew.dropna(subset=['category'])
x = dfnew_cleaned.select_dtypes(include='number').drop(columns=['category', 'Unnamed: 0', 'cluster'])
y = dfnew_cleaned['category']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

l_pred=lg.fit(x_train, y_train).predict(x_test.astype('int'))

l_pred=lg.fit(x_train, y_train).predict(x_test.astype('int'))

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



def create_package_column(row):
    if row['category'] == 0:
        return '11-12 LPA'
    elif row['category'] == 1:
        return '08-10 LPA'
    elif row['category'] == 2:
        return '06-07 LPA'
    else:
        return '04-06 LPA'

dfnew['package_offered'] = dfnew.apply(create_package_column, axis=1)
display(dfnew.head())

display(dfnew['package_offered'].value_counts())

package_offered = {
    '11-12 LPA': 0,
    '08-10 LPA': 1,
    '06-07 LPA': 2,
    '04-06 LPA': 3
}

dfnew['package_offered'] = dfnew.apply(categorize_student_revised, axis=1)
dfnew['package_offered'] = dfnew['package_offered'].replace(category_mapping)
display(dfnew.head())
display(dfnew['package_offered'].value_counts())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

x_train

x_test.head()

y_train.head()

y_test.head()

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

y_prediction=lr.predict(x_test)

y_test.head()


y_prediction[0:5]

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_prediction)



y=dfnew[['package_offered']]
x=dfnew[['category']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

x_test.head()

lr2=LinearRegression()

lr2.fit(x_train,y_train)

y_prediction=lr2.predict(x_test)

y_prediction[0:5]

y_prediction.flatten()[0:10]

mean_squared_error(y_test,y_prediction)

y_test.head()


display(dfnew['package_offered'].value_counts())

real=pd.read_csv('new1.csv')

def categorize_student_revised(row):
    if (row['Programming_Score'] > 8 and
        row['Confidence_Level'] > 8.5 and
        row['Technical_Skills_Score'] > 8.5):
        return 'above 80%'

    elif (row['Programming_Score'] > 7 and row['Programming_Score'] <= 8 and
          row['Confidence_Level'] > 7.5 and row['Confidence_Level'] <= 8.5 and
          row['Technical_Skills_Score'] > 7.5 and row['Technical_Skills_Score'] <= 7.5):
        return 'between 60-80%'

    elif (row['Programming_Score'] > 6 and row['Programming_Score'] <= 7 and
          row['Confidence_Level'] > 6.5 and row['Confidence_Level'] <= 7.5 and
          row['Technical_Skills_Score'] > 6.5 and row['Technical_Skills_Score'] <= 7.5):
        return 'between 50-60%'

    else:
        return 'below 50%'

category_mapping = {
    'above 80%': 0,
    'between 60-80%': 1,
    'between 50-60%': 2,
    'below 50%': 3
}

real['category'] = real.apply(categorize_student_revised, axis=1)
real['category'] = real['category'].replace(category_mapping)
real['category'] = pd.to_numeric(real['category'], errors='coerce')
real.dropna(subset=['category'], inplace=True)

display(real.head())
display(real['category'].value_counts())

real = real.dropna(subset=['category'])
l_pred1 = lr2.predict(real[['category']])
l_pred1 = np.round(l_pred1).astype(int)
l_pred1 = np.clip(l_pred1, 0, 3)
l_pred1

# Store predictions in a dictionary
predictions_dict = {f'prediction_{i}': l_pred1[i] for i in range(len(l_pred1))}
print("Predictions Dictionary:", predictions_dict)

# Get value counts of predicted species and map to names
species_map = {0: '11-12 LPA', 3: '04-06 LPA'}
predicted_species_counts = pd.Series(l_pred1.flatten()).map(species_map).value_counts()
print("\nPredicted Species Value Counts:\n", predicted_species_counts)

for i in range(len(l_pred1)):
    if l_pred1[i]==0:
        print('Great offer' )
    elif l_pred1[i]==3:
        print(' Good offer')
    else :
        print( 'Good job')

DEPLOYMENT

!pip install anvil-uplink
import anvil.server
import anvil.media


import pandas as pd
import anvil.mpl_util
import matplotlib.pyplot as plt

anvil.server.connect("server_MFFLHALR5ZX7AH6PBFK6J7NU-KBJV4SN7PUKPYO42")

@anvil.server.callable
def process_csv(file):
    with anvil.media.TempFile(file) as filepath:
        df = pd.read_csv(filepath)
        real_prediction=lr2.predict(df)
        real_prediction=real_prediction.flatten()
        # Plot it in the normal Matplotlib way
        plt.figure(1, figsize=(10,5))
        plt.plot(real_prediction)
        # Return this plot as a PNG image in a Media object
        return anvil.mpl_util.plot_image()
