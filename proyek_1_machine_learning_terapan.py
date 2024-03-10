# -*- coding: utf-8 -*-
"""Proyek 1 Machine Learning Terapan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1izlHr-1sfy2Zy14CYRg8SMQrg1OVDAfN

# Proyek 1 Machine Learning Terapan: Used Cars Price Prediction

### Menyiapkan Library yang Dibutuhkan
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

"""### Menyiapkan Dataset"""

!pip install -q kaggle

!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d 'colearninglounge/used-cars-price-prediction'

!cp used-cars-price-prediction.zip /tmp

import os
import zipfile
local_zip = '/tmp/used-cars-price-prediction.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

df = pd.read_csv('/tmp/train.csv')

"""## Data Understanding"""

df

df.info()

df.describe()

df = df.drop('New_Price', axis=1)

df.info()
df.head()

df['Mileage'] = pd.to_numeric(df['Mileage'].str.extract('(\d+\.\d+|\d+)', expand=False), errors='coerce')
df['Engine'] = pd.to_numeric(df['Engine'].str.extract('(\d+\.\d+|\d+)', expand=False), errors='coerce')
df['Power'] = pd.to_numeric(df['Power'].str.extract('(\d+\.\d+|\d+)', expand=False), errors='coerce')

df = df.dropna()
df.shape

df.info()
df.head()

df['Year'] = df['Year'].astype(str)
df['Seats'] = df['Seats'].astype(int)
df.info()

df = df.drop('Name', axis=1)
df.info()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
diamonds=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

diamonds.shape

categorical_columns = ['Location', 'Year', 'Fuel_Type', 'Transmission', 'Owner_Type']
numerical_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']

feature = categorical_columns[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df2 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df2)
count.plot(kind='bar', title=feature);

feature = categorical_columns[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df2 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df2)
count.plot(kind='bar', title=feature);

feature = categorical_columns[2]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df2 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df2)
count.plot(kind='bar', title=feature);

feature = categorical_columns[3]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df2 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df2)
count.plot(kind='bar', title=feature);

feature = categorical_columns[4]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df2 = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df2)
count.plot(kind='bar', title=feature);

df.hist(bins=50, figsize=(20,15))
plt.show()

for col in categorical_columns:
  sns.catplot(x=col, y="Price", kind="bar", dodge=False, height = 4, aspect = 3,  data=df, palette="Set3")
  plt.title("Rata-rata 'Price' Relatif terhadap - {}".format(col))

sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

df.drop(['Seats', 'Kilometers_Driven'], inplace=True, axis=1)
df.head()

"""## Data Preparation"""

df = pd.concat([df, pd.get_dummies(df['Location'], prefix='Location')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Year'], prefix='Year')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Fuel_Type'], prefix='Fuel_Type')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Transmission'], prefix='Transmission')], axis=1)
df = pd.concat([df, pd.get_dummies(df['Owner_Type'], prefix='Owner_Type')], axis=1)
df.drop(['Location', 'Year', 'Fuel_Type', 'Transmission', 'Owner_Type'], axis=1, inplace=True)
df.head()

pca = PCA(n_components=2, random_state=123)
pca.fit(df[['Engine', 'Power']])
princ_comp = pca.transform(df[['Engine', 'Power']])

pca.explained_variance_ratio_.round(3)

pca = PCA(n_components=1, random_state=123)
pca.fit(df[['Engine', 'Power']])
df['EnginePower'] = pca.transform(df.loc[:, ('Engine', 'Power')]).flatten()
df.drop(['Engine', 'Power'], axis=1, inplace=True)

df.head()

X = df.drop(["Price"],axis =1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

numerical_features = ['Mileage', 'EnginePower']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""## Model Development"""

models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""## Evaluation"""

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:5].copy()
pred_dict = {'y_true':y_test[:5]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""#### Model dengan performa terbaik: Random Forest"""