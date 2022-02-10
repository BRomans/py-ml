import pandas as pd
import numpy as np
from io import StringIO

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

csv_data = '''A,B,C,D\n1.0,2.0,3.0,4.0\n5.0,6.0,,8.0\n10.0,11.0,12.0'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.isnull().sum())  # Count NaN values


imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

print(df.dropna())  # drop rows containing NaN values
print(df.dropna(axis=1))  # drop columns containing NaN values
print(df.dropna(how='all'))  # drop columns containing only NaN values
print(df.dropna(thresh=4))  # drop row containing that have at least 4 non-NaN values
print(df.dropna(subset=['C']))  # drop rows where NaN values are in the specified column

# Handling categorical data
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                  ['red', 'L', 13.5, 'class2'],
                  [ 'blue', 'XL', 15.3, 'class1']], columns=['color', 'size', 'price', 'classlabel'])

print(df)

# Mapping ordinal features
size_mapping = {
    'XL': 3,
    'L' : 2,
    'M' : 1
    }

df['size'] = df['size'].map(size_mapping)
print('Mapping ordinal features\n',df)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print('Reverse\n', df)

# Encoding class labels
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print('Encoding class labels\n', df)
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print('Decoding\n', df)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print('LabelEncoder', y)

# One-hot encoding on nominal features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print('Normal encoding\n', X)
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
print('One-Hot encoding\n', ct.fit_transform(X))