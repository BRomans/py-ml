import pandas as pd
import numpy as np


# Load the wine dataset and return it as a pandas dataframe
def load_wine():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label',
                       'Alcohol',
                       'Malic Acid',
                       'Ash',
                       'Alcalinity of ash',
                       'Magnesium',
                       'Total phenols',
                       'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity',
                       'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    return X, y