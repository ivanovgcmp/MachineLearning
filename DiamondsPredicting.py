import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

# print(df.head(10).to_string())

# Deleting a "unnamed column"
df = df.drop(['Unnamed: 0'], axis = 1)

# Creating variables for categories
categorial_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

# Replacing categories with numerical values
for i in range(3):
    new = le.fit_transform(df[categorial_features[i]])
    df[categorial_features[i]] = new

# print(df.head(10).to_string())

X = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = df[['price']]

# Separation of data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 25, random_state = 101)

# Training
regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 104)
regr.fit(X_train, y_train.values.ravel())

# Prediction
predictions = regr.predict(X_test)

result = X_test
result['price'] = y_test
result['prediction'] = predictions.tolist()

print(result.to_string())


