import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

print(df.head(10).to_string())

# Deleting a "unnamed column"
df = df.drop(['Unnamed: 0'], axis = 1)

# Creating variables for categories

