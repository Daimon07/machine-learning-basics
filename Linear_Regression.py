import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


df = pd.read_csv('secom.csv', delimiter=' ', header=None, encoding='utf-8')
df_two = pd.read_csv('secom_labels.csv', sep=r'\s+', header=None, encoding='utf-8') 
df_two = df_two.drop(columns=df_two.columns[1])

passed = (df_two.iloc[:, 0] == -1).sum()
failed = (df_two.iloc[:, 0] == 1).sum()
failure_rate = (failed / (failed + passed)) * 100

x = df.to_numpy()
y = df_two.to_numpy()

shapeX = np.shape(x)
shapeY = np.shape(y)
missingValues = np.isnan(x).sum(axis=0)
maxNan = missingValues.max(axis=0)
totalNan = missingValues.sum(axis=0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42, stratify=y.ravel())

missingFraction = np.isnan(X_train).mean(axis=0)
keepColumns = missingFraction <= 0.5

X_train = X_train[:, keepColumns]
X_test = X_test[:, keepColumns]

medians = np.nanmedian(X_train, axis=0)
X_train = np.where(np.isnan(X_train), medians, X_train)
X_test = np.where(np.isnan(X_test), medians, X_test)

print(np.isnan(X_train).sum())
print(np.isnan(X_test).sum())