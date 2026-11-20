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

# print(shapeX) # (1567, 590)
# print(shapeY) # (1567, 1)
# print(missingValues) 
# print(maxNan) # 1429
# print(totalNan) # 41951

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
stratify=y.ravel()
print(X_train, X_test, y_train, y_test)

print(pd.DataFrame(X_train).iloc[:10, :10])