import pandas as pd
import numpy as np
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

std = np.std(X_train, axis=0)
keepStd = std > 0
X_train = X_train[:, keepStd]
X_test = X_test[:, keepStd]
newStd = np.std(X_train, axis=0)

mean = np.mean(X_train, axis=0)

X_train_standardized = (X_train - mean) / newStd
X_test_standardized = (X_test - mean) / newStd

ones_train = np.ones((X_train.shape[0], 1))
X_train_OLS = np.concatenate((ones_train, X_train_standardized), axis=1)
ones_test = np.ones((X_test.shape[0], 1))
X_test_OLS = np.concatenate((ones_test, X_test_standardized), axis=1)

rank = np.linalg.matrix_rank(X_train_OLS)
beta, residuals, rank, singular_values = np.linalg.lstsq(X_train_OLS, y_train, rcond=None)

y_test_ones = np.ones((y_test.shape[0], 1))
y_train_pred = X_train_OLS @ beta 
y_test_pred = X_test_OLS @ beta

mse_train = np.sum((y_train - y_train_pred)**2) / y_train.size
mse_test = np.sum((y_test - y_test_pred)**2) / y_test.size
mse_test_pass = np.sum(((y_test - (-y_test_ones))**2) / y_test.size)

# print(mse_train) 0.13169642451183636
# print(mse_test) 0.5992126865798275
# print(mse_test_pass) 0.26751592356687903

max_coeff = np.max(beta)
norm = np.linalg.norm(beta)

# print(norm) 279.3841472224295
# print(max_coeff) 172.8120168737687

identity_matrix = np.eye(X_train_OLS.shape[1])
identity_matrix[0, 0] = 0
lambda_value = 1

A = (np.matrix_transpose(X_train_OLS) @ X_train_OLS) + (lambda_value * identity_matrix)
B = (np.linalg.matrix_transpose(X_train_OLS) @ y_train)
beta_ridge = np.linalg.solve(A,B)


y_train_pred_ridge = X_train_OLS @ beta_ridge
y_test_pred_ridge = X_test_OLS @ beta_ridge

mse_train = np.sum((y_train - y_train_pred_ridge)**2) / y_train.size
mse_test = np.sum((y_test - y_test_pred_ridge)**2) / y_test.size

print(mse_train) #0.1444
print(mse_test) #0.3003





