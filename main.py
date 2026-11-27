import pandas as pd
import numpy as np

from data import load_data
from preprocessing import preprocess
from models import ols, ridge
from evaluation import mse
from plots import plot_ridge


x, y = load_data()

passed = (y[:, 0] == -1).sum()
failed = (y[:, 0] == 1).sum()
failure_rate = (failed / (failed + passed)) * 100

shapeX = np.shape(x)
shapeY = np.shape(y)
missingValues = np.isnan(x).sum(axis=0)
maxNan = missingValues.max(axis=0)
totalNan = missingValues.sum(axis=0)

X_train_OLS, X_test_OLS, y_train, y_test = preprocess(x, y)

beta, residuals, rank, singular_values = ols(X_train_OLS, y_train)

y_test_ones = np.ones((y_test.shape[0], 1))
y_train_pred = X_train_OLS @ beta
y_test_pred = X_test_OLS @ beta

mse_train, mse_test, mse_test_pass = mse(
    y_train_pred, y_test_pred, y_train, y_test, y_test_ones
)

print(mse_train)
print(mse_test)
print(mse_test_pass)

max_coeff = np.max(beta)
norm = np.linalg.norm(beta)

print(norm)
print(max_coeff)

identity_matrix = np.eye(X_train_OLS.shape[1])
identity_matrix[0, 0] = 0
lambda_values = np.logspace(-3, 12, 20)

mse_train_list = []
mse_test_list = []
norm_coeffs = []

for lam in lambda_values:

    beta_ridge = ridge(X_train_OLS, y_train, lam, identity_matrix)

    y_train_pred_ridge = X_train_OLS @ beta_ridge
    y_test_pred_ridge = X_test_OLS @ beta_ridge

    mse_train = np.sum((y_train - y_train_pred_ridge)**2) / y_train.size
    mse_test = np.sum((y_test - y_test_pred_ridge)**2) / y_test.size
    coeffs = np.sum((beta_ridge[1:])**2)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    norm_coeffs.append(coeffs)


plot_ridge(lambda_values, mse_train_list, mse_test_list, norm_coeffs)
