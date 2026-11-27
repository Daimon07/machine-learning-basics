import pandas as pd
import numpy as np

def ols(X_train_OLS, y_train):
    rank = np.linalg.matrix_rank(X_train_OLS)
    beta, residuals, rank, singular_values = np.linalg.lstsq(X_train_OLS, y_train, rcond=None)

    return beta, residuals, rank, singular_values


def ridge(X_train_OLS, y_train, lam, identity_matrix):
    A = (np.matrix_transpose(X_train_OLS) @ X_train_OLS) + (lam * identity_matrix)
    B = (np.linalg.matrix_transpose(X_train_OLS) @ y_train)
    beta_ridge = np.linalg.solve(A,B)

    return beta_ridge
