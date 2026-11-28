import pandas as pd
import numpy as np
from sklearn import linear_model

def ols(x_train, y_train):
    rank = np.linalg.matrix_rank(x_train)
    beta, residuals, rank, singular_values = np.linalg.lstsq(x_train, y_train, rcond=None)

    return beta, residuals, rank, singular_values


def ridge(x_train, y_train, lam, identity_matrix):
    A = (np.matrix_transpose(x_train) @ x_train) + (lam * identity_matrix)
    B = (np.linalg.matrix_transpose(x_train) @ y_train)
    beta_ridge = np.linalg.solve(A,B)

    return beta_ridge

def lasso(x_train, y_train, lam):
    linear_model()
    return;