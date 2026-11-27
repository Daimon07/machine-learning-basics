import pandas as pd
import numpy as np

def mse(y_train_pred, y_test_pred, y_train, y_test):

    mse_train = np.sum((y_train - y_train_pred)**2) / y_train.size
    mse_test = np.sum((y_test - y_test_pred)**2) / y_test.size

    return mse_train, mse_test
