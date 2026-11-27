import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(x, y):
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

    return X_train_OLS, X_test_OLS, y_train, y_test
