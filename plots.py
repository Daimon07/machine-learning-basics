import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ridge(lambda_values, mse_train_list, mse_test_list, norm_coeffs):

    plt.semilogx(lambda_values, mse_train_list, label="Training MSE")
    plt.semilogx(lambda_values, mse_test_list, label="Testing MSE")
    plt.xlabel("Lambda Values")
    plt.ylabel("Error (MSE)")
    plt.title("Ridge: MSE w/ Lambda variations")
    plt.legend()
    plt.grid(True)

    plt.show()

    plt.semilogx(lambda_values, norm_coeffs, label="L2 Norm")
    plt.xlabel("Lambda Values")
    plt.ylabel("Norm Coeffs")
    plt.title("Ridge: Norm coeffs w/ Lambda variations")
    plt.legend()
    plt.grid(True)

    plt.show()
