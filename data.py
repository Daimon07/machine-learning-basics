import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('secom.csv', delimiter=' ', header=None, encoding='utf-8')
    df_two = pd.read_csv('secom_labels.csv', sep=r'\s+', header=None, encoding='utf-8') 
    df_two = df_two.drop(columns=df_two.columns[1])

    x = df.to_numpy()
    y = df_two.to_numpy()

    return x, y


