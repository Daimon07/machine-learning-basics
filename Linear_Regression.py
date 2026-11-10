import pandas as pd
import numpy as np
import math

df = pd.read_csv('secom.csv', delimiter=' ', header=None, encoding='utf-8', skiprows=1)
df_two = pd.read_csv('secom_labels.csv', sep=r'\s+', header=None, encoding='utf-8', engine='python', skiprows=1) 

df = df.columns[0:99]
print(df)

df_two = df_two.drop(columns=df_two.columns[[1,2]])
failed = 0
passed = 0

failed = (df_two.iloc[:, 0] == -1).sum()
passed = (df_two.iloc[:, 0] == 1).sum()

failure_rate = (failed / (failed + passed)) * 100
print(failure_rate)
