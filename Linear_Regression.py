import pandas as pd
import numpy as np
import math

df = pd.read_csv('secom.csv', delimiter=' ', header=None, encoding='utf-8')
df_two = pd.read_csv('secom_labels.csv', sep=r'\s+', header=None, encoding='utf-8', engine='python', skiprows=1) 

print(df)
print(df_two)


df_two = df_two.drop(columns=df_two.columns[[1,2]])
failed = 0
passed = 0

for i in range(0, len(df_two)):
    if df_two.iloc[i, 0] == -1:
        failed += 1
    elif df_two.iloc[i, 0] == 1:
        passed += 1
    else:
        None
print(passed, failed)
