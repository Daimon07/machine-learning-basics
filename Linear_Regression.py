import pandas as pd
import math

df = pd.read_excel('default_of_credit_card_clients.xls', header = 1) # Install xlrd package due to old excel file format
print(df.head())

# CHECKS

default_payment_column = df['default payment next month']
num_defaults = (default_payment_column == 1).sum()
num_non_defaults = (default_payment_column == 0).sum()
num_random = ((default_payment_column != 1) & (default_payment_column != 0)).sum() 


print("Defaults: ", num_defaults)
print("Non-Defaults: ", num_non_defaults)
print("Random: ", num_random)
print(df["EDUCATION"].value_counts())
print(df["MARRIAGE"].value_counts())
print(df["SEX"].value_counts())
print(df.describe())
