import pandas as pd 

file_name = 'sell_per_months_table.csv'

file_df = pd.read_csv(file_name)
print(file_df.shape)