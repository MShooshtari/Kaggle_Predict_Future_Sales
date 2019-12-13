import os
import numpy as np 
import pandas as pd 

data_file = 'Data/sales_train_v2.csv'
data_df = pd.read_csv(data_file).astype(str)
print(data_df.head())

product = pd.DataFrame(data_df['shop_id'] + '_' + data_df['item_id'], columns = ['product'])
print(product)
print(product.shape)


unique_products = product['product'].unique()
unique_products_dict = dict()
for i in range(len(unique_products)):
	unique_products_dict[unique_products[i]] = i


unique_months = list(set(data_df['date_block_num'].astype(int)))
unique_months.sort()

sell_per_months_table = np.zeros((len(unique_products), len(unique_months)))
print(sell_per_months_table)

data_df['item_cnt_day'] = data_df['item_cnt_day'].astype(float).astype(int)

for (idx, row) in data_df.iterrows():
	print(idx)
	shop_id = row['shop_id']
	item_id = row['item_id']
	product_key = shop_id + '_' + item_id
	product_idx = unique_products_dict[product_key]
	month_idx = int(row['date_block_num'])
	sell_per_months_table[product_idx][month_idx] += row['item_cnt_day'] #int(float(row['item_cnt_day']))

output_file = 'sell_per_months_table.csv'
output_df = pd.DataFrame(sell_per_months_table, index=unique_products, columns=unique_months)
output_df.to_csv(output_file)