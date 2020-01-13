# The original idea to have labeled data for supervised learning is coming from this Kaggle Notebook:
# https://www.kaggle.com/fatmakursun/predict-sales-time-series-with-cnn

import pandas as pd 
file_name = 'sell_per_months_table.csv'
file_df = pd.read_csv(file_name)

# print(file_df.head())


timestep = 10

X = []
Y = []

for (idx, row) in file_df.iterrows():
	print(idx)
	temp_id = row[0]
	for i in range (1, len(row)-timestep):
		temp_X = row[i:i+timestep].values
		temp_Y = row[i+timestep]

		X.append(temp_X)
		Y.append(temp_Y)

X_df = pd.DataFrame(X)
Y_df = pd.DataFrame(Y)
Y_df.columns = ['Sale']

labeled_df = pd.concat([X_df, Y_df], axis=1)
print(labeled_df.head())

file_name = 'Sale_Labeled_timestp_'+str(timestep)+'.csv'
labeled_df.to_csv(file_name, index=False)
