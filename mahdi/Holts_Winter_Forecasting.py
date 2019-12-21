import pandas as pd 

# given a series and alpha, return series of smoothed points
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def holts_winter_forecast(row):
	alpha = 0.9
	return (exponential_smoothing(row, alpha))


filename = 'sell_per_months_table.csv'

sell_df = pd.read_csv(filename, index_col=0)

results = dict()

for (idx, row) in sell_df.iterrows():
	prediction = int(holts_winter_forecast(row)[-1])
	results[idx] = prediction


result_file = 'Data/test.csv'
result_df = pd.read_csv(result_file)
print(result_df.head())
output_list = []
for (idx, row) in result_df.iterrows():
	print(idx)
	print(row)
	key = str(row['shop_id'])+'_'+str(row['item_id'])
	if (key in results):
		value = results[key]
	else:
		value = 0

	output_list.append([row['ID'], value])

output_df = pd.DataFrame(output_list, columns=['ID', 'item_cnt_month'])
output_df.to_csv('test.csv')
