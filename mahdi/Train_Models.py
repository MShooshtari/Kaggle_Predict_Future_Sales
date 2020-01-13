from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

import pandas as pd 
import numpy as np 

# def cal_rmse(model, X, y):
#     return (np.sqrt(mean_squared_error(y, model.predict(X))))

def cal_accuracy(model, X, y):
    return (np.sqrt(mean_squared_error(y, model.predict(X))))

train_file_list = ['Sale_Labeled_timestp_30.csv', 'Sale_Labeled_timestp_20.csv', 'Sale_Labeled_timestp_30.csv']


models_names = ['KernelRidge', 'ElasticNet', 'Lasso', 'GradientBoostingRegressor', 'BayesianRidge', 'LassoLarsIC', 'RandomForestRegressor', 'XGBRegressor']
models_list = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), XGBRegressor()]


KR_param_grid = {'alpha': 0.1, 'coef0': 100, 'degree': 1, 'gamma': None, 'kernel': 'polynomial'}
EN_param_grid = {'alpha': 0.001, 'copy_X': True, 'l1_ratio': 0.6, 'fit_intercept': True, 'normalize': False, 
                     'precompute': False, 'max_iter': 300, 'tol': 0.001, 'selection': 'random', 'random_state': None}
LASS_param_grid = {'alpha': 0.0005, 'copy_X': True, 'fit_intercept': True, 'normalize': False, 'precompute': False, 
                'max_iter': 300, 'tol': 0.01, 'selection': 'random', 'random_state': None}
GB_param_grid = {'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 3, 
                                    'min_samples_split': 0.0025, 'min_samples_leaf': 5}
BR_param_grid = {'n_iter': 200, 'tol': 0.00001, 'alpha_1': 0.00000001, 'alpha_2': 0.000005, 'lambda_1': 0.000005, 
             'lambda_2': 0.00000001, 'copy_X': True}
LL_param_grid = {'criterion': 'aic', 'normalize': True, 'max_iter': 100, 'copy_X': True, 'precompute': 'auto', 'eps': 0.000001}
RFR_param_grid = {'n_estimators': 50, 'max_features': 'auto', 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2}
XGB_param_grid = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 300, 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.1,
              'reg_lambda': 0.7, 'max_delta_step': 0, 'min_child_weight': 1, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.2,
              'scale_pos_weight': 1}


params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]



for train_file in train_file_list:
	train_df = pd.read_csv(train_file)

	resultVar = ['Sale']

	X = train_df.drop(resultVar, axis=1)
	y = train_df[resultVar]

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

	for i in range(6,8):#(len(models_list)):
		print(models_names[i])
		model = models_list[i]
		params = params_grid[i]
		print(params)
		pipeline = make_pipeline(RobustScaler(), model.set_params(**params))

		pipeline.fit(X_train, y_train)    

		print('Normal ' + models_names[i])
		print('Train rmse: ')
		print(cal_accuracy(pipeline, X_train, y_train))
		print('Validation rmse: ')
		print(cal_accuracy(pipeline, X_val, y_val))
		# print(models_names[i]+' with transformed skewed columns')
		# print('Train rmse: ')
		# print(cal_accuracy(pipeline_transformed, X_train, y_train))
		# print('Validation rmse: ')
		# print(cal_accuracy(pipeline_transformed, X_val, y_val))
		# print('-----------------------------------------------------')
		# exit()


