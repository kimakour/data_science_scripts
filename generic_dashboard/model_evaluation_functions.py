import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
import csv
import ast

from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from timeit import default_timer as time


def read_dataset(name_dataset, name_results_set):

	index_train = np.load('x_train_index.npy')
	index_test = np.load ('x_test_index.npy')

	df = pd.read_csv(name_dataset)
	results = pd.read_csv(name_results_set)

	df_x = df.iloc[:,:-2]
	df_y = df.iloc[:,-1]
	X_train = df_x.loc[index_train]
	X_test = df_x.loc[index_test]
	y_train = df_y.loc[index_train]
	y_test  = df_y.loc[index_test]

	results.sort_values('loss', ascending = True, inplace = True)
	results.reset_index(inplace = True, drop = True)

	return X_train,X_test,y_train,y_test,results


def plot_results(X_train,X_test,y_train,y_test,results,position):

	params = ast.literal_eval(results.loc[position,'params'])
	params.update({'n_estimators' : results.loc[position,'estimators'] })

	model = GradientBoostingRegressor( **params).fit(X_train, y_train)

	print( 'mse for train :' , mean_squared_error(y_train, model.predict(X_train)))
	print(' variance of train: ', np.var(y_train), ' variance of predicted train : ',np.var(model.predict(X_train)))

	y_pred = model.predict(X_test)
	print( 'mse for test :' , mean_squared_error(y_test, y_pred))
	print(' variance of test: ', np.var(y_test), ' variance of predicted test : ',np.var(y_pred))

	print('Correlation coefficient : ', pearsonr(y_test, y_pred)[0])

	plt.plot(y_test,y_pred,'r.') 
	plt.plot(y_test,y_test,'k-') 
	plt.xlabel('y_test')
	plt.ylabel('y_pred')
	plt.title('Scatter plot y_test vs y_pred')
	return model,y_pred

def define_new_score(X_train,X_test,y_train,y_test,results):
	liste_metrics = list()

	for index, row in results.iterrows():
		params = ast.literal_eval(row['params'])
		params.update({'n_estimators' : row['estimators'] })
		model = GradientBoostingRegressor( **params).fit(X_train, y_train)
		y_pred = model.predict(X_test)
		liste_metrics.append([index, np.var(y_pred), mean_squared_error(y_test, y_pred),spearmanr(np.array(y_test), y_pred)[0]])

	dataframe_results = pd.DataFrame(liste_metrics).drop([0],1)
	dataframe_results = dataframe_results.rename(columns={1:'variance',2:'mse', 3:'corr_coef'})
	return dataframe_results

def modifying_score(dataframe_results , scoring_var = 1, scoring_mse=1, scoring_corr =1):
	dataframe_results['scoring']= (dataframe_results['variance']**scoring_var) * (dataframe_results['corr_coef']**scoring_corr)/ (dataframe_results['mse']**scoring_mse)
	return dataframe_results

def compute_adjusted_cosine_similarity(feature_importances,X_train, X_train_position,X_test , X_test_position):
	mmsc =MinMaxScaler().fit(X_train)
	a = np.multiply(feature_importance,mmsc.transform(np.array(X_train.loc[X_train_position,:]).reshape(1, -1)))
	b = np.multiply(feature_importance,mmsc.transform(np.array(X_test.loc[X_test_position,:]).reshape(1, -1)))
	return cosine_similarity(a,b)[0][0]

