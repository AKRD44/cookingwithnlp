import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import cross_val_score

from pandas.tools.plotting import lag_plot,autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

import warnings
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


from gensim.models import Word2Vec
import logging

# import and setup modules we'll be using in this notebook
import os
import sys
import re
import tarfile
import itertools

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases



logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')




def create_stats(old_df,window,rolling_mean_shifted=True):
	df = old_df.copy()
	for each_col_name in df.columns:
		#here you're including the latest data in your calculations
		if rolling_mean_shifted==False:
			df[each_col_name+"_RollingMean"]=df[each_col_name].rolling(window=window).mean()
		
		else:
		#here we assume we do not have the data for time t yet, if we're predicting for time t. If you pick this one, have one extra row to drop
			df[each_col_name+"_RollingMean"]=df[each_col_name].rolling(window=window).mean().shift() #the shift is because I can't use the value at time t+1 if I'm predicting for t+1
		
		#df[each_col_name+"DifferencedRollingMean"]=df[each_col_name+"RollingMean"]-df[each_col_name+"RollingMean"].shift()
		df[each_col_name+"_RollingStd"]=df[each_col_name].rolling(window=window).std()
		df[each_col_name+"_RelAvg"]=(df[each_col_name]/df[each_col_name+"_RollingMean"])-1
		df[each_col_name+"_RelAvgDifferenced"]=df[each_col_name+"_RelAvg"]-df[each_col_name+"_RelAvg"].shift()
		
		#if you pick this, you gotta drop twice the lenght of windows. Here we've got an average on an average
		RelAvgRollingMean=True
		df[each_col_name+"_RelAvgRollingMean"]=df[each_col_name+"_RelAvg"].rolling(window=window).mean()
		
		#df[each_col_name+"RelAvgRollingMeanDifferenced"]=df[each_col_name+"RelAvgRollingMean"]-df[each_col_name+"RelAvgRollingMean"].shift()
		df[each_col_name+"_RollingStandardization"]=(df[each_col_name]-df[each_col_name+"_RollingMean"])/df[each_col_name+"_RollingStd"]
		#df[each_col_name+"RollingStandardizationDifferenced"]=df[each_col_name+"RollingStandardization"]-df[each_col_name+"RollingStandardization"].shift()
		df[each_col_name+"_RollingNormalization"]=(df[each_col_name]-df[each_col_name+"_RollingMean"])/(df[each_col_name].rolling(window=window).max()-df[each_col_name].rolling(window=window).min())
		#df[each_col_name+"RollingNormalizationDifferenced"]=df[each_col_name+"RollingNormalization"]-df[each_col_name+"RollingNormalization"].shift()
		
		df=df.drop(each_col_name,axis=1)
	
	rows_to_drop=window
	if RelAvgRollingMean==True:
		rows_to_drop=rows_to_drop*2
	if rolling_mean_shifted==False:
		rows_to_drop-=1
	
	#df=df.ix[(window)-1:];
	df=df.ix[rows_to_drop:]; 
	#df.dropna(inplace=True)
	df.fillna(0,inplace=True);
	return df

def chart_stats(df):
	nber_of_graphs=df.columns.shape[0]
	graphs_per_row=3
	nber_of_rows= int(nber_of_graphs/graphs_per_row)

	nber_of_rows_number=nber_of_rows*100
	graphs_per_row_number=graphs_per_row*10
	i=1
	base_number=nber_of_rows_number+graphs_per_row_number
	plt.rcParams['figure.figsize'] = (15, 3*nber_of_rows)
	for each_column in df.columns:	

		plt.subplot(base_number+i) 
		plt.plot(df[each_column])
		plt.title(each_column)
		i+=1
		plt.tight_layout()

	
	
	
from statsmodels.tsa.stattools import adfuller
	
def test_stationarity(timeseries):
	
	#Determing rolling statistics
	rolmean =timeseries.rolling(window=12,center=False).mean()
	rolstd = timeseries.rolling(window=12,center=False).std()
	#rolmean = pd.rolling_mean(timeseries, window=12)
	#rolstd = pd.rolling_std(timeseries, window=12)

	#Plot rolling statistics:
	orig = plt.plot(timeseries, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=False)
	
	#Perform Dickey-Fuller test:
	print ('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries, maxlag=0,autolag=None) #autolag='AIC'
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print (dfoutput)
	
def rmse_cv(model,x_train, y_train,cv=5):
	rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
	#with scoring being blank, by default this would've outputted the accuracy, ex: 95%
	#with scoring="neg_mean_squared_error", we get accuracy -1, so shows by how much you were off and it's negative
	#then with the - in front, gives you the error, but positive. 
	return(rmse)
	
def time_series_info_on_y(y_serie):
	
	#lag_plot(pd.DataFrame(y_serie)) #doesn't show properly

	#autocorrelation_plot(y_serie)
	
	plot_acf(y_serie, lags=31)
	plot_pacf(y_serie, lags=31)

def time_series_heatmap(y_series,lags=12):
	values=pd.DataFrame(y_series)
	number_of_lags_to_check=lags
	column_names=["t"]
	for i in list(range(1,number_of_lags_to_check+1)):
		values = pd.concat([values,values.iloc[:,-1].shift(i)], axis=1)
		column_names.append("t+%i"%i)
	
	values.columns=column_names	
	values=values.dropna()
	sns.heatmap(values.corr().abs())	
	

def train_test(train_pct,x=None,y=None):
	#put in either x or y or both, as long as you identify them. Also put in your training pct, and I'll out the results in dictionary.
	
	data_dict={}
	if type(y) not in [pd.core.series.Series, pd.core.frame.DataFrame]:
		#print("Detected only X")
		train_size=int(x.shape[0]*train_pct)
		x_train=x[:train_size]
		x_test=x[train_size:]
		data_dict["x_train"]=x_train
		data_dict["x_test"]=x_test   
		mid_data_index=int(x_test.shape[0]*0.5)		
	elif type(x) not in [pd.core.series.Series, pd.core.frame.DataFrame]:
		#print("Detected only Y")
		train_size=int(y.shape[0]*train_pct)
		y_train=y[:train_size]
		y_test=y[train_size:]   
		mid_data_index=int(y_test.shape[0]*0.5)
		data_dict["y_train"]=y_train
		data_dict["y_test"]=y_test   
	else:
		if x.shape[0] != y.shape[0]:
			raise Exception("x and y don't have the same number of rows")
		train_size=int(y.shape[0]*train_pct)
		x_train=x[:train_size]
		y_train=y[:train_size]
		x_test=x[train_size:]
		y_test=y[train_size:]
		data_dict["x_train"]=x_train
		data_dict["x_test"]=x_test				   
		data_dict["y_train"]=y_train
		data_dict["y_test"]=y_test	 
		mid_data_index=int(y_test.shape[0]*0.5)
	data_dict["mid_data_index"]=mid_data_index
	data_dict["train_test_index"]=train_size
	
	return data_dict

	
def analyze_sarimax(y,x=None,pct=0.8,season_length=12):
	# Define the p, d and q parameters to take any value between 0 and 2
	p = d = q = range(0, 2)

	# Generate all different combinations of p, q and q triplets
	pdq = list(itertools.product(p, d, q))

	# Generate all different combinations of seasonal p, q and q triplets
	seasonal_pdq = [(x[0], x[1], x[2], season_length) for x in list(itertools.product(p, d, q))]

	print('Examples of parameter combinations for Seasonal ARIMA...')
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


	warnings.filterwarnings("ignore") # specify to ignore warning messages
	best_score, best_cfg = float("inf"), None
	for param in pdq:
		for param_seasonal in seasonal_pdq:
			
			try:
				mod = sm.tsa.statespace.SARIMAX(y,
												exog =x,
												order=param,
												seasonal_order=param_seasonal,
												enforce_stationarity=False,
												enforce_invertibility=False)

				results = mod.fit()

				
				if results.aic< best_score:
					best_score=results.aic
					best_cfg=[param, param_seasonal]
					print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
			except:
				continue

	print("BEST FIT")			
	print('ARIMA{}x{}12 - AIC:{}'.format(best_cfg[0], best_cfg[1], best_score))
	#showing the best one
	mod = sm.tsa.statespace.SARIMAX(y,
									exog =x,
									order=best_cfg[0],#order=(1, 0, 1),
									seasonal_order=best_cfg[1],#seasonal_order=(1, 0, 1, 12),
									enforce_stationarity=False,
									enforce_invertibility=False)

	results = mod.fit()
	print(results.summary().tables[1])
	
	#showing the diagnostics
	results.plot_diagnostics(figsize=(15, 12))
	plt.show()
	
	
	data_dict=train_test(pct,x=x,y=y)
	
	y_test=data_dict["y_test"]
	#x_test=data_dict["x_test"]
	mid_data_point=data_dict["mid_data_point"]
	
	#forecasting  NOT DYNAMIC
	print("FORECASTING NOT DYNAMIC")
	pred = results.get_prediction(start=pd.to_datetime(y_test.index[0]), dynamic=False)
	#The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
	pred_ci = pred.conf_int()
	
	ax = y.plot(label='observed')
	pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

	ax.fill_between(pred_ci.index,
					pred_ci.iloc[:, 0],
					pred_ci.iloc[:, 1], color='k', alpha=.2)

	ax.set_xlabel('Date')
	ax.set_ylabel('Y')
	plt.legend()

	plt.show()
	
	y_forecasted = pred.predicted_mean #you'd add whatever other coefficients you want here. 
	y_truth = logged_bar_data[mid_data_point:]

	# Compute the mean square error
	mse = ((y_forecasted - y_truth) ** 2).mean()
	print('The Mean Squared Error of our forecasts is {}'.format(mse))
	
	
	#forecasting DYNAMIC
	
	print("FORECASTING DYNAMIC")
	pred_dynamic = results.get_prediction(start=pd.to_datetime(y_test.index[0]), dynamic=True, full_results=True)
	pred_dynamic_ci = pred_dynamic.conf_int()
	ax = y[mid_data_point:].plot(label='observed', figsize=(20, 15))
	pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

	ax.fill_between(pred_dynamic_ci.index,
					pred_dynamic_ci.iloc[:, 0],
					pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

	ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(mid_data_point), logged_bar_data.index[-1],
					 alpha=.1, zorder=-1)

	ax.set_xlabel('Date')
	ax.set_ylabel('Y')

	plt.legend()
	plt.show()

#list_of_lags=list(range(1,5)
def make_lags(list_of_lags,y_serie):
	data=pd.DataFrame()
	
	if isinstance(list_of_lags, list):
		pass
	else:#in this case it's just a number, so you must make the range
		
		list_of_lags=list(range(1,list_of_lags))
	
	for each in list_of_lags:
		lag=y_serie.shift(each)
		lag=lag.rename(y_serie.name+"_lag_%02d"%each)  #if you put in more lags like over 100, then you should increase your 02 to 003
		data=pd.concat([data,lag],axis=1)
	data.dropna(inplace=True)
	return data
	
	
#This is to put it back into its original measurement.
def undo_differencing(original_y,prediction):
	original_y=pd.DataFrame(original_y)
	prediction=pd.DataFrame(prediction)
	answer = pd.Series(original_y.iloc[0].values, index=original_y.index)   

	combined=pd.concat([answer,prediction.cumsum()],axis=1).fillna(0)

	combined=combined.sum(axis=1)

	return combined
	
def penn2morphy(penntag, returnNone=False):
	morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
				   'VB':wn.VERB, 'RB':wn.ADV}
	try:
		return morphy_tag[penntag[:2]]
	except:
		return None if returnNone else ''

def my_lemmatize(text_processed): #expects list of strings
	lemmatized_words=[]
	for word, tag in pos_tag(text_processed):
		wntag = penn2morphy(tag)
		if wntag:
			lemmatized_words.append(wnl.lemmatize(word, pos=wntag))
		else:
			lemmatized_words.append(word)
	return lemmatized_words
	
def text_process(text):

	# tokenizing
	tokenizer = RegexpTokenizer(r'\w+') #RegexpTokenizer is to tokenize according to a regex pattern instead of just " "
	text_processed=tokenizer.tokenize(text)

	# removing any stopwords														stopwords.words('english')
	text_processed = [word.lower() for word in text_processed if word.lower() not in STOPWORDS and len(word)>1]

	#BIGRAM #messes up over here
	bigram = Phrases(text_processed, min_count=20) 
	
	bigram_phraser = gensim.models.phrases.Phraser(bigram)
	text_processed=bigram_phraser[text_processed]#this takes in ['word1','word2','etc']
	#text_processed = [bigram[text] for text in text_processed]

	#BEFORE I USED TO DO THIS FOR BIGRAMS, BUT APPARENTLY PHRASER IS BETTER
	#bigram = Phrases(text_processed, min_count=20) 
	#text_processed=bigram[text_processed]#this takes in ['word1','word2','etc']
	#text_processed = [bigram[text] for text in text_processed]
	
	
	
	
	# Lemmatizing
	#need to lemmatize according to each words tag. So if you're dealing with a verb, lemmatize to get the "base" verb
	#ex: running --> run   if you don't specify a tag, it assumes everything is a noun
	# n for noun files, v for verb files, a for adjective files, r for adverb files.
	
	#print(text_processed)
	text_processed = my_lemmatize(text_processed)#expects list of strings

	return " ".join(text_processed).split()


class text_process_gen(object):
	def __init__(self, texts):
		self.texts = texts		
	def __iter__(self):
		print(self.texts.shape)
		
		for text in self.texts:
			
			tokenized_text=text_process(text)
			empty=[]
			if tokenized_text == empty:
				tokenized_text=[" "]
			yield tokenized_text
			


def indexes_in_common(df1,df2):

	dates_in_common=np.intersect1d(df1.index.values, df2.index.values)

	df1=df1.loc[dates_in_common,:]
	df2=df2.loc[dates_in_common]
	
	return df1, df2
	
	
	
	
	
	
def read_emails(fname):
	with zipfile.ZipFile('data/'+fname+'.zip') as z:
		with z.open(fname) as f:
			for line in f:
				processed_line = gensim.utils.to_unicode(line, 'latin1').strip()
				if ":" not in processed_line: #a lot of the unimportant stuff, like email headers and stuff show up with :
					yield processed_line
					
class read_emails_iterator(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		for text in read_emails(self.fname):
			self.tokenize(text) #notice here we just call the tokenize function defined in the class
			
	def tokenize(self,text):
		tokenized_text= list(gensim.utils.tokenize(text, lower=True)) #gotta put list()here or you get generator objects
		tokenized_text=[text for text in tokenized_text if text not in STOPWORDS and len(text)>1]
		empty=[]
		if tokenized_text != empty:
			yield tokenized_text
		
