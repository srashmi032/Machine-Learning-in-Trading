import pandas as pd 
import matplotlib.pyplot as plt


def test():
	df=pd.read_csv("n225.csv")
	print (df.head(10))

	print (df[10:101])


def get_mean_vol(symbol):
	df=pd.read_csv("{}.csv".format(symbol))
	return df['Volume'].mean()
 
def get_max_close(symbol):
	df=pd.read_csv("{}.csv".format(symbol))
	return df['Close'].max()


def file():
 	for symbol in ['n225','bsesn','dji','hsi','ixic']:
 		print ("Max closing price")
 		print (symbol,get_max_close(symbol))

def mean_vol():
	for symbol in ['n225','bsesn','dji','hsi','ixic']:
 		print ("Mean Volume")
 		print (symbol,get_mean_vol(symbol))



def plot2():
	df=pd.read_csv("n225.csv")
	#print (df['Adj Close'])
	df[['Close','Adj Close']].plot()
	plt.show()


def plot1():
	df=pd.read_csv("n225.csv")
	print (df['Adj Close'])
	df['Adj Close'].plot()
	plt.show()


if __name__=="__main__":
	plot2()
	