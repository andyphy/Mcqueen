
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

## loading data as DataFrame type
AAPL_F = pd.read_csv('../individual_stocks_5yr/AAPL_data.csv', index_col='Date')
AAPL_F.index = pd.to_datetime(AAPL_F.index)
SandP500_all_stocks = pd.read_csv('../Dataset/all_stocks_5yr.csv', index_col='Date')
SandP500_all_stocks.index = pd.to_datetime(SandP500_all_stocks.index)
SandP500_all_stocks.info(null_counts=True)


# In[3]:

## pick-up the null data.
GOOG = SandP500_all_stocks[SandP500_all_stocks['Name']=='GOOG']
GOOG_null = GOOG[GOOG['Volume'].isnull()]


# In[4]:

type(AAPL_F)


# In[5]:

SandP500_all_stocks.head()


# In[6]:

## 10 Selected Stock List
stock_list = ["AAPL", "FB", "AMZN", 'GOOGL', 'GOOG', 'NVDA', 'IBM', 'MSFT', 'INTC', 'MU']

## Use loop to select data
for temp in stock_list:
    ## conditional selection
    temp_stock_data = SandP500_all_stocks[SandP500_all_stocks['Name'] == temp] 
    if temp == stock_list[0]:
        stock_price_selected = temp_stock_data
    else:      
        stock_price_selected = stock_price_selected.append(temp_stock_data)

## check output dimension
stock_price_selected.shape


# In[7]:

## reshape the data by pivot function
close_pivot_data = pd.pivot_table(stock_price_selected, values="Close", 
                                 columns = "Name", index="Date")
volume_pivot_data = pd.pivot_table(stock_price_selected, values="Volume", 
                                 columns = "Name", index="Date")
open_pivot_data = pd.pivot_table(stock_price_selected, values="Open", 
                                 columns = "Name", index="Date")
high_pivot_data = pd.pivot_table(stock_price_selected, values="High", 
                                 columns = "Name", index="Date")
low_pivot_data = pd.pivot_table(stock_price_selected, values="Low", 
                                 columns = "Name", index="Date")


# In[8]:

close_pivot_data.plot(subplots=True, figsize=(16, 8), layout=(4, 3), title ="Close Price", sharex=True)


# In[9]:

volume_pivot_data.plot(subplots=True, figsize=(16, 8), layout=(4, 3), title ="Volume", sharex=True)


# In[10]:

## plot candlestick chart


# In[11]:

## Compute Summary Statistic
close_stat_summary = close_pivot_data.describe()
volume_stat_summary = volume_pivot_data.describe()
open_stat_summary = open_pivot_data.describe()
high_stat_summary = high_pivot_data.describe()
low_stat_summary = low_pivot_data.describe()
close_stat_summary


# In[12]:

## ma, EMA, BBand
AAPL = SandP500_all_stocks[SandP500_all_stocks['Name'] == 'AAPL'][:]  # last [:] can reomve warning of "a value is trying to be copy of a slice from a dataframe"
AAPL["20ma"] = AAPL['Close'].rolling(window=20, win_type='boxcar').mean()
AAPL["20ema"] = AAPL['Close'].ewm(span=20).mean()
AAPL["std"] = AAPL['Close'].rolling(window=20).std()
AAPL["Upper Band"] = AAPL['20ema'] + 2*AAPL['std']
AAPL["Lower Band"] = AAPL['20ema'] - 2*AAPL['std']
AAPL_BBand = AAPL[0:200].plot(y=['Close', '20ma', 'Upper Band', 'Lower Band', '20ema'],figsize=(12,8), title='Bollinger Bands')


# In[13]:

## NVDA Stock Close Price
NVDA = SandP500_all_stocks[SandP500_all_stocks['Name'] == 'NVDA'][:]
#NVDA_Close = NVDA['Close'].plot(figsize=(20,12), title="NVDA Close Stock Price")
## NVDA Time Series Analysis
NVDA['First Difference'] = NVDA['Close'] - NVDA['Close'].shift()
NVDA['1st Diff MA'] = NVDA['First Difference'].rolling(window=20).mean()
NVDA['1st Diff Std'] = NVDA['First Difference'].rolling(window=20).std()
NVDA_TS_analysis = NVDA.plot(y=['First Difference', '1st Diff MA', '1st Diff Std'], figsize=(12, 8))


# In[14]:

## NVDA close price of log scale
NVDA['Natural Log'] = NVDA['Close'].apply(lambda x:np.log(x))
NVDA['First Log Difference'] = NVDA['Natural Log'] - NVDA['Natural Log'].shift()
NVDA['Log 1st Diff MA'] = NVDA['First Log Difference'].rolling(window=20).mean()
NVDA['Log 1st Diff Std'] = NVDA['First Log Difference'].rolling(window=20).std()

NVDA_Log_TS_analysis = NVDA.plot(y=['First Log Difference', 'Log 1st Diff MA', 'Log 1st Diff Std'], figsize=(12, 8))

'''
# plot normal & natural log
fig, axes = plt.subplots(1, 2, figsize=(18,6))
      
axes[0].plot(NVDA['Close'])
axes[0].set_title("NVDA Close - Normal scale")

axes[1].plot(NVDA['Natural Log'])
axes[1].set_yscale("log")
axes[1].set_title("NVDA Close - Logarithmic scale (y)");
'''



# In[15]:

## ETS decomposition for NVDA

from statsmodels.tsa.seasonal import seasonal_decompose ## ignore warning, will disappear excute twice 
## index must be datetime type 

NVDA_decomposition = seasonal_decompose(NVDA['Natural Log'], model='multiplacative', freq=20)  
trend = NVDA_decomposition.trend
seasonal = NVDA_decomposition.seasonal
residual = NVDA_decomposition.resid

NVDA_ETS_Multi, axes = plt.subplots(4, 1, figsize=(12, 8))
axes[0].plot(NVDA['Natural Log'])
axes[0].set(ylabel='Observed:log')
axes[1].plot(trend)
axes[1].set(ylabel='Trend')
axes[2].plot(seasonal)
axes[2].set(ylabel='Seasonal')
axes[3].plot(residual)
axes[3].set(ylabel='Residual')
plt.tight_layout()


# In[16]:

NVDA_decomposition_add = seasonal_decompose(NVDA['Natural Log'], model='additive', freq=20)  
trend = NVDA_decomposition_add.trend
seasonal = NVDA_decomposition_add.seasonal
residual = NVDA_decomposition_add.resid

NVDA_ETS_Add, axes = plt.subplots(4, 1, figsize=(12, 8))
axes[0].plot(NVDA['Natural Log'])
axes[0].set(ylabel='Observed:log')
axes[1].plot(trend)
axes[1].set(ylabel='Trend')
axes[2].plot(seasonal)
axes[2].set(ylabel='Seasonal')
axes[3].plot(residual)
axes[3].set(ylabel='Residual')
plt.tight_layout()

