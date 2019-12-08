import pandas as pd
import numpy as np
import quandl
df=quandl.get('WIKI/GOOGL')
df=df[[ 'Adj. Open' ,  'Adj. High' ,   'Adj. Low' ,  'Adj. Close' ,  'Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['PCT_CHANGE']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
np.set_printoptions(threshold=np.inf)
res=df.head()
print(res)


