import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df=quandl.get('WIKI/GOOGL')
df=df[[ 'Adj. Open' ,  'Adj. High' ,   'Adj. Low' ,  'Adj. Close' ,  'Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['PCT_CHANGE']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df=df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

forecast_column='Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out=int(math.ceil(0.01*len(df)))

df['label']=df[forecast_column].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)

forecast_set=clf.predict(X_lately)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
np.set_printoptions(threshold=np.inf)
print(forecast_set , accuracy , forecast_out )
df['Forecast']=np.nan

last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix + one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+ [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
