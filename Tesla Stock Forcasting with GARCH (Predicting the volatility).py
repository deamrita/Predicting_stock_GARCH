#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Read data
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


# In[2]:


#in practice do not supress these warnings, they carry important information about the status of your model
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning, ConvergenceWarning
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=HessianInversionWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# In[11]:


tickerSymbol = 'TSLA'
data = yf.Ticker(tickerSymbol)


# In[12]:


prices = data.history(start='2015-02-01', end='2022-04-29').Close
returns = 100*prices.pct_change().dropna()


# In[13]:


plt.figure(figsize=(10,4))
plt.plot(prices)
plt.ylabel('Prices', fontsize=20)


# In[14]:


#taking return to covert this to stationary
plt.figure(figsize=(10,4))
plt.plot(returns)
plt.ylabel('Return', fontsize=20)


# In[15]:


plot_pacf(returns**2)
plt.show()


# In[8]:


#Fit GARCH(3,0) model


# In[28]:


model = arch_model(returns, p=3, q=0)


# In[29]:


model_fit = model.fit()


# In[30]:


model_fit.summary()


# In[ ]:





# In[33]:


#Rolling Predictions


# In[34]:


rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


# In[35]:


rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])


# In[36]:


plt.figure(figsize=(10,4))
true, = plt.plot(returns[-365:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# # How to use the model

# In[37]:


train = returns
model = arch_model(train, p=3, q=0)
model_fit = model.fit(disp='off')


# In[38]:


pred = model_fit.forecast(horizon=7)
future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)


# In[39]:


plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Volatility Prediction - Next 7 Days', fontsize=20)


# In[ ]:




