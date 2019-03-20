# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:48:34 2019

@author: YASH SAINI
"""

import numpy as np
import quandl
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web

start=datetime.datetime(2012,1,1)
end=datetime.datetime(2017,1,1)
#Reading data 
df=web.DataReader('GOOG','yahoo',start,end)
train_df=df.iloc[:,2:3].values
'''
train_df=df.iloc[0:100,0:]['Open']
train_df=np.array(train_df)
train_df=np.reshape(train_df,(-1,1))'''


 
# Normalize
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(train_df)

#Creating DS with 60 timestamps i.e 3 months and 1 output
''' Use 60 days previous prices to predict present prices'''
X_train=[]
Y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])

X_train,Y_train=np.array(X_train),np.array(Y_train)

# Restructure the data
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

''' RNN Model '''

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

reg=Sequential() #Continuous Value
# 50 neurons to be there

#1st layer
reg.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
reg.add(Dropout(0.2))

#2nd layer
reg.add(LSTM(units=50,return_sequences=True))
reg.add(Dropout(0.2))

#3rd Layer
reg.add(LSTM(units=50,return_sequences=True))
reg.add(Dropout(0.2))

#4th Layer
reg.add(LSTM(units=50,return_sequences=False))
reg.add(Dropout(0.2))

#Output
reg.add(Dense(units=1))

#compiling RNN
reg.compile(optimizer='adam',loss='mean_squared_error')

#Fit
reg.fit(X_train,Y_train,batch_size=32,epochs=100)

''' Test'''

#Test and make predictions
#real stock price in january 2017
start=datetime.datetime(2017,1,1)
end=datetime.datetime(2017,1,31)
#Reading data 
df1=web.DataReader('GOOG','yahoo',start,end)
test_df=df1.iloc[:,2:3].values

#Predicted data
dt=pd.concat((df['Open'],df1['Open']),axis=0)
inp=dt[len(dt)-len(df1)-60:].values
inp=np.reshape(inp,(-1,1))
inp=sc.transform(inp)

X_test=[]
for i in range(60,80):
    X_test.append(inp[i-60:i,0])
   

X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pred_price=reg.predict(X_test)
pred_price=sc.inverse_transform(pred_price)

''' Plotting the prices'''
plt.plot(test_df,color='red',label="Real Stock Price")
plt.plot(pred_price,color='blue',label="Predicted Stock Price")
plt.title("GOOGLE STOCK PRICE PREDICTION")
plt.xlabel("Time") 
plt.ylabel("Price")
plt.legend()
plt.figure(figsize=(10,8))
plt.show()
