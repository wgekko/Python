#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 02:36:12 2021

@author: walter
"""

# descripcion : este es un programa que usa redes neuronales recurrentes
# llamada Long short term memory (LSTM)
# para predecir el cierre de acciones price usando datos de 60 dias

# iportantod librerias

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential 
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Using TensorFlow backend.
# los dastos del precio de la accion 

#df=web.DataReader('AMD', data_source='yahoo', start='2019-01-05', end ='2021-12-21' )
df=pd.read_csv('pred_BTC-USD-2021.csv', index_col='Date', parse_dates=['Date'])

plt.figure(figsize=(16,8))
plt.title('precio de cierre historico')
plt.plot(df['Close'])
plt.xlabel('Dias', fontsize=18)
plt.ylabel('precio cierre', fontsize=18)
plt.show()

data= df.filter(['Close'])
dataset= data.values
training_data_len= math.ceil(len(dataset)* .8) 
training_data_len             

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

train_data= scaled_data[0: training_data_len, : ]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()
        
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_train.shape
                     
# desarrollando el modelo de LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

# compilar el modelo 
model.compile(optimizer='adam', loss='mean_squared_error')

# entranar el modelo 
model.fit(x_train, y_train, batch_size=1, epochs=1)

# creamos el test de los datos 
# creamos un nuevo array para datos de 1543 a 2003
test_data = scaled_data[training_data_len -60, :]
x_test= []
y_test = dataset[training_data_len:, : ]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

# convirtiendo los datos a numpy array
x_text = np.array(x_test)
# reshape los datos
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# tomamos los valores del modelo de prediccion
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

# gerenado los plot de los datos

train = data[:training_data_len]
valid = data[training_data_len]
valid['prediccion']

# visualizacion 

plt.figure(figsize=(16,8))
plt.title('Modelo')
plt.xlabel('Dias')
plt.ylabel('Px  cierre', fontsize=18)
plt.plot(train['Cierre'])
plt.plot(valid[['cierre', 'Prediccion']])
plt.legend(['tendencia', 'valor', 'Prediciones'], loc='lower right')
plt.show()

# mostramos la validacion y prediccion de precios 
valid

# tomamos los precios de la accion , creamos un nuevo area de trabajo
amd_accion = web.DataReader('AMD', data_source='yahoo', start='2019-01-05', end='2021-12-21' )
# 
new_df = amd_accion.filter(['Close'])
# aqui consideramos los 60 dias de cierre y los convertimos el area de trabajo en array
last_60_days= new_df[-60:].values 
last_60_days_scaled = scaler.transform(last_60_days)
# creamos una lista vacia
X_test = []
X_test.append(last_60_days_scaled)
# convertimos el X_test datos en un numpy array
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


amd_accion1 = web.DataReader('AMD', data_source='yahoo', start='2021-12-22', end='2021-12-22' )
print(amd_accion1['Close'])



