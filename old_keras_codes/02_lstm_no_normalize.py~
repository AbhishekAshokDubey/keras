#!/usr/bin/env python2.7

import numpy as np
import math
#import pandas as pd
#import sys

datafile_path = r"/home/abhishek/Desktop/sensor_time_data.csv";
time_col_number = 0;
output_col_number = 2; # ignoring the date col number
output_col_number = 9;
train_fraction = 0.8;


#input = pd.read_csv(datafile_path, skipinitialspace=True, parse_dates=[0], infer_datetime_format=True);
#input = np.genfromtxt(datafile_path,dtype=float,delimiter = ',',names = True)
input = np.genfromtxt(datafile_path,dtype=float,delimiter = ',', skip_header=1)

time_col = input[:,time_col_number] # store date column separately
input = np.delete(input,np.s_[time_col_number],1) # remove date column for analysis

output = input[:,output_col_number] # output_col_number after removing date as output
input = np.delete(input,np.s_[output_col_number],1)

train_index = int(input.shape[0] * train_fraction)

train_x = input[:train_index,]
train_y = output[:train_index,]

test_x = input[train_index:,]
test_y = output[train_index:,]
'''
train_x_mean = train_x.mean(0)
train_x_std = train_x.std(0)
train_y_mean = train_y.mean(0)
train_y_std = train_y.std(0)

input_data = np.nan_to_num(np.divide((train_x - train_x_mean),train_x_std))
output_data = np.nan_to_num(np.divide((train_y - train_y_mean),train_y_std))

test_input_data = np.nan_to_num(np.divide((test_x - train_x_mean),train_x_std))
'''
input_data = train_x
output_data = train_y
test_input_data = test_x

############################# recurrent data #############################################
time_delay = 10;
X_train_rec = np.zeros(shape=(input_data.shape[0] - time_delay + 1,time_delay,input_data.shape[1]));
Y_train_rec = np.zeros(shape=(input_data.shape[0] - time_delay + 1,1));
temp = [];
for i in range(input_data.shape[0] - time_delay + 1):
    temp = input_data[i:i+time_delay];
    X_train_rec[i] = temp.reshape(time_delay,input_data.shape[1])
    Y_train_rec[i] = output_data[i]

X_test_rec = np.zeros(shape=(test_input_data.shape[0] - time_delay + 1,time_delay,test_input_data.shape[1]));
temp = [];
for i in range(test_input_data.shape[0] - time_delay + 1):
    temp = input_data[i:i+time_delay];
    X_test_rec[i] = temp.reshape(time_delay,test_input_data.shape[1])
############################################################################################


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(output_dim=12, batch_input_shape=(None,time_delay, input_data.shape[1]), activation='tanh', inner_activation='tanh'))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=6))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X_train_rec, Y_train_rec, nb_epoch=10000, batch_size=1000)
#score = model.evaluate(X_test, y_test)

norm_predictions = model.predict(X_test_rec)
#predictions = (norm_predictions * train_y_std) + train_y_mean;
predictions = norm_predictions

output_save = np.concatenate((test_y.reshape(test_y.shape[0],1)[0+time_delay-1:], predictions), axis=1)
np.savetxt("output_sensor_time_date_no_norm.csv", output_save, delimiter=",")
