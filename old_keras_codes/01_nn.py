####!/usr/bin/env python2.7

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

train_x_mean = train_x.mean(0)
train_x_std = train_x.std(0)
train_y_mean = train_y.mean(0)
train_y_std = train_y.std(0)

input_data = np.nan_to_num(np.divide((train_x - train_x_mean),train_x_std))
output_data = np.nan_to_num(np.divide((train_y - train_y_mean),train_y_std))

test_input_data = np.nan_to_num(np.divide((test_x - train_x_mean),train_x_std))

#########################################################################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l1, l2

model = Sequential()
model.add(Dense(12, input_dim=23, W_regularizer=l1(0.01)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(6, W_regularizer=l2(0.01)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(input_data, output_data, nb_epoch=100, batch_size=1000)
#score = model.evaluate(X_test, y_test)

norm_predictions = model.predict(test_input_data)
predictions = (norm_predictions * train_y_std) + train_y_mean;

output_save = np.concatenate((test_y.reshape(test_y.shape[0],1), predictions), axis=1)
np.savetxt("output_sensor_time_date.csv", output_save, delimiter=",")
