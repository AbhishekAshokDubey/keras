# -*- coding: utf-8 -*-
import sys
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import pandas as pd

from dl_util import *

os.chdir(r"path")
empty_symbol = "-"

#if len(sys.argv) - 1 == 5:
#    train_file, test_file, input_features, output_response, layer_config, layer_type, layer_actv = sys.argv

###################################################################
# subtitue values for the command line parameters
train_file = r"C:\Users\Adubey4\Desktop\smartsignal\data\data.csv";
test_file = empty_symbol;
train_test_split_ratio = 0.75
input_features = ["f1", "f2", ..., "f3"]
output_response = ["y1"]
layer_config = [10, 6, 4, 1]
layer_type = ["LSTM", "LSTM", "LSTM",""] # RNN, LSTM, GRU, Dense = ""
layer_type = ["", "", "",""] # RNN, LSTM, GRU, Dense = ""
layer_actv = ["", "tanh", "sigmoid", ""] # tanh, sigmoid
drop_out = [-1,0.5,0.5,-1]
window_size = 10
return_seq = []
nb_epoch = 2

###################################################################
# read data
if input_features == "-":
    df_train = pd.read_csv(train_file)
    col_names = list(df_train.columns)
    input_features = col_names.remove(output_response[0])
else:
    df_train = pd.read_csv(train_file, usecols = input_features+output_response)

if test_file != empty_symbol:
    df_test = pd.read_csv(test_file)
else:
    df_test = df_train.iloc[int(df_train.shape[0]*train_test_split_ratio):,:]
    df_train = df_train.iloc[:int(df_train.shape[0]*train_test_split_ratio),:]


####################################################################
# Filling the blank values for the parameters & generating other parameters
layer_type = ["Dense" if i=="" else i for i in layer_type]
layer_actv = ["sigmoid" if i=="" else i for i in layer_actv]

# Recurrent unit/ Time series domain
if len(return_seq) == 0 and layer_type[0]=="LSTM":
    return_seq = get_default_return_seq(layer_type)
    input_feature_shape = (window_size, df_train.shape[1] - len(output_response))
    train_X, train_Y = create_dataset(df_train[input_features], df_train[output_response], window_size=window_size)
    test_X, test_Y = create_dataset(df_test[input_features], df_test[output_response], window_size=window_size)
else:
    # normal iid data
    return_seq = len(layer_type)*[False]
    input_feature_shape = (None, df_train.shape[1] - len(output_response))
    train_X, train_Y = np.asarray(df_train[input_features]), np.asarray(df_train[output_response])
    test_X, test_Y = np.asarray(df_test[input_features]), np.asarray(df_test[output_response])
###################################################################


####################################################################
# Build & compile the model
model = Sequential()
layer_count = len(layer_config)
for i in range(layer_count):
    model.add(get_layer(unit_count = layer_config[i], ltype = layer_type[i], actv = layer_actv[i], input_shape = input_feature_shape, drop_out = drop_out[i],return_sequences=return_seq[i]))
    if drop_out[i] > 0 and drop_out[i] < 1:
        model.add(Dropout(drop_out[i]))
    input_shape = None;

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

model.fit(train_X,train_Y, nb_epoch=nb_epoch)

model.save("saved_model.h5")
model = [];
model = load_model('saved_model.h5')

pred = model.predict(test_X)


# TODO:
# multiple response variable
# outlier removal
# normalization
# classification
