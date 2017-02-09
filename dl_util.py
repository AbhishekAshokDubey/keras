# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 00:28:19 2017

@author: ADubey4
"""
from keras.layers import Dense, LSTM
import numpy as np

def get_default_return_seq(layer_type):
    return_seq = []
    for i in range(len(layer_type)-1):
        if (layer_type[i] == "LSTM") and (layer_type[i+1] == "LSTM"):
            return_seq.append(True)
        else:
            return_seq.append(False)
    return_seq.append(False)
    return return_seq


def get_layer(unit_count, ltype="Dense", input_shape=None, isfirst=False, islast=False, actv="sigmoid", return_sequences = False, drop_out = ""):
    if ltype == "Dense":
        if input_shape:
            return Dense(unit_count, batch_input_shape= input_shape, activation = actv)
        else:
            return Dense(unit_count, activation = actv)
    if ltype == "LSTM":
        if input_shape:
            return LSTM(unit_count, return_sequences=return_sequences, input_shape= input_shape)
        else:
            return LSTM(unit_count, return_sequences=return_sequences)

def create_dataset(X,Y, window_size=1):
    X = np.asarray(X)
    Y = np.asarray(Y)
    dataX, dataY = [], []
    for i in range(len(X)- window_size + 1):
        dataX.append(X[i:(i+window_size),:])
        dataY.append(Y[i+window_size-1])
    return np.array(dataX), np.array(dataY)
