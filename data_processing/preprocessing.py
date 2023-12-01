# Standard libraries
import matplotlib.pyplot as plt

import numpy as np
import os
from time import time

# ML libraries
import tensorflow as tf
import keras
from keras.layers.core import *
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn import metrics

# Define some constants
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
N_BANDS = 6
N_TIMESTEPS = 3
BATCH_SIZE = 4096
N_EPOCHS = 25
HEIGHT = 3660
WIDTH = 3660

# Fix random seed for reproducibility
np.random.seed(42)




"""
https://github.com/geoaigroup/challenges/blob/main/ai4foodsecurity-challenge/lstm-cnn.ipynb

The data are stored as numpy arrays with dimension height x width x bands x timesteps. 
All of the reflectance values are in the range [0,1]. We also add 
two spectral indices (NDWI and LSWI) and SAR bands (VV, VH, incidence angle/IA).
"""

def add_lswi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, N_TIMESTEPS])
    # copy the values from the original array
    for i in range(X.shape[2]):
        _X[:,:,i,:] = X[:,:,i,:]
    # calculate values for LSWI channel
    for i in range(N_TIMESTEPS):
        lswi = (X[:,:,NIR,i]-X[:,:,SWIR1,i])/(X[:,:,NIR,i]+X[:,:,SWIR1,i])
        _X[:,:,-1,i] = lswi
    # make sure we didn't introduce any NaNs
    _X[np.where(np.isnan(_X))] = 0
    return _X

def add_ndwi_channel(X):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, N_TIMESTEPS])
    # copy the values from the original array
    for i in range(X.shape[2]):
        _X[:,:,i,:] = X[:,:,i,:]
    # calculate values for NDWI channel
    for i in range(N_TIMESTEPS):
        ndwi = (X[:,:,GREEN,i]-X[:,:,SWIR1,i])/(X[:,:,GREEN,i]+X[:,:,SWIR1,i])
        _X[:,:,-1,i] = ndwi
    # make sure we didn't introduce any NaNs
    _X[np.where(np.isnan(_X))] = 0
    return _X

def add_sar_channel(X, band, path):
    _X = np.ndarray([HEIGHT, WIDTH, X.shape[2]+1, N_TIMESTEPS])
    # copy the values from the original array
    for i in range(X.shape[2]):
        _X[:,:,i,:] = X[:,:,i,:]
    # load the corresponding SAR band
    if band=='vv' or band=='VV':
        sarpath = path.replace('pheno_timeseries', 'vv_timeseries')
        #sarpath = path.replace('fixed_timeseries', 'vv_timeseries')
    elif band=='vh' or band=='VH':
        sarpath = path.replace('pheno_timeseries', 'vh_timeseries')
        #sarpath = path.replace('fixed_timeseries', 'vh_timeseries')
    elif band=='ia' or band=='IA':
        sarpath = path.replace('pheno_timeseries', 'ia_timeseries')
        #sarpath = path.replace('fixed_timeseries', 'ia_timeseries')
    sar = np.load(sarpath).astype(np.float32)
    for i in range(N_TIMESTEPS):
        _X[:,:,-1,i] = sar[...,i]
    # make sure we didn't introduce any NaNs
    _X[np.where(np.isnan(_X))] = 0
    return _X

def load_data(x_path, y_path, flatten=True, convert_nans=True):
    # Load the time series image data
    X = np.load(x_path).astype(np.float32)
    # Load the associated labels
    Y = np.load(y_path).astype(np.int8)
    
    # Convert all the NaNs to zeros
    if convert_nans:
        X[np.where(np.isnan(X))] = 0
        
    X[np.where(X==0)] = 0.00000001
    # Add band indices
    X = add_lswi_channel(X)
    X = add_ndwi_channel(X)
    X = add_sar_channel(X, 'vv', x_path)
    X = add_sar_channel(X, 'vh', x_path)
    X = add_sar_channel(X, 'ia', x_path)
    if flatten:
        # Reduce the h x w x b x t dataset to h*w x b x t
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2], X.shape[3]))
        Y = np.reshape(Y, (Y.shape[0]*Y.shape[1]))
    assert X.shape[0] == Y.shape[0] 
    return X, Y 