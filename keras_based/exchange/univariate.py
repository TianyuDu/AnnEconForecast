"""
The univariate baseline version of exchange rate forecasting.
"""
from typing import Tuple

import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import config
import containers
import methods
from containers import *
from methods import *

# Configuration files

file_dir = "/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange/DEXCHUS.csv"
print(f"Loading CSV file from {file_dir}")
series = methods.load_dataset(dir=file_dir)

container = containers.UnivariateContainer(
    series,
    config=config.data_proc_config
)

# TODO: get_combined_scaled should be removed 
# and will be called directly with UnivariateLSTM object.
train_scaled, test_scaled = container.get_combined_scaled()

model = fit_lstm(
    train_scaled,
    batch_size=config.neural_network_config["batch_size"],
    neurons=config.neural_network_config["neuron"],
    epoch=config.neural_network_config["epoch"]
)

# Reshape data

train_X_res, train_y_res, test_X_res, test_y_res = methods.reshape_data(container)

keras.utils.print_summary(model)

# Make single step prediction on training set.
train_pred = list()
for i in range(container.train_size):
    X, y = container.train_X_scaled[i], container.train_y_scaled[i]
    yhat = methods.forecast_lstm(model, 1, X)
    yhat = np.array([yhat])
    yhat = container.scaler_out.inverse_transform(yhat)
    # yhat = container._invert_difference(yhat, idx=i)
    train_pred.append(yhat)

train_pred = np.squeeze(np.array(train_pred))

test_pred = list()
for i in range(container.test_size):
    # Test index: index in test set
    # global_idx = container.train_size + test_idx
    X, y = container.test_X_scaled[i], container.test_y_scaled[i]
    yhat = forecast_lstm(model, 1, X)
    yhat = np.array([yhat])
    yhat = container.scaler_out.inverse_transform(yhat)
    # yhat = container._invert_difference(yhat, idx=global_idx)
    test_pred.append(yhat)

test_pred = np.squeeze(np.array(test_pred))

pred = [None] * container.num_obs
pred = np.array(pred).astype(np.float32)

for (i, yhat) in zip(range(container.train_size), train_pred):
    # i is the index of differenced value in training set.
    yhat = container._invert_difference(yhat, idx=i)
    pred[i + 1] = yhat

for (i, yhat) in zip(range(container.test_size), test_pred):
    global_i = container.train_size + i
    yhat = container._invert_difference(yhat, idx=global_i)
    pred[global_i + 1] = yhat

methods.uni_visualize(
    actual_val=container.raw,
    pred_val=pred,
    break_point=container.train_size
)

# methods.visualize3(container.raw, train_pred, test_pred, dir=None)
