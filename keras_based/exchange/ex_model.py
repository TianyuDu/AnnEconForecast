"""
Univariate Version
"""
import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint

import methods
from methods import *
import containers
from containers import *

config = {
    "batch_size": 1,
    "epoch": 10,
    "neuron": 128,
    "test_ratio": 0.2,
    "lag_for_sup": 48
}

# Load dataset.
series = load_dataset(
    dir="/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange/DEXCHUS.csv")

# Transform to stationary data to Delta 1
raw_values = series.values

sample_data = np.array([x ** 2 for x in range(10)]).reshape(-1,1)
sample_series = pd.Series(sample_data.reshape(-1,))

c = UnivariateContainer(sample_series)
# cont = UnivariateContainer(raw_values.reshape(-1,1))

# diff would have length = len(raw_value) - 1 as it's taking the gaps.
diff_values = difference(raw_values, lag=1)

# Transform
# Use the current gap of differencing to predict the next gap differencing.
sup = gen_sup_learning(diff_values, config["lag_for_sup"])
sup = sup.values

# Split Data Set
total_sample_size = len(sup)
test_size = int(total_sample_size * config["test_ratio"])
train_size = int(total_sample_size - test_size)

print(
    f"Total sample found {total_sample_size}, {test_size} will be used as test set."
)
train, test = sup[:-test_size], sup[-test_size:]
train2, test2 = sup[:train_size], sup[train_size:]

# Generate scaler and scaling datasets.
# scaler on input matrix(X) and output(y)
scaler_in, scaler_out, train_scaled, test_scaled = gen_scaler(
    train, test, tar_idx=0)

# Fit model
model = fit_lstm(
    train_scaled,
    batch_size=config["batch_size"],
    epoch=config["epoch"],
    neurons=config["neuron"]
)

keras.utils.print_summary(model)

# Reshape to the shape of input tensor to network.
# Also applying scaler on it.
# Then feed into the network and make predication.
train_reshaped_X, train_reshaped_y = reshape_and_split(
    train_scaled, tar_idx=0
)

test_reshaped_X, test_reshaped_y = reshape_and_split(
    test_scaled, tar_idx=0
)

np.all(train_reshaped_X == train_X_res)
np.all(train_reshaped_y == train_y_res)
np.all(test_reshaped_X == test_X_res)
np.all(test_reshaped_y == test_y_res)

# DIRECT output from model.
# Raw model output on training set.
model_out_train = model.predict(train_reshaped_X, batch_size=1)
# Raw model output on testing set.
model_out_test = model.predict(test_reshaped_X, batch_size=1)

# Reconstruct Direct Generated Data from model.
model_out_train = scaler_out.inverse_transform(model_out_train)
model_out_train = model_out_train.reshape(-1,)
model_out_test = scaler_out.inverse_transform(model_out_test)
model_out_test = model_out_test.reshape(-1,)


# Constructing prediction on Training data (Single Step)
train_pred = list()
for i in range(len(train_scaled)):
    X, y = train_scaled[i, 1:], train_scaled[i, 0]
    yhat = forecast_lstm(
        model,
        batch_size=1,
        X=X
    )
    yhat = invert_scale(scaler_out, X, yhat)
    yhat += raw_values[i-1]
    train_pred.append(yhat)

train_pred = np.squeeze(np.array(train_pred))

# Constructing prediction on Test data (Single Ste)
test_pred = list()
for i in range(len(test_scaled)):
    # Make one-step forecast
    X, y = test_scaled[i, 1:], test_scaled[i, 0]
    X, y = test_reshaped_X[i].reshape(-1), test_reshaped_y[i]
    yhat = forecast_lstm(
        model,
        batch_size=1,
        X=X)
    yhat = invert_scale(scaler_out, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    test_pred.append(yhat)

test_pred = np.squeeze(np.array(test_pred))

# Visualize
visualize(raw_values, train_pred, test_pred, dir="./figure/")
# plt.plot(train_pred, alpha=0.6, linewidth=0.6)
# plt.plot(test_pred, alpha=0.6, linewidth=0.6)
# plt.legend(["Raw", "TrainPred", "TestPred"])
# plt.show()
