"""
Tutorial: Multivariate Time Series Forecasting with LSTMs in Keras.
"""
import sys
from datetime import datetime

import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

sys.path.append("/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange")
import config
import containers
import methods


def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")

dataset = pd.read_csv(
    "./data/PRSA.csv",
    parse_dates=[["year", "month", "day", "hour"]],
    index_col=0,
    date_parser=parse)

# Drop number column, clean the data frame.
dataset = dataset.drop(columns=["No"])
dataset.columns = [
    "pollution", "dew", "temp", "press", 
    "wnd_dir", "wnd_spd", "snow", "rain"]
dataset.index.name = "date"
dataset["pollution"].fillna(0, inplace=True)

# Drop hr=0 to hr=23 (first 24 hrs.)
dataset = dataset[24:]
dataset.to_csv("./data/pollution.csv")
# Data cleaned, create new csv file to store the new data.

# load dataset
dataset = pd.read_csv("./data/pollution.csv", header=0, index_col=0, engine="c")
values = dataset.values

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group], linewidth=0.6, alpha=0.9)
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()


# LSTM Data Preparation
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input seq. (t-n, ..., t-1)
    # sup learning over n lagged vars.
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f"var{j+1, i}") for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]
    
    # put it all together.
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values.
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Load dataset
dataset = pd.read_csv("./data/pollution.csv", header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = sklearn.preprocessing.LabelEncoder()
# encode wind_direction
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype(np.float32)
# normalize
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1) 

reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)

# TODO: one-hot encoding for wind direction.
# TODO: Making all series stationary with differencing and seasonal adjustment


values = reframed.values
n_train_hours = 365*24

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# Pollution (target y) is the last column
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = keras.Sequential()
model.add(
    keras.layers.LSTM(
        units=50,
        input_shape=(train_X.shape[1], train_X.shape[2])
    )
)

model.add(keras.layers.Dense(1))
model.compile(loss="mae", optimizer="adam")

hist = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=1, shuffle=False)

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()

# Evaluate model.
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

test_y = test_y.reshape(len(test_y), 1)

inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

rmse = np.sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat))
print(f"RMSE={rmse}")

plt.plot(inv_yhat, alpha=0.6, linewidth=0.6, label="yhat")
plt.plot(inv_y, alpha=0.6, linewidth=0.6, label="y")
plt.Legend()
plt.show()

# multiple lagged time steps.
n_hrs = 3
n_fea = 8
reframed = series_to_supervised(scaled, n_hrs, 1)

# Inputs (X) would be in shape [samples, timesteps, features]
n_obs = n_hrs * n_fea
train_X, train_y = train[:, :n_obs], train[:, -n_fea]
test_X, test_y = test[:, :n_obs], test[:, -n_fea]
print(train_X.shape, len(train_X), train_y.shape)

# Reshape
train_X = train_X.reshape((train_X.shape[0], n_hrs, n_fea))
test_X = test_X.reshape((test_X.shape[0], n_hrs, n_fea))
