import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint

import sys
sys.path.extend(["../"])
from core.tools.data_import import *
from core.tools.time_series import *
from core.tools.visualize import *
from core.models.baseline_rnn import *
from constants import *


# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 90

df = load_dataset(
    "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/DEXCAUS.csv")
prepared_df = differencing(df, periods=PERIODS, order=ORDER)
prepared_df.head()
prepared_df.dropna(inplace=True)

TRAIN_RATIO = 0.9
# Normalize the sequence
scaler = StandardScaler().fit(prepared_df[:int(TRAIN_RATIO*len(prepared_df))].values)
prepared_df["DEXCAUS_period1_order1"] = scaler.transform(prepared_df.values)

X_raw, y_raw = gen_supervised_sequence(
    prepared_df, LAGS, prepared_df.columns[0], sequential_label=False)

(X_train, X_test, y_train, y_test) = train_test_split(
    X_raw, y_raw,
    test_size=1 - TRAIN_RATIO,
    shuffle=False)

(X_train, X_val, y_train, y_val) = train_test_split(
    X_train, y_train,
    test_size=0.1,
    shuffle=False)


op = lambda x: x.reshape(-1, 1)
y_train = op(y_train)
y_test = op(y_test)
y_val = op(y_val)

print(f"Training and testing set generated,\
\nX_train shape: {X_train.shape}\
\ny_train shape: {y_train.shape}\
\nX_test shape: {X_test.shape}\
\ny_test shape: {y_test.shape}\
\nX_validation shape: {X_val.shape}\
\ny_validation shape: {y_val.shape}")

# Model Parameters
num_time_steps = LAGS
# Number of series used to predict. (including concurrent)
num_inputs = 1
num_outputs = 1
num_neurons = 64
# Number of output series
learning_rate = 0.1
epochs = 100
# Training Settings
report_periods = epochs // 10

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_outputs])

cell = tf.contrib.rnn.LSTMCell(
    num_units=num_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_output = tf.reshape(rnn_outputs, [-1, num_time_steps * num_neurons])

W = tf.Variable(tf.random_normal([num_time_steps * num_neurons, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

pred = tf.add(tf.matmul(stacked_output, W), b)

# pred = tf.layers.dense(stacked_output, 1)

loss = tf.losses.mean_squared_error(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

hist = {"train": [], "val": []}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(500):
        sess.run(train, feed_dict={X: X_train, y: y_train})
        train_mse = loss.eval(feed_dict={X: X_train, y: y_train})
        val_mse = loss.eval(feed_dict={X: X_val, y: y_val})
        hist["train"].append(train_mse)
        hist["val"].append(val_mse)
        if e % report_periods == 0:
            print(
                f"\nIteration [{e}], Training MSE {train_mse:0.7f}; Validation MSE {val_mse:0.7f}")

    p_train = pred.eval(feed_dict={X: X_train})
    p_test = pred.eval(feed_dict={X: X_test})
    p_val = pred.eval(feed_dict={X: X_val})


plt.close()
plt.figure(figsize=(32, 16))
plt.plot(p_train.reshape(-1, 1), alpha=0.6)
plt.plot(y_train.reshape(-1, 1), alpha=0.6)
plt.legend(["Training Prediction", "Training Actual"])
plt.grid(True)
plt.show()

plt.close()
plt.figure(figsize=(32, 16))
plt.plot(p_test.reshape(-1, 1), alpha=0.6)
plt.plot(y_test.reshape(-1, 1), alpha=0.6)
plt.legend(["Testing Prediction", "Testing Actual"])
plt.grid(True)
plt.show()

plt.close()
plt.figure(size=(32, 16))
plt.plot(np.log(hist["train"]))
plt.plot(np.log(hist["val"]))
plt.legend(["Training Loss", "Validation Loss"])
plt.grid(True)
plt.show()
