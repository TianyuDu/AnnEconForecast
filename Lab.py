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
from core.models.stat_models import *
from core.tools.metrics import *
from constants import *

pprint(DATA_DIR)

# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 12

df = load_dataset(DATA_DIR["0"])
prepared_df = differencing(df, periods=PERIODS, order=ORDER)
prepared_df.head()
prepared_df.dropna(inplace=True)

prepared_df.head()

TRAIN_RATIO = 0.8

# Normalize the sequence
scaler = StandardScaler().fit(prepared_df[:int(TRAIN_RATIO*len(prepared_df))].values)
prepared_df.iloc[:,0] = scaler.transform(prepared_df.values)

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

print(f"Training and testing set generated,\nX_train shape: {X_train.shape}\ny_train shape: {y_train.shape}\nX_test shape: {X_test.shape}\ny_test shape: {y_test.shape}\nX_validation shape: {X_val.shape}\ny_validation shape: {y_val.shape}")

# Model Parameters
num_time_steps = LAGS
# Number of series used to predict. (including concurrent)
num_inputs = 1
num_outputs = 1
num_neurons = 64
# Number of output series
learning_rate = 0.01
epochs = 500
# Training Settings
report_periods = 1

# the graph
tf.reset_default_graph()

with tf.name_scope("Data_feed"):
    X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs], name="Predictor_X")
    y = tf.placeholder(tf.float32, [None, num_outputs], name="Label_y")

with tf.name_scope("RNN"):
    cell = tf.nn.rnn_cell.LSTMCell(
        num_units=num_neurons,
        name="LSTM_Cell")

    # multi_cell = tf.nn.rnn_cell.MultiRNNCell(
    #     [tf.nn.rnn_cell.LSTMCell(num_units=x)
    #     for x in [512, num_neurons]]
    # )

    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    stacked_output = tf.reshape(rnn_outputs, [-1, num_time_steps * num_neurons])

with tf.name_scope("Output_layer"):
    W = tf.Variable(tf.random_normal([num_time_steps * num_neurons, 1]), dtype=tf.float32, name="Weight")
    b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name="Bias")

    pred = tf.add(tf.matmul(stacked_output, W), b, name="Prediction")

    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("predictions", pred)

# pred = tf.layers.dense(stacked_output, 1)

with tf.name_scope("Metrics"):
    loss = tf.reduce_mean(tf.square(y - pred), name="mse")

    mape = tf.reduce_mean(tf.abs(tf.divide(y - pred, y)))
    tf.summary.scalar("mean_squared_error", loss)
    tf.summary.scalar("mean_absolute_percentage_error", mape)

with tf.name_scope("Train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_optimizer")
    # gvs = optimizer.compute_gradients(loss)
    # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
    # train = optimizer.apply_gradients(capped_gvs)
    train = optimizer.minimize(loss)


# X_batches = X_train[:-84].reshape(100, -1, num_time_steps, num_inputs)
# y_batches = y_train[:-84].reshape(100, -1, num_outputs)
# print(X_batches.shape)
# print(y_batches.shape)

tb_dir = "./tensorboard/test/1"

start = datetime.now()
hist = {"train": [], "val": []}
with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tb_dir + "/train")
    val_writer = tf.summary.FileWriter(tb_dir + "/validation")
    train_writer.add_graph(sess.graph)
    # val_writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
#         for X_batch, y_batch in zip(X_batches, y_batches):
#             sess.run(train, feed_dict={X: X_batch, y: y_batch})
        sess.run(train, feed_dict={X: X_train, y: y_train})
        train_mse = loss.eval(feed_dict={X: X_train, y: y_train})
        val_mse = loss.eval(feed_dict={X: X_val, y: y_val})
        hist["train"].append(train_mse)
        hist["val"].append(val_mse)
        if e % report_periods == 0:
            s_train = sess.run(merged_summary, feed_dict={X: X_train, y: y_train})
            s_val = sess.run(merged_summary, feed_dict={X: X_val, y:y_val})
            train_writer.add_summary(s_train, e)
            val_writer.add_summary(s_val, e)
            print(
                f"\nIteration [{e}], Training MSE {train_mse:0.7f}; Validation MSE {val_mse:0.7f}")

    # p_train = pred.eval(feed_dict={X: X_train})
    # p_test = pred.eval(feed_dict={X: X_test})
    # p_val = pred.eval(feed_dict={X: X_val})
print(f"Time taken for {epochs} epochs: ", datetime.now()-start)


# plt.close()
# plt.figure(figsize=(32, 16))
# plt.plot(p_train.reshape(-1, 1), alpha=0.6)
# plt.plot(y_train.reshape(-1, 1), alpha=0.6)
# plt.legend(["Training Prediction", "Training Actual"])
# plt.grid(True)
# plt.title("Training Set Result")
# plt.show()

# plt.close()
# plt.figure(figsize=(16, 8))
# plt.plot(p_train.reshape(-1, 1)[-100:], alpha=0.6)
# plt.plot(y_train.reshape(-1, 1)[-100:], alpha=0.6)
# plt.legend(["Training Prediction", "Training Actual"])
# plt.grid(True)
# plt.title("Training Set Result: last 100")
# plt.show()


# plt.close()
# plt.figure(figsize=(32, 16))
# plt.plot(p_test.reshape(-1, 1), alpha=0.6)
# plt.plot(y_test.reshape(-1, 1), alpha=0.6)
# plt.legend(["Testing Prediction", "Testing Actual"])
# plt.grid(True)
# plt.title("Testing Set Result")
# plt.show()

# plt.close()
# plt.figure(figsize=(16, 8))
# plt.plot(p_test.reshape(-1, 1)[-100:], alpha=0.6)
# plt.plot(y_test.reshape(-1, 1)[-100:], alpha=0.6)
# plt.legend(["Testing Prediction", "Testing Actual"])
# plt.grid(True)
# plt.title("Testing set Result: last 100")
# plt.show()

# plt.close()
# plt.figure(figsize=(16, 8))
# plt.plot(np.log(hist["train"][5:]))
# plt.plot(np.log(hist["val"][5:]))
# plt.legend(["Training Loss", "Validation Loss"])
# plt.grid(True)
# plt.show()
