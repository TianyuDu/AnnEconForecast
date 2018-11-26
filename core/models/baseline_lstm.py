"""
This file contains the baseline LSTM model for time series forecasting.
"""
import sys
from pprint import pprint
from typing import Dict, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from core.models.baseline_rnn import *
from core.models.stat_models import *
from core.tools.data_import import *
from core.tools.metrics import *
from core.tools.time_series import *
from core.tools.visualize import *

sys.path.extend(["../"])

def prepare_dataset(
    file_dir: str,
    periods: int=1,
    order: int=1
) -> pd.DataFrame:
    df = load_dataset(file_dir)
    prepared_df = differencing(df, periods=periods, order=order)
    prepared_df.head()
    prepared_df.dropna(inplace=True)

    print("First few rows of dataset loaded:")
    print(prepared_df.head())
    return prepared_df


# Normalize the sequence
def normalize(
    df: pd.DataFrame,
    train_ratio: float,
    lags: int
) -> Tuple[np.ndarray]:
    scaler = StandardScaler().fit(
        df[:int(train_ratio*len(df))].values)
    df.iloc[:, 0] = scaler.transform(df.values)

    X_raw, y_raw = gen_supervised_sequence(
        df, lags, df.columns[0], sequential_label=False)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X_raw, y_raw,
        test_size=1 - train_ratio,
        shuffle=False)

    (X_train, X_val, y_train, y_val) = train_test_split(
        X_train, y_train,
        test_size=0.1,
        shuffle=False
    )

    def trans(x): return x.reshape(-1, 1)

    y_train = trans(y_train)
    y_test = trans(y_test)
    y_val = trans(y_val)

    print(
        f"Training and testing set generated,\
        \nX_train shape: {X_train.shape}\
        \ny_train shape: {y_train.shape}\
        \nX_test shape: {X_test.shape}\
        \ny_test shape: {y_test.shape}\
        \nX_validation shape: {X_val.shape}\
        \ny_validation shape: {y_val.shape}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )


def exec_core(
    epochs: int,
    num_time_steps: float,
    num_inputs: int,
    num_outputs: int,
    num_neurons: int,
    learning_rate: float,
    tensorboard_dir: str,
    data_collection: Dict[str, np.ndarray],
    clip_grad: Union[bool, float]=None
) -> Dict[str, float]:
    print("Resetting Tensorflow defalut graph...")
    tf.reset_default_graph()

    for i in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        exec(f"{i} = data_collection['{i}']")

    with tf.name_scope("DATA_FEED"):
        X = tf.placeholder(
            tf.float32,
            [None, num_time_steps, num_inputs],
            name="Feature_X"
        )
        y = tf.placeholder(
            tf.float32,
            [None, num_outputs],
            name="Label_y"
        )

    with tf.name_scope("RECURRENT_UNITS"):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=num_neurons,
            name="LSTM_Cell"
        )

        # multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        #     [tf.nn.rnn_cell.LSTMCell(num_units=x)
        #     for x in [512, num_neurons]]
        # )

        rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        stacked_output = tf.reshape(
            rnn_outputs, [-1, num_time_steps * num_neurons])

    with tf.name_scope("OUTPUT_LAYER"):
        W = tf.Variable(tf.random_normal(
            [num_time_steps * num_neurons, 1]), dtype=tf.float32, name="Weight")
        b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name="Bias")

        pred = tf.add(tf.matmul(stacked_output, W), b, name="Prediction")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("predictions", pred)

    # pred = tf.layers.dense(stacked_output, 1)

    with tf.name_scope("METRICS"):
        loss = tf.reduce_mean(tf.square(y - pred), name="mse")

        mape = tf.reduce_mean(tf.abs(tf.divide(y - pred, y)))
        tf.summary.scalar("mean_squared_error", loss)
        tf.summary.scalar("mean_absolute_percentage_error", mape)

    with tf.name_scope("OPTIMIZER"):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name="Adam_optimizer")
        
        if clip_grad is None:
            print("Note: no gradient clipping is applied.\
            \nIf possible gradient exploding detected (e.g. nan loss), try use clip_grad.")
            train = optimizer.minimize(loss)
        else:
            print("Applying gradient clipping...")
            print(f"\tClip by values: {clip_grad}")
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [
                (tf.clip_by_value(grad, - clip_grad, clip_grad), var)
                for grad, var in gvs
            ]
            train = optimizer.apply_gradients(capped_gvs)


    # X_batches = X_train[:-84].reshape(100, -1, num_time_steps, num_inputs)
    # y_batches = y_train[:-84].reshape(100, -1, num_outputs)
    # print(X_batches.shape)
    # print(y_batches.shape)

    tb_dir = tensorboard_dir

    start = datetime.now()
    # hist = {"train": [], "val": []}
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
            # hist["train"].append(train_mse)
            # hist["val"].append(val_mse)
            if e % report_periods == 0:
                s_train = sess.run(merged_summary, feed_dict={
                                X: X_train, y: y_train})
                s_val = sess.run(merged_summary, feed_dict={X: X_val, y: y_val})
                train_writer.add_summary(s_train, e)
                val_writer.add_summary(s_val, e)
                print(
                    f"\nIteration [{e}], Training MSE {train_mse:0.7f}; Validation MSE {val_mse:0.7f}")

        # p_train = pred.eval(feed_dict={X: X_train})
        p_test = pred.eval(feed_dict={X: X_test})
        # p_val = pred.eval(feed_dict={X: X_val})
    print(f"Time taken for [{epochs}] epochs: ", datetime.now()-start)
    metric_test = merged_scores(actual=y_test, pred=p_test, verbose=True)
    return metric_test
