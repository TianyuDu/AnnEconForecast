"""
This file contains the stacked version LSTM model for time series forecasting.
"""
import sys
from pprint import pprint
from typing import Dict, Union, Iterable

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
from core.tools.rnn_prepare import *

sys.path.extend(["../"])


# ======== Test ========
sample_parameters = {
    "epochs": 250,
    "num_time_steps": 12,
    "num_inputs": 1,
    "num_outputs": 1,
    "num_neurons": (128, 64),
    "learning_rate": 0.01,
    "report_periods": 10,
    "tensorboard_dir": f"/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/tensorboard/{EXPERIMENT_NAME}",
    "model_path": f"/Users/tianyudu/Desktop/{EXPERIMENT_NAME}/my_model"
}
# ======== End ========
def exec_core(
    parameters: Dict[str, object],
    data_collection: Dict[str, np.ndarray],
    clip_grad: float = None,
    prediction_checkpoints: Iterable[int] = [-1]
) -> Tuple[
        Dict[str, float],
        Dict[int, Dict[str, np.ndarray]]
]:
    print("Resetting Tensorflow defalut graph...")
    tf.reset_default_graph()

    globals().update(parameters)
    globals().update(data_collection)

    predictions = dict()
    assert all(isinstance(x, int)
               for x in prediction_checkpoints), "Invalid checkpoint of recording."
    assert all(-1 <= x <=
               epochs for x in prediction_checkpoints), "Checkpoint out of range."

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

    # with tf.name_scope("RECURRENT_UNITS"):
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(
            num_units=x,
            name=f"LSTM_Cell_{i}"
            )
        for i, x in enumerate(num_neurons)]
    )

    rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
    stacked_output = tf.reshape(
        rnn_outputs, [-1, num_time_steps * num_neurons[-1]])

    with tf.name_scope("OUTPUT_LAYER"):
        W = tf.Variable(tf.random_normal(
            [num_time_steps * num_neurons[-1], 1]), dtype=tf.float32, name="Weight")
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
        saver = tf.train.Saver()
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
                s_val = sess.run(merged_summary, feed_dict={
                                 X: X_val, y: y_val})
                train_writer.add_summary(s_train, e)
                val_writer.add_summary(s_val, e)
            if e % (report_periods * 10) == 0:
                # print 10 times less frequently than the record frequency.
                print(
                    f"\nIteration [{e}], Training MSE {train_mse:0.7f}; Validation MSE {val_mse:0.7f}")
            if e in prediction_checkpoints:
                p_train = pred.eval(feed_dict={X: X_train})
                p_test = pred.eval(feed_dict={X: X_test})
                p_val = pred.eval(feed_dict={X: X_val})
                predictions[e] = {
                    "train": p_train,
                    "test": p_test,
                    "val": p_val
                }

        if -1 in prediction_checkpoints:
            p_train = pred.eval(feed_dict={X: X_train})
            p_test = pred.eval(feed_dict={X: X_test})
            p_val = pred.eval(feed_dict={X: X_val})
            predictions[epochs] = {
                "train": p_train,
                "test": p_test,
                "val": p_val
            }

        print("Saving the trained model...")
        saver.save(sess, model_path)
    print(f"Time taken for [{epochs}] epochs: ", datetime.now()-start)
    metric_test = merged_scores(
        actual=pd.DataFrame(y_test),
        pred=pd.DataFrame(p_test),
        verbose=True
    )
    return (metric_test, predictions)
