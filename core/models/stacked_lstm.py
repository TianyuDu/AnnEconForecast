"""
This file contains the stacked version LSTM model for time series forecasting.
"""
import sys
from pprint import pprint
from typing import Dict, Iterable, Union

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
import core.tools.metrics as metrics
from core.tools.rnn_prepare import *
from core.tools.time_series import *
from core.tools.visualize import *

# sys.path.extend(["/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast"])

def make_prediction_all(
    pred: tf.Tensor,
    X: tf.Tensor,
    data: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    p_train = pred.eval(feed_dict={X: data["X_train"]})
    p_test = pred.eval(feed_dict={X: data["X_test"]})
    p_val = pred.eval(feed_dict={X: data["X_val"]})
    collection = {
        "train": p_train,
        "test": p_test,
        "val": p_val
    }
    return collection


def exec_core(
    param: Dict[str, object],
    data: Dict[str, np.ndarray],
    prediction_checkpoints: Iterable[int]=[-1],
    verbose: bool=False
) -> Tuple[
        Dict[str, float],
        Dict[int, Dict[str, np.ndarray]]
]:
    """
    # TODO: write the doc string.
    """
    param["num_time_steps"] = param["LAGS"]
    if verbose:
        print("Resetting Tensorflow defalut graph...")
    tf.reset_default_graph()

    assert all(isinstance(x, int)
               for x in prediction_checkpoints), "Invalid checkpoint of recording."
    assert all(
        -1 <= x <= param["epochs"] for x in prediction_checkpoints), "Checkpoint out of range."

    with tf.name_scope("DATA_FEED"):
        X = tf.placeholder(
            tf.float32,
            [None, param["num_time_steps"], param["num_inputs"]],
            name="FEATURE"
        )
        y = tf.placeholder(
            tf.float32,
            [None, param["num_outputs"]],
            name="LABEL"
        )

    # with tf.name_scope("RECURRENT_UNITS"):
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(
            num_units=units,
            name=f"LSTM_CELL_{i}"
            )
        for i, units in enumerate(param["num_neurons"])
        ]
    )

    rnn_outputs, states = tf.nn.dynamic_rnn(
        multi_cell,
        X,
        dtype=tf.float32
    )

    stacked_output = tf.reshape(
        rnn_outputs,
        [-1, param["num_time_steps"] * param["num_neurons"][-1]]
    )

    with tf.name_scope("OUTPUT_LAYER"):
        W = tf.Variable(
            tf.random_normal(
                [param["num_time_steps"] * param["num_neurons"][-1], 1]
            ),
            dtype=tf.float32,
            name="WEIGHT"
        )

        b = tf.Variable(
            tf.random_normal([1]),
            dtype=tf.float32,
            name="BIAS"
        )

        pred = tf.add(
            tf.matmul(stacked_output, W), b,
            name="PREDICTION"
        )

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("predictions", pred)

    # pred = tf.layers.dense(stacked_output, 1)

    with tf.name_scope("METRICS"):
        # MSE is the main loss we focus on and it's the metric used for optimization.
        loss = tf.reduce_mean(tf.square(y - pred), name="MSE")
        mape = tf.reduce_mean(tf.abs(tf.divide(y - pred, y)))

        tf.summary.scalar("MSE", loss)
        tf.summary.scalar("MAPE", mape)

    with tf.name_scope("OPTIMIZER"):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=param["learning_rate"],
            name="ADAM_OPTIMIZER"
        )

        if param["clip_grad"] is None:
            # No Gradient Clipping
            if verbose:
                print("Note: no gradient clipping is applied.\
                \nIf possible gradient exploding detected (e.g. nan loss), try use clip_grad.")
            train = optimizer.minimize(loss)
        else:
            # With Gradient Clipping
            if verbose:
                print("Applying gradient clipping...")
                print(f"\tClip by values: {param['clip_grad']}")
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [
                (tf.clip_by_value(
                    grad, - param["clip_grad"], param["clip_grad"]), var)
                for grad, var in gvs
            ]
            train = optimizer.apply_gradients(capped_gvs)


    # TODO: remove this
    tb_dir = param["tensorboard_path"]

    start = datetime.now()
    
    predictions = dict()
    if verbose:
        print("Running training session...")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged_summary = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter(
            param["tensorboard_path"] + "/train")

        val_writer = tf.summary.FileWriter(
            param["tensorboard_path"] + "/validation")
        train_writer.add_graph(sess.graph)

        # val_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        print("Training model...")
        for e in range(param["epochs"]):
            # Use this if train with batches
            # for X_batch, y_batch in zip(X_batches, y_batches):
            #     sess.run(train, feed_dict={X: X_batch, y: y_batch})

            sess.run(train, feed_dict={X: data["X_train"], y: data["y_train"]})

            if e % param["report_periods"] == 0:
                # In those periods, training summary is written to tensorboard record.
                # Summary on training set.
                train_summary = sess.run(
                    merged_summary,
                    feed_dict={X: data["X_train"], y: data["y_train"]}
                )
                # Summary on validation set.
                val_summary = sess.run(
                    merged_summary,
                    feed_dict={X: data["X_val"], y: data["y_val"]}
                )

                train_writer.add_summary(train_summary, e)
                val_writer.add_summary(val_summary, e)

            if e % (param["report_periods"] * 10) == 0 and verbose:
                # print out training result 10 times less frequently than the record frequency.
                train_mse = loss.eval(
                    feed_dict={X: data["X_train"], y: data["y_train"]}
                )
                val_mse = loss.eval(
                    feed_dict={X: data["X_val"], y: data["y_val"]}
                )

                print(
                    f"\nIteration [{e}], Training MSE {train_mse:0.7f}; Validation MSE {val_mse:0.7f}")
            

            if e in prediction_checkpoints:
                predictions[e] = make_prediction_all(pred, X, data)

        if -1 in prediction_checkpoints:
            predictions[param["epochs"]] = make_prediction_all(pred, X, data)

        print("Saving the model...")
        saver.save(sess, param["model_path"])
    print(f"Time taken for [{param['epochs']}] epochs: ", datetime.now() - start)
    print("Final result:")
    metric_test = metrics.merged_scores(
        actual=pd.DataFrame(data["y_test"]),
        pred=pd.DataFrame(list(predictions.values())[-1]["test"]),
        verbose=True
    )
    return (metric_test, predictions)


def restore_model(
    parameters: Dict[str, object],
    data_collection: Dict[str, np.ndarray],
    prediction_checkpoints: Iterable[int] = [-1],
    verbose: bool = False
) -> Tuple[
        Dict[str, float],
        Dict[int, Dict[str, np.ndarray]]
]:
    raise NotImplementedError()
