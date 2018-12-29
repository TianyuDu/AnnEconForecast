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

import core.models.generic_rnn as generic_rnn


class StackedLSTM(generic_rnn.GenericRNN):
    """
    The stacked (multi-layer) long short term memory object.
    """

    def __init__(
        self,
        param: Dict[str, object],
        prediction_checkpoints: Iterable[int] = [-1],
        verbose: bool = True
    ) -> None:
        super().__init__(param, prediction_checkpoints, verbose)
        self.build()

    def build(
        self
    ) -> None:
        """
        Build the computational graph.
        """
        if self.verbose:
            print("Building the computational graph...")
        self._build_data_io()
        self._build_recurrent()
        self._build_output_layer()
        self._build_metrics()
        self._build_optimizer()
        if self.verbose:
            print("The graph is built.")

    def _build_data_io(self) -> None:
        """
        A helper func. building the data IO tensors.
        """
        if self.verbose:
            print("Building data IO tensors...")
        # IO nodes handling dataset.
        with tf.name_scope("DATA_IO"):
            self.X = tf.placeholder(
                tf.float32,
                [None, self.param["num_time_steps"], self.param["num_inputs"]],
                name="FEATURE")
            if self.verbose:
                print(
                    f"Feature(input) tensor is built, shape={str(self.X.shape)}")

            self.y = tf.placeholder(
                tf.float32,
                [None, self.param["num_outputs"]],
                name="LABEL")
            if self.verbose:
                print(
                    f"Label(output) tensor is built, shape={str(self.y.shape)}")

    def _build_recurrent(self) -> None:
        """
        A helper func. building the recurrent part of network.
        """
        if self.verbose:
            print("Building the recurrent structure...")

        # TODO: add customized activation functions.
        self.multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(
                num_units=units,
                name=f"LSTM_LAYER_{i}")
                for i, units in enumerate(self.param["num_neurons"])
             ])
        if self.verbose:
            print(
                "Multi-layer LSTM structure is built: neurons={self.param['num_neurons']}.")

        # rnn_outputs.shape is (None, num_time_steps, num_neurons[-1])
        self.rnn_outputs, self.states = tf.nn.dynamic_rnn(
            self.multi_cell,
            self.X,
            dtype=tf.float32)

        if self.verbose:
            print(
                f"(dynamic_rnn) rnn_outputs shape={str(self.rnn_outputs.shape)}.")
            print(f"(dynamic_rnn) states shape={str(self.states)}.")

        # Stack everything together.
        self.stacked_output = tf.reshape(
            self.rnn_outputs,
            [-1,  # equivalently, put None as the first element in shape.
             self.param["num_time_steps"] * self.param["num_neurons"][-1]]
        )
        if self.verbose:
            print(
                f"Recurrent structure is built, the stacked output shape={str(self.stacked_output.shape)}")

    def _build_output_layer(self) -> None:
        """
        A helper func. building the output part of the network.
        """
        if self.verbose:
            print("Building the output layer...")
        with tf.name_scope("OUTPUT_LAYER"):
            # Transform each stacked RNN output to a single real value.
            self.W = tf.Varaible(
                tf.random_normal(
                    [self.param["num_time_steps"] * self.param["num_neurons"][-1],
                    1]
                ),
                dtype=tf.float32,
                name="OUTPUT_WEIGHT"
            )
            if self.verbose:
                print(
                    f"Output weight tensor is built, shape={str(self.W.shape)}")

            self.b = tf.Variable(
                tf.random_normal([1]),
                dtype=tf.float32,
                name="OUTPUT_BIAS"
            )
            if self.verbose:
                print(
                    f"Output bias tensor is built, shape={str(self.b.shape)}")

            self.pred = tf.add(
                tf.matmul(self.stacked_output),
                self.b,
                name="PREDICTION"
            )
            if self.verbose:
                print(
                    f"Prediction tensor is built, shape={str(self.pred.shape)}")

            # Tensorboard summary monitor.
            tf.summary.histogram("output_weights", self.W)
            tf.summary.histogram("output_biases", self.b)
            tf.summary.histogram("predictions", self.pred)
            if self.verbose:
                print("\fSummaries on tensors are added to tensorboard.")

    def _build_metrics(self) -> None:
        """
        A helper func. building the performance metrics.
        """
        if self.verbose:
            print("Building model preformance metrics...")
        with tf.name_scope("METRICS"):
            # MSE is the main loss we focus on
            # and it's the metric used for optimization.
            # so we just name the MSE using 'loss'
            self.loss = tf.losses.mean_squared_error(
                labels=self.y,
                predictions=self.pred
            )

            self.rmse = tf.sqrt(
                tf.reduce_mean(tf.square(self.y - self.pred)),
                name="RMSE"
            )

            self.mape = tf.reduce_mean(
                tf.abs(tf.divide(self.y - self.pred, self.y)),
                name="MAPE"
            )
            if self.verbose:
                print("\tLoss tensors are built.")

            tf.summary.scalar("MSE", self.loss)
            tf.summary.scalar("RMSE", self.rmse)
            tf.summary.scalar("MAPE", self.mape)
            if self.verbose:
                print("\tSummaries on losses are added to tensorbard.")

    def _build_optimizer(self) -> None:
        """
        A helper func. building the optimizer in neural network.
        """
        if self.verbose:
            print("Building the optimizer...")
        with tf.name_scope("OPTIMIZER"):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.param["learning_rate"],
                name="OPTIMIZER"
            )

            # Applying Gradient Clipping, if requested.
            if self.param["clip_grad"] is None:
                # No G.C.
                if self.verbose:
                    print("\tNote: no gradient clipping is applied.\
                    \n\tIf possible gradient exploding detected (e.g. nan loss), \
                    try use clip_grad.")
                self.train = self.optimizer.minimize(self.loss)
            else:
                # Apply G.C.
                assert type(self.param["clip_grad"]) in [float, int]\
                    and self.parm["clip_grad"] > 0,\
                    "Gradient Clipping should be either float or integer and greater than zero."

                # NOTE: I didnot write the graident clipping code, use this function carefully.
                if self.verbose:
                    print("\tApplying gradient clipping...")
                    print(f"\tClip by values: {self.param['clip_grad']}")
                gvs = self.optimizer.compute_gradients(self.loss)
                capped_gvs = [
                    (tf.clip_by_value(
                        grad, - self.param["clip_grad"], self.param["clip_grad"]), var
                     )
                    for grad, var in gvs
                ]
                self.train = self.optimizer.apply_gradients(capped_gvs)

        if self.verbose:
            print("\tThe complete computational graph is built.")

    def train(
        self,
        data: Dict[str, np.ndarray],
        ret: Union[None, List[str], "all"] = None,
        save_to_disk: bool = False
    ) -> Union[None, Dict[str, float, np.ndarray]]:
        """
        Args:
            data:
                A dictionary of dataset containing the following keys.
                    - "X_train" = Training set feature matrix.
                    - "y_train" = Training set label matrix.
                    - "X_val" = Validation set feature matrix.
                    - "y_val" = Validation set label matrix.
                And each value should be a numpy array with following shapes.
                    - "X_*": shape = [*, num_time_steps, num_inputs]
                    - "y_*": shape = [*, num_outputs]
                NOTE: to guarantee the isolation of testing set, it must not
                be included in data dictionary.
            ret:
                - None if no return is needed.
                - Put a list of strings of metrics that one wish to be returned
                from this training session.
                - Also if prediction on training set or validation set is wanted.
                (even if only one str is passed, use singlton in this case)
                NOTE: string format: *_set where set in {train, val, all}
                - if "all" is passed in, all records will be packed in a dictionary
                and returned.
            save_to_disk:
                A bool denotes if the model file (tensorboard, saved model etc)
                will be written to local disk after training is completed.
        Returns:
            If ret is not None, a dictionary with keys from ret and the corresponding
            numerical(float) values of specified metrics.
            Also if prediction on training set or validation set is wanted. 
        """
        # ======== Argument Checking ========
        assert all(
            isinstance(key, str) and isinstance(value, np.ndarray)
            for (key, value) in data.items()
        ), "All keys in data dictionary should be string, and all values should be np.ndarray."

        assert sorted(list(data.keys())) \
        == sorted(["X_train", "y_train", "X_val", "y_val"]),\
        "Data dictionary must have and only has the following keys: X_train, y_train, X_val and y_val."
        
        assert ret is None\
        or ret == "all"\
        or all(isinstance(x, str) for x in ret),\
        "ret should be one of None, 'all', or a list of strings."
        # ======== End ========
        if ret == "full":
            ret = ["pred_all", "mse_train", "mse_val"]
        
        # Record training cost.
        start = datetime.now()

        # Return package, as performance measures.
        pred_all = dict()
        mse_train = dict()
        mse_val = dict()

        if self.verbose:
            print(f"Starting training session, for {self.param['epochs']} epochs.")
        
        with tf.Session() as sess:
            # FIXME: this might not work in an isolated objects for tf.
            saver = tf.train.Saver()
            merged_summary = tf.summary.merge_all()

            if self.verbose:
                print("Creating tensorboard file writers,\
                \nwriting to path {self.param['tensorboard_path']}")
            
            train_writer = tf.summary.FileWriter(
                self.param["tensorboard_path"] + "/train")
            val_writer = tf.summary.FileWriter(
                self.param["tensorboard_path"] + "/validation")
            train_writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())

            if self.verbose:
                print("Training model...")
            
            for e in range(self.param["epochs"]):
                # Consider using mini batches.
                # Use this if train with batches
                # for X_batch, y_batch in zip(X_batches, y_batches):
                #     sess.run(train, feed_dict={X: X_batch, y: y_batch})
                sess.run(
                    self.train,
                    feed_dict={self.X: data["X_train"], self.y: data["y_train"]}
                )

                if e % self.param["report_periods"] == 0:
                    # In those periods, training summary is written to tensorboard record.
                    # Summary on training set.
                    train_summary = sess.run(
                        merged_summary,
                        feed_dict={self.X: data["X_train"], self.y: data["y_train"]}
                    )
                    # Summary on validation set.
                    val_summary = sess.run(
                        merged_summary,
                        feed_dict={self.X: data["X_val"], self.y: data["y_val"]}
                    )

                    train_writer.add_summary(train_summary, e)
                    val_writer.add_summary(val_summary, e)

                if e % (self.param["report_periods"] * 10) == 0 and self.verbose:
                    # print out training result 10 times less frequently than the record frequency.
                    train_mse = self.loss.eval(
                        feed_dict={self.X: data["X_train"],
                                   self.y: data["y_train"]}
                    )
                    val_mse = self.loss.eval(
                        feed_dict={self.X: data["X_val"],
                                   self.y: data["y_val"]}
                    )

                    print(
                        f"\nIteration [{e}], Training MSE {train_mse:0.7f}; \
                        Validation MSE {val_mse:0.7f}")
            
                if e in self.ckpts:
                    pred_all[e] = make_predictions(
                        predictor=self.pred,
                        X=self.X,
                        data=data
                    )
                    mse_train[e] = self.loss.eval(
                        feed_dict={self.X: data["X_train"],
                                   self.y: data["y_train"]}
                    )
                    mse_val[e] = self.loss.eval(
                        feed_dict={self.X: data["X_val"],
                                   self.y: data["y_val"]}
                    )
                    assert isinstance(mse_train[e], float)\
                        and isinstance(mse_val[e], float)

            if -1 in self.ckpts:
                # If the final prediction is required.
                pred_all[self.param["epochs"]] = make_predictions(
                    predictor=self.pred,
                    X=self.X,
                    data=data
                )
            if self.verbose:
                print("Saving the model...")
            
            saver.save(sess, self.param["model_path"])
        
        if self.verbose:
            print(f"Time taken for [{param['epochs']}] epochs: ",\
            datetime.now() - start)

        if ret is not None:
            # Return the pack of records.
            ret_pack = dict()
            for var in ret:
                try:
                    exec(f"ret_pack['{var}'] = {var}")
                except NameError:
                    print(f"{var} is not a valid record variable, ignored.")
            return ret_pack

    def exec_core(self):
        raise NotImplementedError()

def make_predictions(
    predictor: tf.Tensor,
    X: tf.Tensor,
    data: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    p_train = predictor.eval(feed_dict={X: data["X_train"]})
    p_val = predictor.eval(feed_dict={X: data["X_val"]})
    result = {
        "train": p_train,
        "val": p_val
    }
    return result


def exec_core(
    param: Dict[str, object],
    data: Dict[str, np.ndarray],
    prediction_checkpoints: Iterable[int] = [-1],
    verbose: bool = True
) -> Dict[int, Dict[str, np.ndarray]]:
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
            tf.matmul(stacked_output, W),
            b,
            name="PREDICTION"
        )

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("predictions", pred)

    # pred = tf.layers.dense(stacked_output, 1)

    with tf.name_scope("METRICS"):
        # MSE is the main loss we focus on and it's the metric used for optimization.
        loss = tf.reduce_mean(tf.square(y - pred), name="MSE")
        mape = tf.reduce_mean(tf.abs(tf.divide(y - pred, y)), name="MAPE")

        tf.summary.scalar("MSE", loss)
        tf.summary.scalar("MAPE", mape)

    with tf.name_scope("OPTIMIZER"):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=param["learning_rate"],
            name="ADAM_OPTIMIZER"
        )

        if param["clip_grad"] is None:
            # No Gradient Clipping
            print("Note: no gradient clipping is applied.\
            \nIf possible gradient exploding detected (e.g. nan loss), \
            try use clip_grad.")
            train = optimizer.minimize(loss)
        else:
            # With Gradient Clipping
            print("Applying gradient clipping...")
            print(f"\tClip by values: {param['clip_grad']}")
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [
                (tf.clip_by_value(
                    grad, - param["clip_grad"], param["clip_grad"]), var)
                for grad, var in gvs
            ]
            train = optimizer.apply_gradients(capped_gvs)

    start = datetime.now()
    predictions = dict()

    print("Starting training session...")

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
                predictions[e] = make_predictions(pred, X, data)

        if -1 in prediction_checkpoints:
            predictions[param["epochs"]] = make_predictions(pred, X, data)

        print("Saving the model...")
        saver.save(sess, param["model_path"])

    print(f"Time taken for [{param['epochs']}] epochs: ",\
    datetime.now() - start)

    return predictions


# TODO: consider if to drop this method, it's already implemented in a jupyter notebook.
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
