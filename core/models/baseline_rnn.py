"""
The baseline recurrent neural network models.
"""
from typing import Dict

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from core.tools.data_import import *
from core.tools.time_series import *

parameters = {
    "num_time_steps": 24,
    "num_inputs": 1,
    "num_outputs": 1,
    "num_neurons": 64,
    "learning_rate": 0.1
}


class Model:
    def __init__(self) -> None:
        pass

    def read_parameters(
        self,
        para: Dict[str, object]
    ) -> None:
        print("BaselineRnn: loading parameters")
        for para_name, value in zip(para.keys(), para.values()):
            exec(f"self.{para_name} = {value}")


class BaselineRnn(Model):
    def __init__(
        self,
        para: Dict[str, object],
        sequential_label: bool=True
    ) -> None:
        self.read_parameters(para)
        self.build_placeholders(sequential_label)
        self.build_rnn()
        self.build_training()

    def build_placeholders(
        self,
        sequential_label: bool=True
    ) -> None:
        print("Building placeholders...")
        self.X = tf.placeholder(
            tf.float32,
            [None, self.num_time_steps, self.num_inputs],
            name="Input_placeholder")

        nts = {True: self.num_time_steps, False: 1}
        self.y = tf.placeholder(
            tf.float32,
            [None, nts[sequential_label], self.num_outputs],
            name="Output_placeholder")

    def build_rnn(self) -> None:
        print("Building core rnn...")
        self.cell = tf.contrib.rnn.LSTMCell(
            num_units=self.num_neurons
        )
        self.rnn_output, self.states = tf.nn.dynamic_rnn(
            self.cell, self.X, dtype=tf.float32)
        self.outputs = tf.layers.dense(self.rnn_output, self.num_outputs)
    
    def build_training(self) -> None:
        print("Building metrics and operations...")
        self.loss = tf.reduce_mean(
            tf.square(self.outputs, self.y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
