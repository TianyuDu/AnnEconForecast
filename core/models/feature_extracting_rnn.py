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

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

parameters = {
    "num_time_steps": 36,
    "num_inputs": 1,
    "num_outputs": 36,
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
        print("Model: loading parameters")
        for para_name, value in zip(para.keys(), para.values()):
            exec(f"self.{para_name} = {value}")

class ExtractingBaselineRnn(Model):
    def __init__(
        self,
        para: Dict[str, object],
        sequential_label: bool=True
    ) -> None:
        self.SL = sequential_label
        self.read_parameters(para)
        tf.reset_default_graph()
        self.build_placeholders()
        self.build_rnn()
        self.build_training()

    def build_placeholders(
        self) -> None:
        print("Building placeholders...")
        self.X = tf.placeholder(
            tf.float32,
            [None, self.num_time_steps, self.num_inputs],
            name="Input_placeholder")

        if self.SL:
            TS = self.num_time_steps
        else:
            TS = 1
            
        self.y = tf.placeholder(
            tf.float32,
            [None, self.num_outputs, self.num_inputs],
            name="Output_placeholder")

    def build_rnn(self) -> None:
        print("Building core rnn...")
        self.cell = tf.contrib.rnn.LSTMCell(
            num_units=self.num_neurons,
            activation=tf.nn.relu
        )
        
        self.rnn_output, self.states = tf.nn.dynamic_rnn(
            self.cell, self.X, dtype=tf.float32)
        if self.SL:
            self.outputs = tf.layers.dense(self.rnn_output, self.num_outputs)
        else:  # Single time period prediction based on the LAST output only.
            self.outputs = tf.layers.dense(self.rnn_output[:, -1, :], self.num_outputs)
    
    def build_training(self) -> None:
        print("Building metrics and operations...")
        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        
        
class BaselineDNN(Model):
    def __init__(
        self,
        para: Dict[str, object],
    ) -> None:
        self.read_parameters(para)
        tf.reset_default_graph()
        self.build_placeholders()
        self.build_dnn()
        self.build_training()

    def build_placeholders(
        self) -> None:
        print("Building placeholders...")
        self.X = tf.placeholder(
            tf.float32,
            [None, self.num_time_steps],
            name="Input_placeholder")

        TS = self.num_time_steps
            
        self.y = tf.placeholder(
            tf.float32,
            [None, self.num_outputs],
            name="Output_placeholder")

    def build_dnn(self) -> None:
        print("Building core dnn...")
        
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([self.num_time_steps, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

        hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, self.num_outputs])),
                      'biases':tf.Variable(tf.random_normal([self.num_outputs]))}
        
        l1 = tf.add(tf.matmul(self.X ,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
        self.outputs = output
    
    def build_training(self) -> None:
        print("Building metrics and operations...")
        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
