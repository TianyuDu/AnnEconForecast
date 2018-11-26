"""
Models.
"""
import datetime
import numpy as np
import pandas as pd
import keras
import os


class BaseModel():
    def __init__(self):
        self.core = None
        self.container = None
        self.config = None
        self._gen_file_name()
    
    def _gen_file_name(self):
        """
        Generate the directory name to save all relevant files about
        Graphic representation of model,
        Model structure()
        """
        now = datetime.datetime.now()
        self.file_name = now.strftime("%Y%h%d_%H_%M_%s")

    def __str__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model at {hex(id(self))}
        """
    
    def __repr__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model with data container {self.container}
        """

# # TODO: Fix the univariate LSTM
# class UnivariateLSTM(BaseModel):
#     """
#     Univariate LSTM model with customized num of layers.
#     """
#     def __init__(
#         self, 
#         container: containers.UnivariateContainer,
#         config: dict={
#             "batch_size": 1,
#             "epoch": 10,
#             "neuron": [128]}):
#         self.config = config
#         self.container = container
#         self.core = self._construct_lstm()

#     def _construct_lstm(self):
#         core = keras.Sequential()
#         num_lstm_lys = len(self.config["neuron"])

#         batch_size = self.config["batch_size"]
#         neuron_units = self.config["neuron"]

#         core.add(
#             keras.layers.LSTM(
#                 units=neuron_units[0],
#                 batch_input_shape=(batch_size, 1, self.container.num_fea),
#                 stateful=True,
#                 name="lstm_layer_0_input"
#         ))

#         # TODO: deal with multiple LSTM layer issue
#         for i in range(1, num_lstm_lys):
#             core.add(
#                 keras.layers.LSTM(
#                     units=neuron_units[i],
#                     stateful=True,
#                     name=f"lstm_layer_{i}"
#             ))
        
#         core.add(keras.layers.Dense(
#             units=1,
#             name="dense_output"
#         ))
#         core.compile(
#             loss="mean_squared_error",
#             optimizer="adam"
#         )
#         return core

#     def fit_model(self):
#         pass


class MultivariateCnnLSTM(BaseModel):
    def __init__(self):
        pass
