"""
The default hyper-parameter searching program.
This is a control script.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from typing import Dict, List

import sys
sys.path.append(".../")

import constants
from core.tools.metrics import *
from core.tools.visualize import *
from core.tools.time_series import *
from core.tools.data_import import *
import core.tools.rnn_prepare as rnn_prepare

import core.models.stacked_lstm as stacked_lstm

import hps_methods

# data preparation phase.
pprint(constants.DATA_DIR)
choice = None
while choice is None or choice not in constants.DATA_DIR.keys():
    if choice is not None:
        print("Invalid data location received, try again...")
    choice = input("Select Dataset >>> ")
FILE_DIR = constants.DATA_DIR[choice]

print(f"Dataset chosen: \n{FILE_DIR}")

config_name = input("Name of configuration file to load >>> ")

exec(f"import core.training.configs.{config_name} as config")

for att in dir(config):
    if att.endswith("_config"):
        print(f"Loading: {att}")
        exec(f"globals().update")


parameter_collection = hps_methods.gen_para_set(training_config)

for para in parameter_collection:
    prepared_df = rnn_prepare.prepare_dataset(
        file_dir=FILE_DIR,
        periods=PERIODS,
        order=ORDER,
        remove=None
    )
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = rnn_prepare.generate_splited_dataset(
        raw=prepared_df,
        train_ratio=0.8,
        val_ratio=0.1,
        lags=para["num_time_steps"]
    )
    data_collection = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    def checkpoints(z): return [
        z*x for x in range(1, parameters["epochs"] // z)] + [-1]
    
    (metrics_dict, predictions) = stacked_lstm.exec_core(
        parameters=parameters,
        data_collection=data_collection,
        clip_grad=None,
        prediction_checkpoints=checkpoints(
            parameters["epochs"] // 10
        )
    )
    
